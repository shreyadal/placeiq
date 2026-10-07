"""Run with:  python -m pytest tests"""
import os
import tempfile

# must be set before backend modules are imported
os.environ["PLACEIQ_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["PLACEIQ_GRADER"] = "keyword"

import random

from fastapi.testclient import TestClient

from backend import selection
from backend.grading import grade
from backend.main import app
from backend.speech import count_fillers, speech_metrics


def test_speech_metrics():
    # "um so the stack is [1.5 s pause] like a pile", spoken from t=1.0 to t=7.0
    spoken = [("Um,", 1.0, 1.3), ("so", 1.4, 1.6), ("the", 1.7, 1.8), ("stack", 1.9, 2.3), ("is", 2.4, 2.5),
              ("like", 4.0, 4.3), ("a", 4.4, 4.5), ("pile.", 6.6, 7.0)]
    words = [{"word": w, "start": s, "end": e} for w, s, e in spoken]
    m = speech_metrics(words, duration=9.0)
    assert m["word_count"] == 8
    assert m["speaking_seconds"] == 6.0
    assert m["words_per_minute"] == 80  # 8 words in 6 seconds
    assert m["fillers"] == {"um": 1, "like": 1}
    assert m["filler_rate"] == 25.0
    assert m["pause_count"] == 2  # 2.5 -> 4.0 and 4.5 -> 6.6
    assert m["longest_pause_seconds"] == 2.1
    assert speech_metrics([], 3.0)["words_per_minute"] == 0


def test_filler_phrases():
    assert count_fillers("you know i mean it is kind of fast".split()) == {"you know": 1, "i mean": 1, "kind of": 1}


def test_keyword_grader():
    key_points = ["LIFO", "push", "pop"]
    full = grade("q", "ref", key_points, "A stack is LIFO, you push and pop at the top")
    partial = grade("q", "ref", key_points, "It is a LIFO data structure I think")
    assert full["score"] == 1.0 and full["missing_points"] == []
    assert partial["score"] < full["score"] and "push" in partial["missing_points"]
    assert grade("q", "ref", key_points, "")["score"] == 0.0


def test_weak_concepts_are_asked_more():
    mastery = {"DSA": 0.9, "OS": 0.1}
    rng = random.Random(0)
    picks = [selection.pick_concept(mastery, rng) for _ in range(500)]
    assert picks.count("OS") > 5 * picks.count("DSA")


def test_difficulty_follows_mastery():
    assert [selection.target_difficulty(p) for p in (0.2, 0.5, 0.9)] == [1, 2, 3]


def test_question_is_not_repeated_in_session():
    asked = set()
    for _ in range(17):  # there are 17 DSA questions
        asked.add(selection.pick_question("DSA", 2, asked, set())["id"])
    assert len(asked) == 17
    assert selection.pick_question("DSA", 2, asked, set()) is None


def test_full_session():
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={"user_id": "tester", "num_questions": 3}).json()["session_id"]

        first = client.get(f"/sessions/{session_id}/next").json()["question"]
        assert "reference_answer" not in first
        # asking again before answering returns the same question
        assert client.get(f"/sessions/{session_id}/next").json()["question"] == first

        seen = []
        for number in (1, 2, 3):
            q = client.get(f"/sessions/{session_id}/next").json()["question"]
            assert q["number"] == number and q["question_id"] not in seen
            seen.append(q["question_id"])
            body = {"session_question_id": q["session_question_id"], "answer_text": "I am not sure about this one",
                    "speech": {"words_per_minute": 120 + number, "filler_rate": 2.0, "filler_count": 1,
                               "longest_pause_seconds": 1.5},
                    "vision": {"eye_contact_percent": 70, "head_movement": 0.1, "blinks_per_minute": 15}}
            r = client.post(f"/sessions/{session_id}/answers", json=body)
            assert r.status_code == 200 and 0 <= r.json()["score"] <= 1
            # the same question cannot be answered twice
            assert client.post(f"/sessions/{session_id}/answers", json=body).status_code == 409

        assert client.get(f"/sessions/{session_id}/next").json()["done"] is True

        summary = client.post(f"/sessions/{session_id}/end").json()
        assert summary["answered"] == 3
        assert summary["speech"]["words_per_minute"] == 122
        assert summary["vision"]["eye_contact_percent"] == 70
        assert set(summary["mastery"]) == set(selection.concepts())
        assert client.get(f"/sessions/{session_id}/next").status_code == 409

        progress = client.get("/users/tester/progress").json()
        assert len(progress["sessions"]) == 1 and len(progress["weak_concepts"]) == 3

        assert client.get("/sessions/999/next").status_code == 404
