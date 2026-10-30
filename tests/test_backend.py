"""Run with:  python -m pytest tests"""
import os
import tempfile

# must be set before backend modules are imported
os.environ["PLACEIQ_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["PLACEIQ_GRADER"] = "keyword"
os.environ["PLACEIQ_WARMUP"] = "0"

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


def test_ollama_grader_scores_from_point_checks_and_falls_back(monkeypatch):
    import ollama

    from backend import grading

    class FakeClient:
        reply = ('{"checks": [{"point": 1, "candidate_said": "last in first out", "covered": true},'
                 ' {"point": 2, "candidate_said": "nothing", "covered": true},'
                 ' {"point": 3, "candidate_said": "you pop it", "covered": true}], "feedback": "Good start."}')

        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            if FakeClient.reply is None:
                raise ConnectionError("Ollama is not running")
            return ollama.ChatResponse(message=ollama.Message(role="assistant", content=FakeClient.reply))

    monkeypatch.setenv("PLACEIQ_GRADER", "ollama")
    monkeypatch.setattr(grading.ollama, "Client", FakeClient)
    result = grade("q", "ref", ["LIFO", "push", "pop"], "It is last in first out and you pop it off")
    # point 2 is marked covered but with no evidence from the answer, so it does not count
    assert result == {"score": 0.67, "feedback": "Good start.", "missing_points": ["push"], "grader": "ollama"}

    FakeClient.reply = None
    assert grade("q", "ref", ["LIFO", "push", "pop"], "A stack is LIFO with push and pop")["grader"] == "keyword"


def test_gemini_grader_scores_from_point_checks_and_falls_back(monkeypatch):
    from google import genai

    class FakeModels:
        reply = ('{"checks": [{"point": 1, "candidate_said": "last in first out", "covered": true},'
                 ' {"point": 2, "candidate_said": "nothing", "covered": true},'
                 ' {"point": 3, "candidate_said": "you pop it", "covered": true}], "feedback": "Good start."}')

        def generate_content(self, model, contents, config):
            assert "<candidate_answer>" in contents and config.temperature == 0
            if FakeModels.reply is None:
                raise ConnectionError("no network")
            return type("Response", (), {"text": FakeModels.reply})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setenv("PLACEIQ_GRADER", "gemini")
    monkeypatch.setattr(genai, "Client", FakeClient)
    result = grade("q", "ref", ["LIFO", "push", "pop"], "It is last in first out and you pop it off")
    # point 2 is marked covered but with no evidence from the answer, so it does not count
    assert result == {"score": 0.67, "feedback": "Good start.", "missing_points": ["push"], "grader": "gemini"}

    FakeModels.reply = None
    assert grade("q", "ref", ["LIFO", "push", "pop"], "A stack is LIFO with push and pop")["grader"] == "keyword"


def test_gemini_is_the_default_only_when_a_key_is_set(monkeypatch):
    from backend.grading import grader_name

    monkeypatch.delenv("PLACEIQ_GRADER")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert grader_name() == "keyword"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert grader_name() == "gemini"
    monkeypatch.setenv("PLACEIQ_GRADER", "keyword")
    assert grader_name() == "keyword"


def test_weak_concepts_are_asked_more():
    mastery = {"DSA": 0.9, "OS": 0.1}
    rng = random.Random(0)
    picks = [selection.pick_concept(mastery, rng) for _ in range(500)]
    assert picks.count("OS") > 5 * picks.count("DSA")


def test_default_mastery_comes_from_session_scores(monkeypatch):
    monkeypatch.delenv("PLACEIQ_USE_DKT", raising=False)
    mastery = selection.score_mastery([("OS", 0.0), ("OS", 0.0), ("DSA", 1.0)])
    assert mastery["OS"] == round(0.5 / 3, 4) and mastery["DSA"] == 0.75 and mastery["CN"] == 0.5
    # earlier sessions are ignored: only the current session's scores steer the choice
    rng = random.Random(0)
    picks = [selection.next_question([("OS", 1.0)] * 20, [("OS", 0.0), ("OS", 0.0)], set(), set(), rng)[0]["concept"]
             for _ in range(300)]
    assert picks.count("OS") > 2 * picks.count("DSA")


def test_dkt_is_used_only_when_switched_on(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("ml.predict")
    fake.predict_mastery = lambda history: {c: 0.123 for c in selection.concepts()}
    monkeypatch.setitem(sys.modules, "ml.predict", fake)
    monkeypatch.delenv("PLACEIQ_USE_DKT", raising=False)
    assert selection.mastery_for([("OS", 1.0)])["OS"] == 0.75
    monkeypatch.setenv("PLACEIQ_USE_DKT", "1")
    assert selection.mastery_for([("OS", 1.0)])["OS"] == 0.123


def test_difficulty_follows_mastery():
    assert [selection.target_difficulty(p) for p in (0.2, 0.5, 0.9)] == [1, 2, 3]


def test_question_is_not_repeated_in_session():
    asked = set()
    for _ in range(17):  # there are 17 DSA questions
        asked.add(selection.pick_question("DSA", 2, asked, set())["id"])
    assert len(asked) == 17
    assert selection.pick_question("DSA", 2, asked, set()) is None


def test_company_session_uses_generated_question_and_falls_back(monkeypatch):
    import rag.generate

    generated = {"id": "gen-test", "concept": "OS", "difficulty": 2, "question": "A company-style question?",
                 "reference_answer": "The reference.", "key_points": ["alpha", "beta"]}
    with TestClient(app) as client:
        assert "Amazon" in client.get("/companies").json()
        # each question is generated in the background as soon as the session starts or an
        # answer is graded, so the generator is patched before those calls
        monkeypatch.setattr(rag.generate, "generate_question", lambda *args: generated)
        session_id = client.post(
            "/sessions", json={"user_id": "company-tester", "company": "Amazon", "num_questions": 2}).json()["session_id"]
        q = client.get(f"/sessions/{session_id}/next").json()["question"]
        assert q["source"] == "company" and q["question"] == "A company-style question?"
        # when generation is unavailable the bank question is used
        monkeypatch.setattr(rag.generate, "generate_question", lambda *args: None)
        r = client.post(f"/sessions/{session_id}/answers", json={
            "session_question_id": q["session_question_id"], "answer_text": "it involves alpha and beta"})
        assert r.json()["score"] == 1.0
        assert client.get(f"/sessions/{session_id}/next").json()["question"]["source"] == "bank"

        notes = client.get("/companies/Amazon/experiences", params={"concepts": "OS,DBMS"}).json()
        assert len(notes) == 3 and all(n["company"] == "Amazon" for n in notes)
        assert client.get("/companies/Nowhere/experiences", params={"concepts": "OS"}).status_code == 404


def test_simultaneous_next_requests_get_the_same_question(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import rag.generate

    def slow_generate(company, concept, difficulty, exclude):
        time.sleep(0.3)
        return {"id": f"gen-{time.time_ns()}", "concept": concept, "difficulty": difficulty,
                "question": f"Question {time.time_ns()}?", "reference_answer": "The reference.", "key_points": ["alpha"]}

    monkeypatch.setattr(rag.generate, "generate_question", slow_generate)
    with TestClient(app) as client:
        session_id = client.post(
            "/sessions", json={"user_id": "double-tester", "company": "Amazon", "num_questions": 3}).json()["session_id"]
        with ThreadPoolExecutor(3) as pool:
            replies = list(pool.map(lambda _: client.get(f"/sessions/{session_id}/next").json()["question"], range(3)))
        assert len({q["session_question_id"] for q in replies}) == 1
        assert all(q["number"] == 1 for q in replies)


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
