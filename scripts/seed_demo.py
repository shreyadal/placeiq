"""Fill the app with one simulated candidate so the results page has something to show.

    python scripts/seed_demo.py --user demo --sessions 5

Talks to a running backend over HTTP, exactly like the frontend does. The candidate is
strong in some concepts and weak in others, and improves a little each session. The
speech and camera numbers are made up; real ones come from the interview page.
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

QUESTIONS = {q["id"]: q for q in json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "questions.json").read_text(encoding="utf8"))}

# chance of giving a full answer, per concept, in the first session
SKILL = {"DSA": 0.85, "OOP": 0.75, "ML": 0.6, "CN": 0.5, "DBMS": 0.3, "OS": 0.2}


def call(base, path, body=None):
    request = urllib.request.Request(
        base + path, method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--user", default="demo")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--questions", type=int, default=8)
    args = parser.parse_args()
    rng = random.Random(1)

    for s in range(args.sessions):
        session_id = call(args.api, "/sessions", {"user_id": args.user, "num_questions": args.questions})["session_id"]
        while True:
            question = call(args.api, f"/sessions/{session_id}/next")["question"]
            if question is None:
                break
            key_points = QUESTIONS[question["question_id"]]["key_points"]
            p_good = min(0.95, SKILL[question["concept"]] + 0.07 * s)
            if rng.random() < p_good:
                answer = "The main points are " + ", ".join(key_points) + "."
            else:
                answer = "I am not fully sure, but I think it relates to " + key_points[0] + "."
            call(args.api, f"/sessions/{session_id}/answers", {
                "session_question_id": question["session_question_id"],
                "answer_text": answer,
                "speech": {"words_per_minute": rng.randint(125, 165), "filler_rate": round(rng.uniform(1, 6), 1),
                           "filler_count": rng.randint(0, 5), "longest_pause_seconds": round(rng.uniform(0.8, 2.5), 2)},
                "vision": {"eye_contact_percent": rng.randint(55, 85), "head_movement": round(rng.uniform(2, 7), 1),
                           "blinks_per_minute": round(rng.uniform(10, 22), 1)},
            })
        summary = call(args.api, f"/sessions/{session_id}/end", {})
        print(f"session {session_id}: average score {summary['average_score']}, weak: {summary['weak_concepts']}")


if __name__ == "__main__":
    main()
