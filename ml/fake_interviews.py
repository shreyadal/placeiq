"""Generate fake interview logs so the pipeline can be tested before real users exist.

    python -m ml.fake_interviews --users 400

Each fake candidate gets a hidden ability per concept. The chance of a good answer rises
with ability and falls with question difficulty, and ability grows a little with every
attempt. A model that learns anything should therefore beat 0.5 AUC on this data, and
should rank a candidate's truly weak concepts lowest.
"""
import argparse
import csv
import json
import math
import random

from ml.common import DATA_DIR

LOG_COLUMNS = ["user_id", "question_id", "concept", "score", "timestamp"]
LOGS_PATH = DATA_DIR / "interview" / "logs.csv"
ABILITIES_PATH = DATA_DIR / "interview" / "fake_abilities.json"

DAY_MS = 24 * 60 * 60 * 1000
MINUTE_MS = 60 * 1000


def generate(num_users, seed=7):
    rng = random.Random(seed)
    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf8"))
    by_concept = {}
    for q in questions:
        by_concept.setdefault(q["concept"], []).append(q)
    concepts = sorted(by_concept)

    rows, abilities = [], {}
    for u in range(1, num_users + 1):
        user_id = f"u{u:04d}"
        # an overall level plus a per-concept offset, so each candidate has clear strong and weak areas
        level = rng.gauss(0, 0.6)
        ability = {c: level + rng.gauss(0, 1.0) for c in concepts}

        now = 1_750_000_000_000 + rng.randrange(60 * DAY_MS)
        for _ in range(rng.randint(3, 8)):  # sessions
            for _ in range(rng.randint(6, 10)):  # questions per session
                concept = rng.choice(concepts)
                question = rng.choice(by_concept[concept])
                # difficulty 1/2/3 shifts the logit by +0.9 / 0 / -0.9
                logit = 1.4 * ability[concept] - 0.9 * (question["difficulty"] - 2)
                p_good = 1 / (1 + math.exp(-logit))
                score = min(1.0, max(0.0, rng.gauss(p_good, 0.2)))
                rows.append([user_id, question["id"], concept, round(score, 2), now])
                ability[concept] += 0.06  # practice helps a little
                now += rng.randint(2, 5) * MINUTE_MS
            now += rng.randint(1, 6) * DAY_MS
        # ability at the end of the log, kept so predictions can be checked against the truth
        abilities[user_id] = {c: round(a, 3) for c, a in ability.items()}
    return rows, abilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows, abilities = generate(args.users, args.seed)
    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGS_PATH, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)
        writer.writerows(rows)
    ABILITIES_PATH.write_text(json.dumps(abilities), encoding="utf8")
    print(f"wrote {len(rows)} interactions for {args.users} users to {LOGS_PATH}")


if __name__ == "__main__":
    main()
