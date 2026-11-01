"""Check the trained interview DKT against the simulated candidates' true abilities.

    python -m ml.test_model                  # after: python -m ml.train --model dkt --dataset interview

The simulated candidates in data/interview_simulated.csv come from ml/fake_interviews.py
(seed 7, 400 users), which gives each candidate a hidden ability per concept. Re-running
the simulator reproduces the csv row for row (checked below), so its abilities are the
ground truth. Only the 20% of candidates in data/interview/test.csv are used: the model
never trained on them.

For each held-out candidate the model reads their whole answer history and predicts the
chance of a good answer per concept. That is compared with their true ability at the end
of the history:
  - Spearman correlation across the 6 concepts (1 = same ranking, 0 = unrelated)
  - whether the truly weakest concept is among the 2 the model predicts weakest
    (picking 2 of 6 at random would hit 1/3 of the time)
The same two numbers are given for the app's fallback heuristic (average score per
concept, backend.selection.score_mastery) as a baseline. The results are saved under
"weak_concepts" in ml/artifacts/metrics.json.
"""
import argparse
import csv
import json
import statistics
from functools import lru_cache

import pandas as pd
from scipy.stats import spearmanr

from ml.common import DATA_DIR, METRICS_JSON, update_metrics
from ml.fake_interviews import generate, p_good_answer
from ml.interview_data import DATASET
from ml.predict import load_history, predict_mastery

LOGS_CSV = DATA_DIR / "interview_simulated.csv"
SIM_USERS, SIM_SEED = 400, 7


@lru_cache(maxsize=1)
def true_abilities():
    """The simulator's abilities, after checking that it reproduces the csv exactly."""
    rows, abilities = generate(SIM_USERS, SIM_SEED)
    with open(LOGS_CSV, encoding="utf8") as f:
        logged = list(csv.reader(f))[1:]
    same = len(rows) == len(logged) and all(
        [str(g) for g in gen[:3]] + [float(gen[3]), str(gen[4])] == log[:3] + [float(log[3]), log[4]]
        for gen, log in zip(rows, logged))
    if not same:
        raise SystemExit(f"the simulator no longer reproduces {LOGS_CSV.name}, so its abilities cannot be used")
    return abilities


@lru_cache(maxsize=1)
def held_out_users():
    keyid2idx = json.loads((DATA_DIR / DATASET / "keyid2idx.json").read_text(encoding="utf8"))
    idx2uid = {i: u for u, i in keyid2idx["uid"].items()}
    test = pd.read_csv(DATA_DIR / DATASET / "test.csv")
    return tuple(sorted(idx2uid[int(i)] for i in test["uid"].unique()))


def score_average(history, concepts):
    """Same as backend.selection.score_mastery, without importing the backend."""
    totals = {c: [0.5, 1] for c in concepts}
    for c, s in history:
        totals[c][0] += s
        totals[c][1] += 1
    return {c: t / n for c, (t, n) in totals.items()}


def compare(predicted, truth):
    concepts = sorted(truth)
    rho = spearmanr([predicted[c] for c in concepts], [truth[c] for c in concepts]).statistic
    weakest = min(truth, key=truth.get)
    top2 = sorted(predicted, key=predicted.get)[:2]
    return rho, weakest in top2


def check_candidate(user):
    """Predicted mastery next to true ability for one held-out candidate."""
    if user not in held_out_users():
        raise KeyError(user)
    history = load_history(user, LOGS_CSV)
    truth = true_abilities()[user]
    predicted = predict_mastery(history)
    rho, hit = compare(predicted, truth)
    return {
        "candidate_id": user, "answers": len(history),
        "predicted_mastery": predicted, "true_ability": truth,
        # the same ability put on a 0-1 scale: the simulator's mean score on a medium question.
        # Only the ranking is scored (Spearman), so this is for display.
        "true_p_good_medium": {c: round(p_good_answer(a), 4) for c, a in truth.items()},
        "spearman": round(float(rho), 4) if rho == rho else None,
        "true_weakest": min(truth, key=truth.get),
        "predicted_weakest_2": sorted(predicted, key=predicted.get)[:2],
        "true_weakest_in_predicted_2": bool(hit),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=5, help="print the per-concept table for this many candidates")
    args = parser.parse_args()

    abilities = true_abilities()
    users = held_out_users()
    concepts = sorted(next(iter(abilities.values())))
    results = {"DKT": ([], []), "score average": ([], [])}

    for n, user in enumerate(users):
        history = load_history(user, LOGS_CSV)
        truth = abilities[user]
        dkt = predict_mastery(history)
        for name, predicted in (("DKT", dkt), ("score average", score_average(history, concepts))):
            rho, hit = compare(predicted, truth)
            results[name][0].append(rho)
            results[name][1].append(hit)

        if n < args.show:
            rho, hit = compare(dkt, truth)
            print(f"\n{user}  ({len(history)} answers)   Spearman {rho:+.2f}   "
                  f"true weakest {min(truth, key=truth.get)} in predicted bottom 2: {'yes' if hit else 'no'}")
            print(f"  {'concept':8s}{'predicted p(good)':>19s}{'true ability':>14s}")
            for c in sorted(concepts, key=truth.get):
                print(f"  {c:8s}{dkt[c]:19.3f}{truth[c]:14.2f}")

    print(f"\nheld-out candidates: {len(users)}  (true ability is on a logit scale; only its ranking is compared)")
    print(f"{'':15s}{'mean Spearman':>15s}{'median Spearman':>17s}{'weakest in bottom 2':>21s}")
    for name, (rhos, hits) in results.items():
        rhos = [r for r in rhos if r == r]  # nan when every prediction is equal
        print(f"{name:15s}{statistics.mean(rhos):15.3f}{statistics.median(rhos):17.3f}"
              f"{sum(hits):>12d}/{len(hits)} = {sum(hits) / len(hits):.1%}")
    print(f"{'random':15s}{0:15.3f}{0:17.3f}{'':>12s}  = {2 / 6:.1%}")

    summary = {"command": "python -m ml.test_model", "candidates": len(users), "concepts": concepts,
               "random_top2_rate": round(2 / 6, 4)}
    for name, key in (("DKT", "dkt"), ("score average", "score_average")):
        rhos, hits = results[name]
        rhos = [r for r in rhos if r == r]
        summary[key] = {"mean_spearman": round(statistics.mean(rhos), 4),
                        "median_spearman": round(statistics.median(rhos), 4),
                        "weakest_in_top2": sum(hits), "weakest_in_top2_rate": round(sum(hits) / len(hits), 4)}
    update_metrics("weak_concepts", summary)
    print(f"wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
