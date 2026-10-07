"""Convert interview logs to the files pykt trains on.

    python -m ml.interview_data                 # uses data/interview/logs.csv
    python -m ml.interview_data path/to/logs.csv

Input: csv with columns user_id, question_id, concept, score (0-1), timestamp (ms).
Output: data/interview/data.txt in pykt's six-lines-per-student format, then pykt's own
split step produces the train/test sequence files. See ml/README.md for both formats.
"""
import argparse

import pandas as pd

from ml.common import DATA_DIR
from ml.preprocess import split

DATASET = "interview"
# pykt models predict a binary response, so a 0-1 score is cut here
CORRECT_THRESHOLD = 0.5


def score_to_response(score):
    return 1 if float(score) >= CORRECT_THRESHOLD else 0


def logs_to_data_txt(logs_csv, data_txt):
    df = pd.read_csv(logs_csv, dtype={"user_id": str, "question_id": str, "concept": str})
    for column in ("user_id", "question_id", "concept"):
        # pykt joins fields with "," and splits multi-concept tags on "_"
        if df[column].str.contains("[,_]").any():
            raise ValueError(f"{column} values must not contain ',' or '_'")

    lines = []
    for user_id, user_df in df.groupby("user_id", sort=False):
        user_df = user_df.sort_values("timestamp", kind="stable")
        lines += [
            f"{user_id},{len(user_df)}",
            ",".join(user_df["question_id"]),
            ",".join(user_df["concept"]),
            ",".join(str(score_to_response(s)) for s in user_df["score"]),
            ",".join(str(int(t)) for t in user_df["timestamp"]),
            "NA",  # time spent per question is not recorded
        ]
    with open(data_txt, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    return df["user_id"].nunique(), len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_csv", nargs="?", default=str(DATA_DIR / DATASET / "logs.csv"))
    args = parser.parse_args()

    out_dir = DATA_DIR / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    users, interactions = logs_to_data_txt(args.logs_csv, out_dir / "data.txt")
    print(f"wrote {interactions} interactions from {users} users to {out_dir / 'data.txt'}")
    split(DATASET)


if __name__ == "__main__":
    main()
