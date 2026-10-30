"""Turn the two csv files in this folder into the files pykt trains on.

    pip install pykt-toolkit pandas
    python data/prepare.py                     (from the repo root)

For each dataset this writes, into data/<name>/:
  data.txt         one student per 6 lines (id and length, questions, concepts, responses,
                   timestamps, time spent); pykt's input format
  keyid2idx.json   raw id -> index for questions, concepts and students
  train_valid.csv  80% of students, each assigned to one of 5 folds
  test.csv         the other 20% of students, fold = -1
See README.md for the columns.
"""
import json
from pathlib import Path

import pandas as pd
from pykt.preprocess import split_datasets as pykt

DATA_DIR = Path(__file__).parent
# pykt models predict a binary response, so a 0-1 interview score is cut here
CORRECT_THRESHOLD = 0.5
# students with fewer answers than this are left out
MIN_SEQ_LEN = 3
TEST_RATIO = 0.2
KFOLD = 5


def write_data_txt(df, path):
    """df has one row per answer, with string columns user, question, concept, response, timestamp."""
    lines = []
    for user, rows in df.groupby("user", sort=False):
        lines += [
            f"{user},{len(rows)}",
            ",".join(rows["question"]),
            ",".join(rows["concept"]),
            ",".join(rows["response"]),
            ",".join(rows["timestamp"]) if rows["timestamp"].notna().all() else "NA",
            "NA",  # time spent per question is not recorded
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def split(name):
    """The first half of pykt's split_datasets.main: read, map ids, split students into test and folds."""
    out = DATA_DIR / name
    df, keys = pykt.read_data(str(out / "data.txt"), MIN_SEQ_LEN)
    max_concepts = pykt.get_max_concepts(df)
    # a question tagged with two skills becomes two rows; the second is marked is_repeat = 1
    df, keys = pykt.extend_multi_concepts(df, keys)
    df, key_to_index = pykt.id_mapping(df)
    key_to_index["max_concepts"] = max_concepts
    (out / "keyid2idx.json").write_text(json.dumps(key_to_index), encoding="utf8")

    columns = [k for k in pykt.ALL_KEYS if k in keys | {"fold"}]
    # both steps shuffle whole students with a fixed seed, so no student is in train and test
    train_valid, test = pykt.train_test_split(df, TEST_RATIO)
    train_valid = pykt.KFold_split(train_valid, KFOLD)
    train_valid[columns].to_csv(out / "train_valid.csv", index=None)

    test = test.copy()
    test["fold"] = -1
    test["cidxs"] = pykt.get_inter_qidx(test)  # a running number for every test interaction
    test[columns + ["cidxs"]].to_csv(out / "test.csv", index=None)


def assist2009():
    df = pd.read_csv(DATA_DIR / "assistments2009_cleaned.csv", dtype=str)
    # the cleaned csv is already in time order within each student
    df = pd.DataFrame({"user": df["user_id"], "question": df["problem_id"], "concept": df["skill_id"],
                       "response": df["correct"], "timestamp": None})
    (DATA_DIR / "assist2009").mkdir(exist_ok=True)
    write_data_txt(df, DATA_DIR / "assist2009" / "data.txt")
    split("assist2009")


def interview():
    df = pd.read_csv(DATA_DIR / "interview_simulated.csv", dtype={"user_id": str, "question_id": str, "concept": str})
    for column in ("user_id", "question_id", "concept"):
        # pykt joins fields with "," and splits multi-concept tags on "_"
        if df[column].str.contains("[,_]").any():
            raise ValueError(f"{column} values must not contain ',' or '_'")
    df = df.sort_values("timestamp", kind="stable")
    df = pd.DataFrame({"user": df["user_id"], "question": df["question_id"], "concept": df["concept"],
                       "response": (df["score"] >= CORRECT_THRESHOLD).astype(int).astype(str),
                       "timestamp": df["timestamp"].astype(str)})
    df = df.sort_values("user", kind="stable")
    (DATA_DIR / "interview").mkdir(exist_ok=True)
    write_data_txt(df, DATA_DIR / "interview" / "data.txt")
    split("interview")


if __name__ == "__main__":
    assist2009()
    interview()
