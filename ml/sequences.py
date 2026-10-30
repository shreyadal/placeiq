"""Turn the student splits made by data/prepare.py into the windowed files pykt trains on.

    python -m ml.sequences assist2009
    python -m ml.sequences interview

data/prepare.py runs the first half of pykt's split_datasets.main (read, id mapping, test
split, 5 folds) and writes data/<name>/train_valid.csv and test.csv, one row per student.
This runs the second half on those files, unchanged: every history is cut into windows of
MAXLEN answers and padded with -1 (see ml/README.md), giving
  train_valid_sequences.csv, test_sequences.csv, test_window_sequences.csv
and the dataset's entry in data/data_config.json, which ml/train.py reads.
"""
import argparse
import json

import pandas as pd
from pykt.preprocess import split_datasets as pykt

from ml.common import CONFIG_PATH, DATA_DIR, KFOLD, MAXLEN, MIN_SEQ_LEN


def build(name):
    out = DATA_DIR / name
    # dtype=str keeps the comma-joined lists and ids exactly as prepare.py wrote them
    train_valid = pd.read_csv(out / "train_valid.csv", dtype=str)
    test = pd.read_csv(out / "test.csv", dtype=str)
    keys = set(train_valid.columns)  # fold, uid, questions, concepts, responses, [timestamps,] is_repeat

    pykt.generate_sequences(train_valid, keys, MIN_SEQ_LEN, MAXLEN).to_csv(
        out / "train_valid_sequences.csv", index=None)
    test_keys = list(keys) + ["cidxs"]
    pykt.generate_sequences(test, test_keys, MIN_SEQ_LEN, MAXLEN).to_csv(out / "test_sequences.csv", index=None)
    pykt.generate_window_sequences(test, test_keys, MAXLEN).to_csv(out / "test_window_sequences.csv", index=None)

    # pykt's write_config updates config[name] in place and fails if the file is empty or the key is missing
    config = json.loads(CONFIG_PATH.read_text(encoding="utf8")) if CONFIG_PATH.exists() else {}
    config.setdefault(name, {})
    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf8")
    key_to_index = json.loads((out / "keyid2idx.json").read_text(encoding="utf8"))
    pykt.write_config(dataset_name=name, dkeyid2idx=key_to_index, effective_keys=keys, configf=CONFIG_PATH.as_posix(),
                      dpath=out.as_posix(), k=KFOLD, min_seq_len=MIN_SEQ_LEN, maxlen=MAXLEN)

    # cached tensors from an earlier run would no longer match the new csv files
    for stale in out.glob("*.pkl"):
        stale.unlink()
    print(f"{name}: {len(train_valid)} train/valid and {len(test)} test students -> sequence files in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["assist2009", "interview"])
    build(parser.parse_args().dataset)
