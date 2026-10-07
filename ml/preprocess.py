"""Turn a raw dataset into the files pykt trains on.

    python -m ml.preprocess assist2009

Two steps, both done by pykt:
  1. raw csv -> data.txt        (one student per 6 lines, see ml/README.md)
  2. data.txt -> train/test csv (id mapping, 80/20 split, 5 folds, windows of 200)
"""
import argparse
import json
import re

from pykt.preprocess import split_datasets
from pykt.preprocess.data_proprocess import process_raw_data

from ml.common import CONFIG_PATH, DATA_DIR, KFOLD, MAXLEN, MIN_SEQ_LEN

RAW_FILES = {
    "assist2009": "skill_builder_data_corrected_collapsed.csv",
}


def split(dataset_name):
    """Run pykt's split step on data/<dataset_name>/data.txt and record the dataset in data_config.json."""
    dname = (DATA_DIR / dataset_name).as_posix()

    # pykt's write_config updates config[dataset_name] in place and fails if the key is missing
    config = json.loads(CONFIG_PATH.read_text(encoding="utf8")) if CONFIG_PATH.exists() else {}
    config.setdefault(dataset_name, {})
    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf8")

    # cached tensors from an earlier split would no longer match the new csv files
    for stale in (DATA_DIR / dataset_name).glob("*.pkl"):
        stale.unlink()

    split_datasets.main(dname, f"{dname}/data.txt", dataset_name, CONFIG_PATH.as_posix(), MIN_SEQ_LEN, MAXLEN, KFOLD)


def fix_user_id_lines(data_txt):
    """pykt groups by ['user_id'] and writes str(key). Current pandas returns that key as a
    1-tuple, so the first line of each student reads "('64525',),5" instead of "64525,5"."""
    with open(data_txt, encoding="utf8") as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines), 6):
        lines[i] = re.sub(r"^\('(.*)',\)", r"\1", lines[i])
    with open(data_txt, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(RAW_FILES))
    args = parser.parse_args()

    raw_path = DATA_DIR / args.dataset / RAW_FILES[args.dataset]
    # the ASSISTments file has a few latin-1 bytes in the free-text answer column, but pykt
    # reads it as utf-8, so hand pykt a re-encoded copy
    utf8_path = raw_path.with_name(raw_path.stem + "_utf8.csv")
    if not utf8_path.exists():
        utf8_path.write_text(raw_path.read_text(encoding="latin-1"), encoding="utf8", newline="")

    # pykt splits this path on "/" to find the folder, so it must use forward slashes
    _, data_txt = process_raw_data(args.dataset, {args.dataset: utf8_path.as_posix()})
    fix_user_id_lines(data_txt)
    split(args.dataset)


if __name__ == "__main__":
    main()
