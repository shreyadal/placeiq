"""Predict which concepts a user is most likely to fail.

    python -m ml.predict u0001

DKT is used for serving because one forward pass over a user's history returns a
probability for every concept at once. SAKT and AKT answer "what about concept X next?"
one concept at a time, which would need six passes here.
"""
import argparse
import json
from functools import lru_cache

import pandas as pd
import torch
from pykt.models import load_model

from ml.common import CKPT_DIR, DATA_DIR, EMB_TYPE, MAXLEN, MODEL_CONFIGS, load_data_config
from ml.interview_data import DATASET, score_to_response

# returned for every concept when a user has no history yet: no evidence either way
NO_HISTORY_P = 0.5


@lru_cache(maxsize=1)
def _load():
    """Load the DKT trained on interview data and the concept-name -> index mapping pykt made."""
    # oneDNN's LSTM kernel fails with "could not create a primitive" in some processes on
    # Windows (seen in a uvicorn started from a background shell); torch's own kernel is
    # just as correct and, for a model this small, about as fast
    torch.backends.mkldnn.enabled = False
    data_config = load_data_config()[DATASET]
    model = load_model("dkt", dict(MODEL_CONFIGS["dkt"]), data_config, EMB_TYPE, str(CKPT_DIR / DATASET / "dkt"))
    model.eval()
    keyid2idx = json.loads((DATA_DIR / DATASET / "keyid2idx.json").read_text(encoding="utf8"))
    return model, keyid2idx["concepts"]


def load_history(user_id, logs_csv=None):
    """A user's (concept, score) pairs in time order, read from the interview log csv."""
    logs_csv = logs_csv or DATA_DIR / "interview_simulated.csv"
    df = pd.read_csv(logs_csv, dtype={"user_id": str})
    df = df[df["user_id"] == str(user_id)].sort_values("timestamp", kind="stable")
    return list(zip(df["concept"], df["score"]))


def predict_mastery(history):
    """Probability of a good answer on each concept, given a list of (concept, score) pairs."""
    model, concept2idx = _load()
    # the model has never seen concepts outside its training data, so those are ignored
    history = [(c, s) for c, s in history if c in concept2idx][-MAXLEN:]
    if not history:
        return {concept: NO_HISTORY_P for concept in concept2idx}

    concepts = torch.tensor([[concept2idx[c] for c, _ in history]])
    responses = torch.tensor([[score_to_response(s) for _, s in history]])
    with torch.no_grad():
        # output shape is (1, len(history), num_concepts); the last step uses the whole history
        probs = model(concepts, responses)[0, -1]
    return {concept: round(float(probs[idx]), 4) for concept, idx in concept2idx.items()}


def predict_weak_concepts(user_id, top_k=3, history=None):
    """The top_k concepts the user is most likely to fail, weakest first.

    history is a list of (concept, score) pairs in time order. If it is not given, the
    user's rows are read from data/interview_simulated.csv.
    Returns [{"concept": "OS", "p_correct": 0.31}, ...].
    """
    if history is None:
        history = load_history(user_id)
    mastery = predict_mastery(history)
    weakest = sorted(mastery.items(), key=lambda item: item[1])[:top_k]
    return [{"concept": concept, "p_correct": p} for concept, p in weakest]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()
    for item in predict_weak_concepts(args.user_id, args.top_k):
        print(f"{item['concept']:5s} p(correct) = {item['p_correct']:.2f}")
