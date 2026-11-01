"""Tests for the /model endpoints and for the numbers they show.

The endpoints that run the model are skipped when no checkpoint has been trained
(checkpoints are not committed): python -m ml.train --model dkt --dataset interview
"""
import json
import os
import tempfile

os.environ.setdefault("PLACEIQ_DB", os.path.join(tempfile.mkdtemp(), "test.db"))
os.environ.setdefault("PLACEIQ_GRADER", "keyword")
os.environ.setdefault("PLACEIQ_WARMUP", "0")

import pytest
from fastapi.testclient import TestClient

from backend import selection
from backend.main import app
from ml.common import CKPT_DIR, METRICS_JSON

needs_model = pytest.mark.skipif(not selection.DKT_CHECKPOINT.exists(), reason="DKT checkpoint not trained")
needs_metrics = pytest.mark.skipif(not METRICS_JSON.exists(), reason="ml/artifacts/metrics.json not written")
CONCEPTS = {"CN", "DBMS", "DSA", "ML", "OOP", "OS"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@needs_metrics
def test_metrics_endpoint_returns_metrics_json_unchanged(client):
    assert client.get("/model/metrics").json() == json.loads(METRICS_JSON.read_text(encoding="utf8"))


@needs_metrics
def test_metrics_match_the_checkpoints_they_describe():
    """metrics.json must describe the checkpoints on disk, not an older run."""
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf8"))
    for dataset in ("interview", "assist2009"):
        result_json = CKPT_DIR / dataset / "dkt" / "result.json"
        if not result_json.exists():
            pytest.skip(f"{dataset} checkpoint not trained")
        result = json.loads(result_json.read_text(encoding="utf8"))
        for key in ("test_auc", "test_auc_no_repeat", "test_acc", "valid_auc", "best_epoch", "epochs_run",
                    "train_seconds"):
            assert metrics[dataset][key] == result[key], (dataset, key)


@needs_model
@needs_metrics
def test_weak_concept_metrics_reproduce_from_the_check_endpoint(client):
    """Averaging /model/check over every held-out candidate gives the numbers in metrics.json."""
    saved = json.loads(METRICS_JSON.read_text(encoding="utf8"))["weak_concepts"]
    candidates = client.get("/model/candidates").json()
    assert len(candidates) == saved["candidates"]
    checks = [client.get(f"/model/check/{c}").json() for c in candidates]
    rhos = [c["spearman"] for c in checks if c["spearman"] is not None]
    # each candidate's Spearman is rounded to 4 places, so the mean can differ by that rounding
    assert sum(rhos) / len(rhos) == pytest.approx(saved["dkt"]["mean_spearman"], abs=1e-4)
    assert sum(c["true_weakest_in_predicted_2"] for c in checks) == saved["dkt"]["weakest_in_top2"]


@needs_model
def test_predict(client):
    history = [{"concept": "OS", "score": 0.1, "timestamp": t} for t in range(5)]
    history += [{"concept": "DSA", "score": 0.9, "timestamp": t} for t in range(5, 10)]
    reply = client.post("/model/predict", json={"history": history}).json()
    assert reply["source"] == "dkt" and reply["answers"] == 10
    assert set(reply["mastery"]) == CONCEPTS and all(0 <= p <= 1 for p in reply["mastery"].values())
    assert reply["mastery"]["OS"] < reply["mastery"]["DSA"]
    # order comes from the timestamps, not from the list order
    assert client.post("/model/predict", json={"history": history[::-1]}).json() == reply

    assert client.post("/model/predict", json={"history": [{"concept": "Physics", "score": 1, "timestamp": 0}]}
                       ).status_code == 422
    assert client.post("/model/predict", json={"history": [{"concept": "OS", "score": 2, "timestamp": 0}]}
                       ).status_code == 422


@needs_model
def test_check(client):
    candidates = client.get("/model/candidates").json()
    candidate = candidates[0]
    reply = client.get(f"/model/check/{candidate}").json()
    assert reply["candidate_id"] == candidate and reply["answers"] > 0
    assert set(reply["predicted_mastery"]) == set(reply["true_ability"]) == CONCEPTS
    assert -1 <= reply["spearman"] <= 1
    # a candidate the model trained on is refused
    trained_on = next(f"u{i:04d}" for i in range(1, 401) if f"u{i:04d}" not in candidates)
    assert client.get(f"/model/check/{trained_on}").status_code == 404
    assert client.get("/model/check/nobody").status_code == 404


def test_model_endpoints_without_a_checkpoint(client, monkeypatch):
    monkeypatch.setattr(selection, "DKT_CHECKPOINT", selection.Path("no/such/model.ckpt"))
    history = [{"concept": "OS", "score": 0.5, "timestamp": 0}]
    assert client.post("/model/predict", json={"history": history}).status_code == 503
    assert client.get("/model/check/u0001").status_code == 503
