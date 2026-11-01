"""Endpoints that expose the trained DKT model: its test metrics, its predictions, and a
check against the simulated candidates whose true ability is known.

The numbers come from ml/artifacts/metrics.json, which ml.train and ml.test_model write.
"""
import json
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend import selection
from ml.common import METRICS_JSON

log = logging.getLogger(__name__)
router = APIRouter(prefix="/model", tags=["model"])

Concept = Literal["CN", "DBMS", "DSA", "ML", "OOP", "OS"]


class HistoryItem(BaseModel):
    concept: Concept
    score: float = Field(ge=0, le=1)
    timestamp: int


class PredictRequest(BaseModel):
    history: list[HistoryItem] = Field(max_length=5000)


def require_model():
    if not selection.DKT_CHECKPOINT.exists():
        raise HTTPException(503, "no trained DKT checkpoint; run: python -m ml.train --model dkt --dataset interview")


@router.get("/metrics")
def metrics():
    """metrics.json exactly as the training and test scripts wrote it."""
    if not METRICS_JSON.exists():
        raise HTTPException(404, "ml/artifacts/metrics.json not found; run ml.train and ml.test_model")
    return json.loads(METRICS_JSON.read_text(encoding="utf8"))


@router.post("/predict")
def predict(body: PredictRequest):
    """Predicted chance of a good answer on each concept, from a candidate's answer history."""
    require_model()
    from ml.predict import predict_mastery

    history = sorted(body.history, key=lambda h: h.timestamp)
    try:
        mastery = predict_mastery([(h.concept, h.score) for h in history])
    except Exception as e:
        log.exception("DKT prediction failed")
        raise HTTPException(503, f"DKT prediction failed: {type(e).__name__}: {e}") from e
    return {"source": "dkt", "answers": len(history), "mastery": mastery}


@router.get("/candidates")
def candidates():
    """Ids of the held-out simulated candidates the model never trained on."""
    from ml.test_model import held_out_users

    return list(held_out_users())


@router.get("/check/{candidate_id}")
def check(candidate_id: str):
    """For a held-out simulated candidate: predicted mastery next to their true ability."""
    require_model()
    from ml.test_model import check_candidate

    try:
        return check_candidate(candidate_id)
    except KeyError:
        raise HTTPException(404, "not a held-out simulated candidate; see GET /model/candidates") from None
