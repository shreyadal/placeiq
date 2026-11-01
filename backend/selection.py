"""Choose the next question from the user's estimated mastery.

Two ways to estimate mastery, chosen with the PLACEIQ_USE_DKT environment variable:
  1 (default)  the DKT knowledge tracing model in ml/, run over the user's whole history
  0            the candidate's own average score per concept in the current session
DKT is used only when its checkpoint exists and the user has at least MIN_DKT_ANSWERS
answers; otherwise the average-score heuristic is used. Every result says which one
it came from: source "dkt" or "heuristic".
"""
import json
import logging
import os
import random
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"
DKT_CHECKPOINT = Path(__file__).resolve().parent.parent / "ml" / "saved_model" / "interview" / "dkt" / "qid_model.ckpt"
# a concept with no answers yet counts as one imaginary answer scoring this much
PRIOR_SCORE = 0.5
# with fewer answers than this the model has almost nothing to go on
MIN_DKT_ANSWERS = 3


def use_dkt():
    return os.environ.get("PLACEIQ_USE_DKT", "1") == "1"


def dkt_available():
    return use_dkt() and DKT_CHECKPOINT.exists()


def load_dkt():
    """Load the DKT model into memory. Returns False (and logs why) if it cannot be used."""
    if not dkt_available():
        return False
    try:
        from ml.predict import _load
        _load()
        return True
    except Exception:
        log.exception("loading the DKT model failed; the heuristic will be used")
        return False


@lru_cache(maxsize=1)
def question_bank():
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf8"))


def concepts():
    return sorted({q["concept"] for q in question_bank()})


def score_mastery(history):
    """Average score per concept from a list of (concept, score) pairs, pulled towards
    PRIOR_SCORE by one imaginary answer: an unseen concept is 0.5, and a single answer
    scoring 0 gives 0.25 rather than 0."""
    totals = {c: [PRIOR_SCORE, 1] for c in concepts()}
    for concept, score in history:
        if concept in totals:
            totals[concept][0] += score
            totals[concept][1] += 1
    return {c: round(total / count, 4) for c, (total, count) in totals.items()}


def mastery_and_source(history, fallback_history=None):
    """Estimated chance of a good answer per concept, with where it came from.

    history is the user's (concept, score) pairs across all sessions, in time order.
    Returns (mastery, source, reason): source is "dkt" when the model made the estimate,
    "heuristic" when it is score_mastery of fallback_history (default: history), and
    reason says why the model was not used (None when it was).
    """
    fallback_history = history if fallback_history is None else fallback_history
    if not use_dkt():
        reason = "PLACEIQ_USE_DKT=0"
    elif not DKT_CHECKPOINT.exists():
        reason = "no trained DKT checkpoint"
    elif len(history) < MIN_DKT_ANSWERS:
        reason = f"fewer than {MIN_DKT_ANSWERS} answers so far"
    else:
        try:
            from ml.predict import predict_mastery

            mastery = predict_mastery(history)
            return {c: mastery.get(c, 0.5) for c in concepts()}, "dkt", None
        except Exception as e:
            log.exception("DKT prediction failed; using the heuristic")
            reason = f"DKT prediction failed: {type(e).__name__}"
    return score_mastery(fallback_history), "heuristic", reason


def mastery_for(history):
    """Just the mastery from mastery_and_source."""
    return mastery_and_source(history)[0]


def pick_concept(mastery, rng=random):
    """Sample a concept, favouring weak ones.

    Weight is (1 - p)^2: a concept at p=0.2 is drawn 16x as often as one at p=0.8. Sampling
    rather than always taking the weakest keeps some coverage of other concepts, which the
    model needs in order to notice that a "strong" concept has slipped.
    """
    names = list(mastery)
    weights = [(1 - mastery[c]) ** 2 + 0.02 for c in names]
    return rng.choices(names, weights=weights)[0]


def target_difficulty(p_correct):
    """Aim a little above what the user can already do: easy when weak, hard when strong."""
    if p_correct < 0.4:
        return 1
    if p_correct < 0.7:
        return 2
    return 3


def pick_question(concept, difficulty, asked_in_session, seen_before, rng=random):
    """Pick a bank question on the concept: never one already asked in this session,
    preferably one the user has never seen, and as close to the target difficulty as possible."""
    candidates = [q for q in question_bank() if q["concept"] == concept and q["id"] not in asked_in_session]
    if not candidates:
        return None
    fresh = [q for q in candidates if q["id"] not in seen_before] or candidates
    closest = min(abs(q["difficulty"] - difficulty) for q in fresh)
    return rng.choice([q for q in fresh if abs(q["difficulty"] - difficulty) == closest])


def next_question(history, session_history, asked_in_session, seen_before, rng=random):
    """Returns (question, mastery, source, reason). question is None once the whole bank
    has been asked in this session.

    history is the user's (concept, score) pairs across all sessions, session_history only
    those from the current session. DKT reads the first; the heuristic the second.
    """
    mastery, source, reason = mastery_and_source(history, session_history)
    remaining = dict(mastery)
    while remaining:
        concept = pick_concept(remaining, rng)
        question = pick_question(concept, target_difficulty(mastery[concept]), asked_in_session, seen_before, rng)
        if question:
            return question, mastery, source, reason
        del remaining[concept]  # every question on this concept was already asked
    return None, mastery, source, reason
