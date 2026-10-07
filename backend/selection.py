"""Choose the next question from the user's predicted mastery."""
import json
import logging
import random
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"


@lru_cache(maxsize=1)
def question_bank():
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf8"))


def concepts():
    return sorted({q["concept"] for q in question_bank()})


def mastery_for(history):
    """Probability of a good answer per concept, from the knowledge tracing model.

    If the model has not been trained yet, every concept gets 0.5 so the app still runs
    (questions are then picked without personalisation).
    """
    try:
        from ml.predict import predict_mastery

        mastery = predict_mastery(history)
    except (FileNotFoundError, KeyError):
        log.warning("no trained DKT checkpoint found; using uniform mastery")
        mastery = {}
    return {c: mastery.get(c, 0.5) for c in concepts()}


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


def next_question(history, asked_in_session, seen_before, rng=random):
    """Returns (question, mastery). question is None once the whole bank has been asked in this session."""
    mastery = mastery_for(history)
    remaining = dict(mastery)
    while remaining:
        concept = pick_concept(remaining, rng)
        question = pick_question(concept, target_difficulty(mastery[concept]), asked_in_session, seen_before, rng)
        if question:
            return question, mastery
        del remaining[concept]  # every question on this concept was already asked
    return None, mastery
