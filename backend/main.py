"""PlaceIQ API.

    uvicorn backend.main:app --reload          (from the project root)

A session is a fixed number of questions. Each answer is graded 0-1 and stored; the
stored scores are the history that knowledge tracing reads to choose the next question.
"""
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import db, selection
from backend.grading import grade

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    db.init()
    yield


app = FastAPI(title="PlaceIQ", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSession(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    company: str | None = None
    num_questions: int = Field(default=5, ge=1, le=20)


class SubmitAnswer(BaseModel):
    session_question_id: int
    answer_text: str = Field(max_length=20000)
    # delivery metrics from /speech/analyze and from the browser's face tracking
    speech: dict | None = None
    vision: dict | None = None


def get_session(conn, session_id):
    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if session is None:
        raise HTTPException(404, "session not found")
    return session


def public_question(row, number, total):
    """What the candidate may see: the question without its reference answer."""
    return {
        "session_question_id": row["id"], "question_id": row["question_id"], "concept": row["concept"],
        "difficulty": row["difficulty"], "question": row["question"], "source": row["source"],
        "number": number, "total": total,
    }


def company_question(company, concept, difficulty, exclude):
    """A company-style question from the RAG module, or None if RAG is not set up."""
    try:
        from rag.generate import generate_question
    except ImportError:
        return None
    try:
        return generate_question(company, concept, difficulty, exclude)
    except Exception:
        log.exception("company question generation failed; falling back to the question bank")
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/concepts")
def list_concepts():
    return selection.concepts()


@app.get("/companies")
def list_companies():
    try:
        from rag.store import companies
    except ImportError:
        return []
    return companies()


@app.post("/sessions")
def start_session(body: StartSession):
    with db.connect() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (user_id, company, num_questions, started_at) VALUES (?, ?, ?, ?)",
            (body.user_id, body.company or None, body.num_questions, db.now_ms()))
        return {"session_id": cursor.lastrowid, "num_questions": body.num_questions}


@app.get("/sessions/{session_id}/next")
def next_question(session_id: int):
    with db.connect() as conn:
        session = get_session(conn, session_id)
        if session["ended_at"]:
            raise HTTPException(409, "session has ended")
        asked = conn.execute(
            "SELECT sq.*, a.id AS answer_id FROM session_questions sq "
            "LEFT JOIN answers a ON a.session_question_id = sq.id WHERE sq.session_id = ? ORDER BY sq.id",
            (session_id,)).fetchall()
        total = session["num_questions"]

        # asking again without answering returns the same question, so a page refresh
        # cannot be used to skip one
        if asked and asked[-1]["answer_id"] is None:
            return {"done": False, "question": public_question(asked[-1], len(asked), total)}
        if len(asked) >= total:
            return {"done": True, "question": None}

        history = db.user_history(conn, session["user_id"])
        asked_ids = {row["question_id"] for row in asked}
        seen_before = {row["question_id"] for row in conn.execute(
            "SELECT DISTINCT question_id FROM answers WHERE user_id = ?", (session["user_id"],))}
        question, mastery = selection.next_question(history, asked_ids, seen_before)
        if question is None:
            return {"done": True, "question": None}
        source = "bank"

        if session["company"]:
            asked_texts = [row["question"] for row in asked]
            generated = company_question(session["company"], question["concept"], question["difficulty"], asked_texts)
            if generated:
                question, source = generated, "company"

        cursor = conn.execute(
            "INSERT INTO session_questions (session_id, question_id, concept, difficulty, question, "
            "reference_answer, key_points, source, asked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, question["id"], question["concept"], question["difficulty"], question["question"],
             question["reference_answer"], json.dumps(question["key_points"]), source, db.now_ms()))
        row = conn.execute("SELECT * FROM session_questions WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return {
            "done": False,
            "question": public_question(row, len(asked) + 1, total),
            # why this question was chosen
            "mastery_before": mastery,
        }


@app.post("/sessions/{session_id}/answers")
def submit_answer(session_id: int, body: SubmitAnswer):
    with db.connect() as conn:
        session = get_session(conn, session_id)
        question = db.to_dict(conn.execute(
            "SELECT * FROM session_questions WHERE id = ? AND session_id = ?",
            (body.session_question_id, session_id)).fetchone())
        if question is None:
            raise HTTPException(404, "question not found in this session")
        if conn.execute("SELECT 1 FROM answers WHERE session_question_id = ?", (question["id"],)).fetchone():
            raise HTTPException(409, "this question was already answered")

    # grading calls an external API, so it runs outside the database connection
    result = grade(question["question"], question["reference_answer"], question["key_points"], body.answer_text)

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO answers (session_question_id, session_id, user_id, question_id, concept, answer_text, "
            "score, feedback, missing_points, grader, speech, vision, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (question["id"], session_id, session["user_id"], question["question_id"], question["concept"],
             body.answer_text, result["score"], result["feedback"], json.dumps(result["missing_points"]),
             result["grader"], json.dumps(body.speech) if body.speech else None,
             json.dumps(body.vision) if body.vision else None, db.now_ms()))
    return {**result, "reference_answer": question["reference_answer"]}


def average(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def summarise(conn, session):
    """Scores, delivery averages and updated mastery for one session."""
    answers = [db.to_dict(r) for r in conn.execute(
        "SELECT a.*, sq.question, sq.difficulty FROM answers a JOIN session_questions sq "
        "ON sq.id = a.session_question_id WHERE a.session_id = ? ORDER BY a.id", (session["id"],))]
    by_concept = {}
    for a in answers:
        by_concept.setdefault(a["concept"], []).append(a["score"])
    speech = [a["speech"] for a in answers if a["speech"]]
    vision = [a["vision"] for a in answers if a["vision"]]
    mastery = selection.mastery_for(db.user_history(conn, session["user_id"]))
    return {
        "session_id": session["id"], "user_id": session["user_id"], "company": session["company"],
        "started_at": session["started_at"], "ended_at": session["ended_at"],
        "answered": len(answers), "average_score": average([a["score"] for a in answers]),
        "score_by_concept": {c: average(s) for c, s in by_concept.items()},
        "speech": {
            "words_per_minute": average([s.get("words_per_minute") for s in speech]),
            "filler_rate": average([s.get("filler_rate") for s in speech]),
            "filler_count": sum(s.get("filler_count", 0) for s in speech),
            "longest_pause_seconds": max((s.get("longest_pause_seconds", 0) for s in speech), default=None),
        } if speech else None,
        "vision": {
            "eye_contact_percent": average([v.get("eye_contact_percent") for v in vision]),
            "head_movement": average([v.get("head_movement") for v in vision]),
            "blinks_per_minute": average([v.get("blinks_per_minute") for v in vision]),
        } if vision else None,
        "mastery": mastery,
        "weak_concepts": sorted(mastery, key=mastery.get)[:3],
        "answers": [{
            "question": a["question"], "concept": a["concept"], "difficulty": a["difficulty"],
            "answer_text": a["answer_text"], "score": a["score"], "feedback": a["feedback"],
            "missing_points": a["missing_points"], "grader": a["grader"],
            "speech": a["speech"], "vision": a["vision"],
        } for a in answers],
    }


@app.post("/sessions/{session_id}/end")
def end_session(session_id: int):
    with db.connect() as conn:
        session = get_session(conn, session_id)
        if not session["ended_at"]:
            conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (db.now_ms(), session_id))
            session = get_session(conn, session_id)
        return summarise(conn, session)


@app.get("/sessions/{session_id}")
def session_summary(session_id: int):
    with db.connect() as conn:
        return summarise(conn, get_session(conn, session_id))


@app.get("/users/{user_id}/progress")
def user_progress(user_id: str):
    """Mastery per concept after each session, for the chart on the results page."""
    with db.connect() as conn:
        sessions = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND id IN (SELECT session_id FROM answers) ORDER BY id",
            (user_id,)).fetchall()
        rows = conn.execute(
            "SELECT session_id, concept, score FROM answers WHERE user_id = ? ORDER BY created_at, id",
            (user_id,)).fetchall()

    points, history = [], []
    for session in sessions:
        scores = [r["score"] for r in rows if r["session_id"] == session["id"]]
        # history grows session by session, so each point is the model's view at that time
        history += [(r["concept"], r["score"]) for r in rows if r["session_id"] == session["id"]]
        points.append({
            "session_id": session["id"], "started_at": session["started_at"], "company": session["company"],
            "answered": len(scores), "average_score": average(scores),
            "mastery": selection.mastery_for(history),
        })
    current = selection.mastery_for(history)
    return {
        "user_id": user_id, "concepts": selection.concepts(), "sessions": points,
        "mastery": current, "weak_concepts": sorted(current, key=current.get)[:3] if history else [],
    }


@app.post("/speech/analyze")
def analyze_speech(audio: UploadFile = File(...)):
    """Transcribe an audio recording and return the transcript with delivery metrics."""
    from backend import speech

    suffix = Path(audio.filename or "").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio.file.read())
        path = tmp.name
    try:
        return speech.analyze(path)
    except Exception as e:
        log.exception("transcription failed")
        raise HTTPException(422, f"could not transcribe the audio: {e}") from e
    finally:
        os.unlink(path)
