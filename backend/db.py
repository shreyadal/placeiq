"""SQLite storage for sessions, the questions asked in them, and the graded answers."""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("PLACEIQ_DB", ROOT / "data" / "placeiq.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    company       TEXT,
    num_questions INTEGER NOT NULL,
    started_at    INTEGER NOT NULL,
    ended_at      INTEGER
);
-- one row per question shown. The full question is copied in because generated
-- (company-style) questions do not exist in data/questions.json.
CREATE TABLE IF NOT EXISTS session_questions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id),
    question_id      TEXT NOT NULL,
    concept          TEXT NOT NULL,
    difficulty       INTEGER NOT NULL,
    question         TEXT NOT NULL,
    reference_answer TEXT NOT NULL,
    key_points       TEXT NOT NULL,
    source           TEXT NOT NULL,
    -- what chose the concept: "dkt" (the model) or "heuristic" (average scores)
    selection_source TEXT,
    asked_at         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS answers (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_question_id INTEGER NOT NULL UNIQUE REFERENCES session_questions(id),
    session_id          INTEGER NOT NULL REFERENCES sessions(id),
    user_id             TEXT NOT NULL,
    question_id         TEXT NOT NULL,
    concept             TEXT NOT NULL,
    answer_text         TEXT NOT NULL,
    score               REAL NOT NULL,
    feedback            TEXT NOT NULL,
    missing_points      TEXT NOT NULL,
    grader              TEXT NOT NULL,
    speech              TEXT,
    vision              TEXT,
    created_at          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS answers_user ON answers(user_id, created_at);
"""

JSON_COLUMNS = ("key_points", "missing_points", "speech", "vision")


def now_ms():
    return int(time.time() * 1000)


@contextmanager
def connect():
    """Open a connection, commit on success, and always close it."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with connect() as conn:
        conn.executescript(SCHEMA)
        # databases made before selection_source existed get the column added
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(session_questions)")}
        if "selection_source" not in columns:
            conn.execute("ALTER TABLE session_questions ADD COLUMN selection_source TEXT")


def to_dict(row):
    """Convert a row to a dict, decoding the columns stored as JSON text."""
    if row is None:
        return None
    data = dict(row)
    for column in JSON_COLUMNS:
        if data.get(column) is not None:
            data[column] = json.loads(data[column])
    return data


def user_history(conn, user_id):
    """Every (concept, score) the user has produced, oldest first. This is the input to knowledge tracing."""
    rows = conn.execute(
        "SELECT concept, score FROM answers WHERE user_id = ? ORDER BY created_at, id", (user_id,)).fetchall()
    return [(r["concept"], r["score"]) for r in rows]
