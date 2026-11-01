"""Generate a question in the style of a company's interviews.

    python -m rag.generate Amazon OS

Retrieval finds experiences from that company about the concept; a language model then
writes one new question in the same style, with a reference answer and key points so it can
be graded like a question from the bank.

PLACEIQ_GENERATOR picks the model:
  gemini   Google's Gemini API; needs GEMINI_API_KEY. The default whenever that key is set.
  ollama   a local model served by Ollama. The default when no Gemini key is set.
If the model cannot be reached, generate_question returns None and the caller uses the
question bank.
"""
import logging
import os
import sys
import uuid

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from backend import gemini

from rag.store import CONCEPT_DESCRIPTIONS, retrieve

log = logging.getLogger(__name__)

# any model pulled into Ollama works; OLLAMA_HOST (default http://localhost:11434) is read by the client
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
# the first call also loads the model into memory, which can take a while on CPU
TIMEOUT_SECONDS = 120

DIFFICULTY_WORDS = {1: "easy", 2: "medium", 3: "hard"}

SYSTEM_PROMPT = """You write questions for a mock interview tool that helps college students prepare for campus placements at a specific company.

You are given notes from past interviews at the company and a concept to test. Write one new question on that concept that would feel at home in this company's interviews. Use the notes to judge the style: how applied or theoretical the questions are, how deep the follow-ups go, and whether they are framed as definitions, scenarios or design problems. Write a new question in that style on the requested concept; the notes are examples of style, and repeating one of their questions would not give the student fresh practice.

The student answers out loud in one to two minutes and cannot write code, so ask something that can be answered by explaining: a concept, a comparison, an approach to a problem and its complexity, or a design decision. The question must be self-contained, since the student sees nothing else.

Then write the reference answer a strong candidate would give, in three or four sentences, and four or five key points as short phrases. A grader compares the student's answer with these, so the key points should be the ideas whose presence separates a good answer from a weak one."""


class GeneratedQuestion(BaseModel):
    question: str
    reference_answer: str
    key_points: list[str]


def generator_name():
    """The model in use: PLACEIQ_GENERATOR if set, else "gemini" when a Gemini key is present, else "ollama"."""
    default = "gemini" if gemini.has_key() else "ollama"
    return os.environ.get("PLACEIQ_GENERATOR", default)


def _generate_with_gemini(prompt):
    return gemini.generate_json(SYSTEM_PROMPT, prompt, GeneratedQuestion)


def _generate_with_ollama(prompt):
    response = ollama.Client(timeout=TIMEOUT_SECONDS).chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        # the JSON schema constrains the output, so it always parses into a GeneratedQuestion
        format=GeneratedQuestion.model_json_schema(),
        # keep the model loaded between questions; the default unloads it after 5 minutes
        keep_alive="30m",
    )
    return GeneratedQuestion.model_validate_json(response.message.content)


def generate_question(company, concept, difficulty, exclude=()):
    """Returns a question dict shaped like the entries of data/questions.json, or None if
    there is nothing to ground it in or the model is unavailable.

    exclude: questions already asked in this session, so they are not repeated.
    """
    experiences = retrieve(company, [concept])
    if not experiences:
        return None

    notes = "\n\n".join(f"<note role=\"{e['role']}\">\n{e['text']}\n</note>" for e in experiences)
    already = "\n".join(f"- {q}" for q in exclude) or "(none yet)"
    prompt = (
        f"Company: {company}\n"
        f"Concept to test: {concept} ({CONCEPT_DESCRIPTIONS.get(concept, concept)})\n"
        f"Difficulty: {DIFFICULTY_WORDS.get(difficulty, 'medium')}\n\n"
        f"<interview_notes>\n{notes}\n</interview_notes>\n\n"
        f"Questions already asked in this session, which the new one should differ from:\n{already}"
    )

    if generator_name() == "gemini":
        try:
            generated = _generate_with_gemini(prompt)
        except Exception as e:
            # no key, a bad key, no network, quota, a blocked or unparseable reply
            log.warning("Gemini question generation failed: %s: %s", type(e).__name__, e)
            return None
    else:
        try:
            generated = _generate_with_ollama(prompt)
        except ollama.ResponseError as e:
            # a 404 here means the model has not been pulled: ollama pull <model>
            log.warning("question generation failed with status %s: %s", e.status_code, e.error)
            return None
        except (ConnectionError, httpx.TransportError) as e:
            log.warning("could not reach Ollama for question generation: %s", e)
            return None
        except ValidationError as e:
            log.warning("question generation returned output that could not be parsed: %s", e)
            return None

    if not generated.question.strip() or not generated.reference_answer.strip() or not generated.key_points:
        log.warning("question generation returned an incomplete question")
        return None
    return {
        "id": f"gen-{uuid.uuid4().hex[:8]}",
        "concept": concept,
        "difficulty": difficulty,
        "question": generated.question,
        "reference_answer": generated.reference_answer,
        "key_points": generated.key_points,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    result = generate_question(sys.argv[1], sys.argv[2], 2)
    if result is None:
        print("no question generated (see warning above)")
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
