"""One call to Google's Gemini API that returns JSON parsed into a pydantic model.

Used by the grader and by the company question generator. The client reads the key from
GEMINI_API_KEY (or GOOGLE_API_KEY). The free tier is often busy (503) or slow (504), so each
model is retried a few times, and then GEMINI_FALLBACK_MODEL is tried before giving up.
"""
import logging
import os

log = logging.getLogger(__name__)

# any Gemini models that support JSON output work; list them with client.models.list()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
# per attempt; with 3 attempts on each of two models the worst case is about two minutes
TIMEOUT_SECONDS = 20
ATTEMPTS = 3


def has_key():
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def generate_json(system_prompt, contents, schema, temperature=None):
    """Returns an instance of schema. Raises the last model's error if every model fails."""
    # imported here so the app runs without the package when Gemini is not used
    from google import genai
    from google.genai import types

    client = genai.Client(http_options=types.HttpOptions(
        timeout=TIMEOUT_SECONDS * 1000,
        retry_options=types.HttpRetryOptions(attempts=ATTEMPTS, initial_delay=1.0, max_delay=4.0,
                                             http_status_codes=[408, 429, 500, 502, 503, 504])))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        # the schema constrains the reply, so it parses into schema
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
    )
    models = [MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL else [])
    for i, model in enumerate(models):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            if not response.text:
                raise ValueError("Gemini returned no text (the reply may have been blocked)")
            return schema.model_validate_json(response.text)
        except Exception as e:
            if i == len(models) - 1:
                raise
            log.warning("Gemini model %s failed (%s: %s), trying %s", model, type(e).__name__, e, models[i + 1])
