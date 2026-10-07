"""Transcribe a spoken answer with Whisper and measure how it was delivered."""
import os
import re
from functools import lru_cache

# base.en is about 140 MB and transcribes faster than real time on a laptop CPU
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

# Whisper tends to clean fillers out of the transcript. Starting it off with a prompt
# that contains them makes it keep them.
FILLER_PROMPT = "Umm, so, uh, basically, like, you know, I mean, hmm, actually it works like this."

# "like" also has real uses ("looks like"), so its count is an upper bound
SINGLE_WORD_FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "mm", "like", "basically", "actually",
                       "literally"}
PHRASE_FILLERS = [("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")]

# a silence between two words longer than this counts as a pause
PAUSE_SECONDS = 0.7


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    # int8 quantisation: about 4x smaller and faster on CPU for a small accuracy cost
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe(audio_path):
    """Returns (transcript, words, duration_seconds); words is a list of {"word", "start", "end"}."""
    segments, info = _model().transcribe(audio_path, word_timestamps=True, initial_prompt=FILLER_PROMPT)
    words, texts = [], []
    for segment in segments:  # a generator: decoding happens as it is consumed
        texts.append(segment.text.strip())
        for w in segment.words or []:
            words.append({"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)})
    return " ".join(texts), words, round(info.duration, 2)


def _normalise(word):
    return re.sub(r"[^a-z']", "", word.lower())


def count_fillers(tokens):
    """Count filler words and phrases. Returns {"um": 3, "you know": 1, ...}."""
    counts = {}
    for token in tokens:
        if token in SINGLE_WORD_FILLERS:
            counts[token] = counts.get(token, 0) + 1
    for first, second in PHRASE_FILLERS:
        n = sum(1 for a, b in zip(tokens, tokens[1:]) if (a, b) == (first, second))
        if n:
            counts[f"{first} {second}"] = n
    return counts


def speech_metrics(words, duration):
    """Delivery metrics from word timestamps.

    - words_per_minute: words divided by the time from the first word to the last, so
      silence before starting or after finishing does not count against the speaker.
      Conversational English is roughly 120-160.
    - fillers: see count_fillers.
    - pauses: gaps between consecutive words longer than PAUSE_SECONDS.
    """
    tokens = [t for t in (_normalise(w["word"]) for w in words) if t]
    if not tokens:
        return {"word_count": 0, "duration_seconds": duration, "speaking_seconds": 0, "words_per_minute": 0,
                "filler_count": 0, "fillers": {}, "filler_rate": 0, "pause_count": 0, "longest_pause_seconds": 0,
                "average_pause_seconds": 0, "total_pause_seconds": 0}

    speaking_seconds = words[-1]["end"] - words[0]["start"]
    wpm = len(tokens) / (speaking_seconds / 60) if speaking_seconds > 0 else 0

    fillers = count_fillers(tokens)
    filler_count = sum(fillers.values())

    pauses = [b["start"] - a["end"] for a, b in zip(words, words[1:])]
    pauses = [p for p in pauses if p > PAUSE_SECONDS]

    return {
        "word_count": len(tokens),
        "duration_seconds": duration,
        "speaking_seconds": round(speaking_seconds, 2),
        "words_per_minute": round(wpm),
        "filler_count": filler_count,
        "fillers": fillers,
        # fillers per 100 words, so long and short answers can be compared
        "filler_rate": round(100 * filler_count / len(tokens), 1),
        "pause_count": len(pauses),
        "longest_pause_seconds": round(max(pauses), 2) if pauses else 0,
        "average_pause_seconds": round(sum(pauses) / len(pauses), 2) if pauses else 0,
        "total_pause_seconds": round(sum(pauses), 2),
    }


def analyze(audio_path):
    transcript, words, duration = transcribe(audio_path)
    return {"transcript": transcript, "metrics": speech_metrics(words, duration)}
