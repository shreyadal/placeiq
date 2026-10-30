# PlaceIQ

A mock technical interview app for college placements. The candidate answers out loud. The
answer is transcribed (Whisper), graded against the question's key points, and measured
for pace, filler words and pauses. Eye contact and head movement come from face tracking in
the browser (MediaPipe). The next question targets the candidate's weakest concepts: DSA, OS,
DBMS, CN, OOP and ML.

The repo has two parts that are shown separately:

- **The app**, `backend/` (FastAPI) and `frontend/` (Next.js), runs end to end on its own.
- **The knowledge tracing model**, `ml/` (DKT via [pykt](https://github.com/pykt-team/pykt-toolkit)),
  is trained and tested on its own. Results are in [ml/RESULTS.md](ml/RESULTS.md).

Other folders: `data/` holds the question bank and the prepared datasets (see
[data/README.md](data/README.md)), `rag/` holds company interview experiences used for
company-style questions, `tests/` holds the backend tests and `scripts/` the demo seeder.

## How to run the app

Needs Python 3.11 and Node 20+. No GPU, Ollama or trained model is needed.

```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

uvicorn backend.main:app --port 8000          # terminal 1, from the repo root

cd frontend                                   # terminal 2
npm ci
npm run build
npm start                                     # http://localhost:3000
```

Then, with both running:

```
python scripts/seed_demo.py --user demo --sessions 5    # fills the results page for user "demo"
python -m pytest tests                                  # backend + RAG tests
```

The first spoken answer and the first company question are slower, because the Whisper and
embedding models download on first use.

Settings (environment variables for the backend):

| variable | default | meaning |
|---|---|---|
| `PLACEIQ_USE_DKT` | `0` | `0`: the next concept is chosen from the candidate's own average score per concept in the current session (weaker concept, more likely). `1`: chosen from the trained DKT model's predictions over the user's whole history |
| `PLACEIQ_GRADER` | `keyword` | `keyword`: the share of key points mentioned. `ollama`: a local LLM checks each key point (needs [Ollama](https://ollama.com) running with `OLLAMA_MODEL`, default `llama3.2`) |
| `WHISPER_MODEL` | `base.en` | faster-whisper model used for transcription |

`GET /health` reports which mastery source and grader are active.

## How to run the model

From the repo root with the venv active. The full steps and numbers are in [ml/RESULTS.md](ml/RESULTS.md).

```
pip install pykt-toolkit scipy
set WANDB_MODE=disabled
python -m ml.sequences assist2009             # prepared split -> pykt's windowed files
python -m ml.sequences interview
python -m ml.train --model dkt --dataset assist2009 --epochs 30 --fold 0
python -m ml.train --model dkt --dataset interview  --epochs 30 --fold 0
python -m ml.test_model                       # predicted vs true weak concepts, held-out candidates
```

On a laptop CPU the interview run takes about a minute and the ASSISTments run about 8 minutes.

## What is not connected yet

- **The model is not wired into the app by default.** The app chooses questions from the
  candidate's own average scores (`PLACEIQ_USE_DKT=0`). Setting `PLACEIQ_USE_DKT=1` makes the
  backend use the DKT checkpoint in `ml/saved_model/interview/dkt/`, which must be trained first.
  That path runs, but it has not been evaluated inside the app. On held-out simulated candidates
  the plain score average currently finds weak concepts better than this first, un-tuned DKT
  (see [ml/RESULTS.md](ml/RESULTS.md)).
- **The model is trained on simulated candidates,** not on answers from real users of the app.
  Answers stored by the app (`data/placeiq.db`) are not yet fed back into training.
- **LLM grading is off by default.** Keyword grading is approximate. Ollama grading is
  implemented but must be switched on with `PLACEIQ_GRADER=ollama`.
- **Face metrics were not tested with a real face** in this round: the automated browser test
  used Chrome's fake camera, which shows a test pattern.
