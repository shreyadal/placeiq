# PlaceIQ

A mock technical interview app for college placements. The candidate answers out loud (or
types). The answer is transcribed (Whisper), graded against the question's key points, and
measured for pace, filler words and pauses. Eye contact and head movement come from face
tracking in the browser (MediaPipe). A trained knowledge tracing model (DKT) estimates the
candidate's mastery of each concept (DSA, OS, DBMS, CN, OOP, ML), and the next question
targets the weakest ones.

- `backend/`: FastAPI. Sessions, question selection, grading, speech metrics, and the `/model` endpoints.
- `frontend/`: Next.js. The interview, the results page and the `/model` page.
- `ml/`: the DKT model (via [pykt](https://github.com/pykt-team/pykt-toolkit)): training, testing,
  prediction. Metrics are in `ml/artifacts/metrics.json`; background in [ml/RESULTS.md](ml/RESULTS.md).
- `data/`: the question bank and the prepared datasets (see [data/README.md](data/README.md)).
- `tests/`: backend and model tests. `scripts/`: the demo seeder.

## Prototype status

This is a prototype for review. The DKT model is trained, tested and used by the app to pick
questions, and the `/model` page shows its accuracy. It was trained on ASSISTments 2009 and on
**simulated** interview data, has **not yet been validated on real students**, and is a
**first, un-tuned run**. On the simulated candidates a plain average of each candidate's scores
still finds their weakest concept more often than this DKT does (see the `/model` page).

## How to run the backend

Needs Python 3.11. From the repo root:

```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

uvicorn backend.main:app --port 8000
```

The backend loads the DKT checkpoint `ml/saved_model/interview/dkt/` at startup. Checkpoints are
not committed, so train the model once first (next section). Without a checkpoint the app still
runs and every question is chosen by the heuristic. `GET /health` shows `dkt_loaded`.

Model endpoints:

| endpoint | returns |
|---|---|
| `GET /model/metrics` | `ml/artifacts/metrics.json` as it is |
| `POST /model/predict` | predicted mastery for the 6 concepts from `{"history": [{"concept", "score", "timestamp"}, ...]}` |
| `GET /model/candidates` | ids of the 80 held-out simulated candidates |
| `GET /model/check/{candidate_id}` | predicted mastery next to true ability for one held-out simulated candidate |

## How to run the frontend

Needs Node 20+. With the backend running:

```
cd frontend
npm ci
npm run build
npm start                                     # http://localhost:3000
```

Pages: `/` starts an interview, `/results` shows mastery and past sessions, `/model` shows the
model's accuracy and a check on simulated candidates. To fill the results page:

```
python scripts/seed_demo.py --user demo --sessions 5    # then open http://localhost:3000/results?user=demo
```

## How questions are chosen

With `PLACEIQ_USE_DKT=1` (the default), the next concept is drawn with weight (1 − mastery)², where
mastery is DKT's predicted chance of a good answer from the candidate's whole history. The
difficulty follows the mastery: easy below 40%, medium below 70%, hard above. The
average-score heuristic (current session's scores) is used instead when the checkpoint is missing,
when the candidate has fewer than 3 answers, or when `PLACEIQ_USE_DKT=0`. Every `/sessions/{id}/next`
response says which one was used: `"source": "dkt"` or `"heuristic"`, with `source_reason`.
The interview and results pages show the same label.

## How to retrain and test the model

From the repo root with the venv active (`pip install pykt-toolkit scipy` if not installed):

```
set WANDB_MODE=disabled
python -m ml.sequences assist2009             # prepared split -> pykt's windowed files
python -m ml.sequences interview
python -m ml.train --model dkt --dataset interview  --epochs 30 --fold 0
python -m ml.train --model dkt --dataset assist2009 --epochs 30 --fold 0
python -m ml.test_model                       # predicted vs true weak concepts, held-out candidates
```

Fold 0 is validation, folds 1–4 are training, and the 20% of students in `test.csv` are held out.
Training stops 10 epochs after the best validation AUC (max 30). Each command writes its
numbers, and the command itself, to `ml/artifacts/metrics.json`; the logs are in `ml/logs/`. On a
laptop CPU the interview run takes under a minute and the ASSISTments run about 5 minutes.

## Tests

```
python -m pytest tests                        # backend, model endpoints, metrics consistency
cd frontend && npx tsc --noEmit && npm run build
```

The model tests are skipped if no checkpoint has been trained. One test fails if
`/model/metrics` differs from `metrics.json`, or if `metrics.json` does not match the checkpoints
on disk.

## Graders

Set `PLACEIQ_GRADER` (environment variable or a line in `.env`):

- `keyword` (default): the share of key points whose words appear in the answer. Needs nothing else.
- `gemini`: Gemini checks each key point against the answer and writes short feedback. Needs
  `GEMINI_API_KEY` in the environment or in `.env` at the repo root (gitignored; never commit a
  key). If the key is missing or the call fails, the answer is graded by keywords and the
  response says so (`fallback_from`, `fallback_reason`). With Gemini on, answers are sent to Google.
- `ollama`: like gemini, with a local model served by [Ollama](https://ollama.com).

Other settings: `GEMINI_MODEL` (default `gemini-3.5-flash`), `GEMINI_FALLBACK_MODEL` (default
`gemini-3.5-flash-lite`, tried when the first is busy), `WHISPER_MODEL` (default `base.en`).

## Not built yet

- **Gemini grading in production use.** The provider exists and is optional, but keyword grading is
  the default. Gemini's grades have not been checked against human grades.
- **RAG for company-style questions.** A prototype is in `rag/` (choosing a company on the start
  page uses it; with a Gemini key it writes the question). It is not evaluated and is not part of this review.
- **A real database.** Sessions are stored in a local SQLite file (`data/placeiq.db`). There are no
  accounts and no server database.
- **Real data.** The model has not been trained or validated on answers from real students.
  Answers stored by the app are not fed back into training.
- **Deployment.** The app runs only locally.
- **Face metrics with a real face.** The automated browser test used Chrome's fake camera.
