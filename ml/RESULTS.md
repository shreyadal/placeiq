# DKT results

> **First, un-tuned run:** a single fold (fold 0 for validation), default hyperparameters
> (dropout 0.2, embedding size 200, lr 0.001, batch 64, seed 42). This is not an optimised
> benchmark: no cross-validation, no hyperparameter search.

The numbers below were first produced on 2026-10-28 and reproduced exactly by a retrain on
2026-11-01 (training times are from that retrain) on a laptop CPU (Windows 11, Python 3.11.9,
torch 2.14.1+cpu, pykt-toolkit 0.0.38). The raw output of each run is in [logs/](logs/), and the
training and test numbers are also in [results.json](results.json).

## Next-answer prediction (held-out test students)

| dataset | test AUC | test AUC, repeats excluded | test accuracy | valid AUC | best epoch | stopped at | training time |
|---|---|---|---|---|---|---|---|
| ASSISTments 2009 | 0.8230 | 0.7519 | 0.7652 | 0.8317 | 6 | epoch 16 of max 30 | 259.2 s (4 min 44 s wall clock) |
| interview (simulated) | 0.8135 | 0.8135 | 0.7470 | 0.7852 | 9 | epoch 19 of max 30 | 21.0 s (37 s wall clock) |

- **Test** is the 20% of students held out by `data/prepare.py` (770 ASSISTments students,
  80 simulated candidates). None of them was used for training or for choosing the epoch.
- **Repeats excluded** scores only rows with `is_repeat = 0`. An ASSISTments answer tagged
  with several skills becomes several rows with the same response, so on the extra rows the
  model has already seen the answer. 9,662 of the ASSISTments test rows are such repeats. Every interview
  question has one concept, so there are no repeats there and both columns are equal.
- **Early stopping:** pykt stops after 10 epochs without a better validation AUC and keeps
  the best epoch's weights.
- **Training time** is what `ml/train.py` measures around training. The wall-clock time covers the
  whole command, including data loading and the test pass. All of these numbers, with the
  dataset sizes, are also in `ml/artifacts/metrics.json`, which the app's `/model` page shows.

## Does the model find each candidate's weak concepts?

`python -m ml.test_model` runs the interview checkpoint over each of the 80 held-out simulated
candidates' full history. It compares the predicted chance of a good answer per concept with the
candidate's true ability, which comes from the simulator. The simulator reproduces
`data/interview_simulated.csv` row for row, which the script checks before using it.

| method | mean Spearman (6 concepts) | median Spearman | true weakest concept in predicted 2 weakest |
|---|---|---|---|
| DKT | 0.667 | 0.743 | 54 / 80 = 67.5% |
| average score per concept (the app's fallback heuristic) | 0.831 | 0.829 | 71 / 80 = 88.8% |
| random guess | 0 | 0 | 33.3% |

**Reading this honestly:** DKT ranks concepts far better than chance, but in this first run it
is *worse* than simply averaging the candidate's scores per concept. One likely reason: DKT only
sees each answer as right or wrong (score ≥ 0.5), while the average uses the full 0–1 score.
The app now uses DKT by default (`PLACEIQ_USE_DKT=1`) so that the model is exercised end to end,
and falls back to the average when the model is missing or a candidate has fewer than 3 answers.
Improving the model past this baseline is the next step.

The script also prints a per-concept table for the first 5 candidates; see
[logs/test_model.log](logs/test_model.log).

## Exact commands

From the repo root, with the venv active (`venv\Scripts\activate`):

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pykt-toolkit scipy

# data/prepare.py already produced data/<name>/train_valid.csv and test.csv;
# this cuts them into pykt's 200-answer windows and writes data/data_config.json
python -m ml.sequences assist2009
python -m ml.sequences interview

set WANDB_MODE=disabled
python -m ml.train --model dkt --dataset interview  --epochs 30 --fold 0
python -m ml.train --model dkt --dataset assist2009 --epochs 30 --fold 0
python -m ml.test_model
```

Checkpoints are written to `ml/saved_model/<dataset>/dkt/` and are not committed. Re-running
gives the same numbers, because the seed, split and code are fixed.
