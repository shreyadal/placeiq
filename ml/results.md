# Knowledge tracing results

AUC of predicting whether the next answer is correct. Validation is fold 0 of the
training students; test is the 20% of students pykt holds out. All runs on CPU.

- **test AUC** is pykt's default: every (question, concept) row is scored.
- **test AUC (no repeats)** scores only the first row of each question. A question with
  several concepts becomes several rows with the same response, so on the later rows the
  model has already seen the answer. This column removes that leak and is the fairer number.

| dataset | model | test AUC | test AUC (no repeats) | test acc | valid AUC | best epoch / max | train time (min) | settings |
|---|---|---|---|---|---|---|---|---|
| assist2009 | DKT | 0.8230 | 0.7519 | 0.7652 | 0.8317 | 6 / 30 | 4.6 | dropout=0.2, emb_size=200, lr=0.001, batch=64 |
