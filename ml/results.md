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
| interview | DKT | 0.8135 | 0.8135 | 0.7470 | 0.7852 | 9 / 40 | 0.8 | dropout=0.2, emb_size=200, lr=0.001, batch=64 |
| assist2009 | SAKT | 0.7200 | 0.7088 | 0.6985 | 0.7279 | 8 / 30 | 30.8 | dropout=0.2, emb_size=200, num_attn_heads=8, num_en=2, lr=0.001, batch=64 |
