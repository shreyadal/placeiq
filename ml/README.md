# ml/ — knowledge tracing

Knowledge tracing answers one question: given everything a student has answered so far,
how likely are they to get the next question on concept X right? PlaceIQ uses that to pick
the concepts a candidate is weakest at.

The models come from [pykt](https://github.com/pykt-team/pykt-toolkit), installed with pip.
Everything in this folder is the code around it: preprocessing, training, the interview
data format and the prediction function the backend calls.

## Run it

All commands run from the project root with the venv active.

```
python -m ml.preprocess assist2009                       # raw csv -> pykt files
python -m ml.train --model dkt  --dataset assist2009     # then sakt, then akt
python -m ml.fake_interviews                             # fake interview logs -> data/interview/logs.csv
python -m ml.interview_data                              # logs.csv -> pykt files
python -m ml.train --model dkt  --dataset interview
python -m ml.predict u0001                               # weakest concepts for one user
```

Scores are written to [results.md](results.md).

## The data format pykt expects

There are three stages. Only the first is ours to produce; pykt makes the other two.

### 1. `data.txt` — six lines per student

```
64525,561                      <- user id, number of interactions
54003,53991,54071,...          <- question ids, in time order
278,278,278,...                <- concept ids (one question with two concepts: 12_45)
0,1,0,1,...                    <- responses, must be 0 or 1
NA                             <- timestamps (ms), or NA if unknown
NA                             <- time spent per question, or NA
```

Any line can be `NA` if the dataset lacks that field, except responses. For ASSISTments 2009
pykt's own converter writes this file from the raw csv (`process_raw_data`). For interview
data we write it ourselves in [interview_data.py](interview_data.py).

### 2. `train_valid_sequences.csv` and `test_sequences.csv` — what the models read

`pykt.preprocess.split_datasets.main` reads `data.txt` and does four things:

1. **Expands multi-concept questions.** A question tagged `12_45` becomes two rows, one per
   concept, both with the same response. The second row is marked `is_repeat = 1`.
2. **Maps ids to indices.** Raw ids become 0..N-1 in order of first appearance. The mapping
   is saved to `keyid2idx.json`, which we need at prediction time to turn "DBMS" into an index.
3. **Splits students.** 20% of students are held out as test (`fold = -1`); the rest are
   divided into 5 folds. The split is by student, so no student is in both train and test.
4. **Cuts and pads.** Each history is cut into windows of 200 interactions. The last window
   is padded with `-1`. Windows shorter than 3 are dropped.

Each row of the csv is one window, with comma-joined strings in every cell:

| column | meaning |
|---|---|
| `fold` | 0-4 for train/valid, -1 for test |
| `uid` | student index |
| `questions` | question indices, padded with -1 |
| `concepts` | concept indices, padded with -1 |
| `responses` | 0/1, padded with -1 |
| `selectmasks` | 1 where the position counts toward the loss and AUC, -1 on padding |
| `is_repeat` | 1 for the extra rows created in step 1 |

### 3. Tensors — what a model sees per batch

`KTDataset` turns each row into a dict of tensors. The key idea is the **shift by one**:
the model reads interactions `0..n-2` and is scored on predicting responses `1..n-1`.

| key | content |
|---|---|
| `cseqs`, `qseqs`, `rseqs` | concepts, questions, responses at positions 0..n-2 |
| `shft_cseqs`, `shft_qseqs`, `shft_rseqs` | the same at positions 1..n-1 (the targets) |
| `masks` | false where either side of the shift is padding |
| `smasks` | which target positions are scored |

DKT, for example, embeds each (concept, response) pair as one id `concept + num_c * response`,
runs an LSTM over the sequence, and outputs a probability for **every** concept at every
step. Training picks out the probability of the concept that actually came next
(`shft_cseqs`) and compares it with the real response (`shft_rseqs`).

`data_config.json` in `data/` ties this together: for each dataset it stores the folder, the
file names, and `num_q` / `num_c`, which set the size of the embedding tables.

## Three things pykt 0.0.38 needed on this setup

- **CPU.** Its data loader builds tensors with `torch.cuda.FloatTensor`. `patch_pykt_for_cpu`
  in [common.py](common.py) swaps in the CPU types.
- **Encoding.** The ASSISTments csv is not valid utf-8, so [preprocess.py](preprocess.py)
  re-encodes a copy first.
- **Newer pandas.** `groupby(['user_id'])` now yields tuple keys, so pykt writes the user id
  as `('64525',)`. `fix_user_id_lines` repairs those lines in `data.txt`.

## Why there are two AUC columns

Because of step 1 above, a question with two concepts gives the model the answer on the
first row and then asks it to predict the same answer on the second. That is a label leak
and it inflates AUC by several points. `test AUC (no repeats)` in results.md scores only
`is_repeat = 0` rows and is the number to quote.

## Interview data

One row per answered question, in `data/interview/logs.csv`:

| column | type | meaning |
|---|---|---|
| `user_id` | string | candidate |
| `question_id` | string | id from `data/questions.json` |
| `concept` | string | DSA, OS, DBMS, CN, OOP or ML |
| `score` | float 0-1 | how good the answer was (from the grader) |
| `timestamp` | int | unix time in milliseconds |

pykt models need a 0/1 response, so the converter treats `score >= 0.5` as correct. Each
question has exactly one concept, so there are no repeat rows and no leak in this dataset.

`predict_weak_concepts(user_id)` in [predict.py](predict.py) runs the trained DKT over the
user's history and reads the output of the last step: one probability per concept. The
lowest ones are the weak concepts.
