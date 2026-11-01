"""Train one knowledge tracing model with pykt and record its AUC.

    python -m ml.train --model dkt --dataset assist2009 --epochs 30

Fold 0 is used for validation and the other four for training. The checkpoint with the
best validation AUC is kept, then scored once on the held-out 20% test students.
"""
import argparse
import ast
import json
import sys
import time

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from pykt.datasets import init_dataset4train
from pykt.models import evaluate, init_model, load_model, train_model
from pykt.utils import set_seed

from ml.common import (CKPT_DIR, EMB_TYPE, LEARNING_RATES, METRICS_JSON, MODEL_CONFIGS, RESULTS_JSON,
                       load_data_config, patch_pykt_for_cpu, update_metrics)

# pykt's train_model stops once this many epochs pass without a better validation AUC
EARLY_STOP_PATIENCE = 10


def train(model_name, dataset, epochs, fold=0, batch_size=64, seed=42, eval_only=False):
    patch_pykt_for_cpu()
    set_seed(seed)

    data_config = load_data_config()
    train_loader, valid_loader, test_loader, _ = init_dataset4train(dataset, model_name, data_config, fold, batch_size)

    model_config = MODEL_CONFIGS[model_name]
    ckpt_path = CKPT_DIR / dataset / model_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    result_path = ckpt_path / "result.json"

    if eval_only:
        result = json.loads(result_path.read_text(encoding="utf8"))
    else:
        model = init_model(model_name, dict(model_config), data_config[dataset], EMB_TYPE)
        opt = torch.optim.Adam(model.parameters(), LEARNING_RATES[model_name])
        start = time.time()
        # save_model=True writes the best-validation-AUC weights to ckpt_path; pykt stops
        # early once 10 epochs pass without improvement
        *_, valid_auc, valid_acc, best_epoch = train_model(
            model, train_loader, valid_loader, epochs, opt, str(ckpt_path), None, None, True)
        result = {
            "dataset": dataset, "model": model_name,
            "valid_auc": round(float(valid_auc), 4), "valid_acc": round(float(valid_acc), 4),
            "best_epoch": int(best_epoch), "max_epochs": epochs,
            "epochs_run": min(int(best_epoch) + EARLY_STOP_PATIENCE, epochs),
            "train_seconds": round(time.time() - start, 1),
            "train_minutes": round((time.time() - start) / 60, 1),
            "config": model_config, "lr": LEARNING_RATES[model_name], "batch_size": batch_size, "seed": seed,
        }

    best = load_model(model_name, dict(model_config), data_config[dataset], EMB_TYPE, str(ckpt_path))
    pred_file = ckpt_path / "test_predictions.txt"
    test_auc, test_acc = evaluate(best, test_loader, model_name, str(pred_file))
    test_csv = f"{data_config[dataset]['dpath']}/{data_config[dataset]['test_file']}"
    result.update({
        "test_auc": round(float(test_auc), 4), "test_acc": round(float(test_acc), 4),
        "test_auc_no_repeat": round(auc_without_repeats(pred_file, test_csv), 4),
    })
    pred_file.unlink()

    result_path.write_text(json.dumps(result, indent=2), encoding="utf8")
    record_result(result)
    if model_name == "dkt":
        update_metrics(dataset, {**result, **dataset_sizes(data_config[dataset]),
                                 "command": "python -m ml.train " + " ".join(sys.argv[1:])})
        print(f"wrote {METRICS_JSON}")
    print(json.dumps(result, indent=2))
    return result


def auc_without_repeats(pred_file, test_csv):
    """AUC over only the first row of each question.

    pykt turns a question tagged with k concepts into k consecutive rows that all carry the
    same response. On rows 2..k the model has already been shown the answer it is asked to
    predict, which inflates the plain AUC. Those rows are marked is_repeat=1 in the test csv,
    so scoring only is_repeat=0 rows gives a number without that leak.
    """
    test_df = pd.read_csv(test_csv)
    trues, preds = [], []
    with open(pred_file, encoding="utf8") as f:
        # evaluate() writes one line per test sequence, in the same order as the csv
        for line, repeats in zip(f, test_df["is_repeat"]):
            # the per-sequence auc field is nan when a sequence has only one class
            row = ast.literal_eval(line.replace("nan", "None"))
            targets, scores = row[3], row[4]
            # predictions start from the second interaction of the sequence
            repeats = repeats.split(",")[1:1 + len(targets)]
            for target, score, repeat in zip(targets, scores, repeats):
                if repeat == "0":
                    trues.append(target)
                    preds.append(score)
    return float(roc_auc_score(trues, preds))


def dataset_sizes(config):
    """Students and answers in each part of the split. Fold 0 of train_valid.csv is the
    validation set, the other folds are trained on.

    An ASSISTments answer tagged with several skills is several rows; is_repeat marks the
    extra ones, so answers counts only rows with is_repeat = 0.
    """
    train_valid = pd.read_csv(f"{config['dpath']}/{config['train_valid_file']}")
    test = pd.read_csv(f"{config['dpath']}/{config['test_file']}")
    sizes = {}
    for part, df in (("train", train_valid[train_valid["fold"] != 0]),
                     ("valid", train_valid[train_valid["fold"] == 0]), ("test", test)):
        sizes[f"{part}_students"] = int(df["uid"].nunique())
        sizes[f"{part}_answers"] = int(sum(r.split(",").count("0") for r in df["is_repeat"]))
    return sizes


def record_result(result):
    """Keep one row per (dataset, model) in results.json; ml/RESULTS.md reports from it."""
    results = json.loads(RESULTS_JSON.read_text(encoding="utf8")) if RESULTS_JSON.exists() else []
    results = [r for r in results if (r["dataset"], r["model"]) != (result["dataset"], result["model"])]
    results.append(result)
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--dataset", default="assist2009")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true", help="re-score the saved checkpoint without training")
    args = parser.parse_args()
    train(args.model, args.dataset, args.epochs, args.fold, args.batch_size, args.seed, args.eval_only)
