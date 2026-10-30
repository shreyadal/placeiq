"""Paths and settings shared by every script in ml/, plus one fix needed to run pykt on a CPU."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_PATH = DATA_DIR / "data_config.json"
CKPT_DIR = ROOT / "ml" / "saved_model"
RESULTS_JSON = ROOT / "ml" / "results.json"

# pykt cuts every student's history into windows of MAXLEN interactions and pads the last one with -1
MAXLEN = 200
MIN_SEQ_LEN = 3
KFOLD = 5
# "qid" tells pykt models to learn their own id embeddings (as opposed to loading pretrained ones)
EMB_TYPE = "qid"

# keyword arguments passed to each pykt model class. AKT is smaller than the paper's
# setting (d_model 256, 4 blocks) so that it trains on a laptop CPU.
MODEL_CONFIGS = {
    "dkt": {"dropout": 0.2, "emb_size": 200},
    "sakt": {"dropout": 0.2, "emb_size": 200, "num_attn_heads": 8, "num_en": 2, "seq_len": MAXLEN},
    "akt": {"dropout": 0.2, "d_model": 128, "d_ff": 256, "num_attn_heads": 8, "n_blocks": 2, "final_fc_dim": 256},
}
LEARNING_RATES = {"dkt": 1e-3, "sakt": 1e-3, "akt": 1e-3}


def patch_pykt_for_cpu():
    """pykt's data loader builds tensors with torch.cuda.FloatTensor, which raises on a
    machine without a GPU. Point those two names at the CPU tensor types instead."""
    if torch.cuda.is_available():
        return
    import pykt.datasets.data_loader as data_loader

    data_loader.FloatTensor = torch.FloatTensor
    data_loader.LongTensor = torch.LongTensor


def load_data_config():
    """Read data/data_config.json. The stored dpath is whatever machine ran the
    preprocessing, so it is replaced with this machine's data folder."""
    with open(CONFIG_PATH, encoding="utf8") as f:
        config = json.load(f)
    for name in config:
        config[name]["dpath"] = (DATA_DIR / name).as_posix()
    return config
