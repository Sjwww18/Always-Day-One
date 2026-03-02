# app/utils/cli.py

import os
import random
import argparse
import numpy as np
import pandas as pd
from typing import Any, List, Tuple

import torch

from app.utils.filepath import get_back_path, get_data_path


# ===== assemble =====
def assemble(result: List[Tuple[Tuple[Any], np.ndarray]], modelname: str, by: str, mode: str="eval") -> str:
    EqtyPath = "EqtyData.pkl"
    EqtyData = pd.read_pickle(get_data_path(EqtyPath))

    # ===== interval 级 =====
    if by == "interval":
        KEYS, VALS = zip(*result)
        DATES, INTERVALS = zip(*KEYS)
        COMBO_np = np.concatenate(VALS, axis=0)

        idx = pd.MultiIndex.from_arrays(
            [DATES, INTERVALS],
            names=["date", "interval"]
        )

    # ===== date 级 =====
    elif by == "date":
        DATES_ls = [d for d, y in result]
        COMBO_np = np.stack([y for _, y in result], axis=0)
        COMBO_np = COMBO_np.reshape(-1, COMBO_np.shape[-1])

        DATES_np = np.repeat(DATES_ls, 51)
        INDEX_np = np.tile(np.arange(51), len(DATES_ls))
        idx = pd.MultiIndex.from_arrays(
            [DATES_np, INDEX_np],
            names=["date", "interval"]
        )
    
    # ===== 未定义 级 =====
    else:
        raise ValueError(f"Unknown by mode: {by}.")

    COMBO = pd.DataFrame(
        COMBO_np,
        index=idx,
        columns=EqtyData
    )

    stem = os.path.splitext(modelname)[0]
    comboname = f"{mode}_{stem}.pkl" if mode != "eval" else f"{stem}.pkl"
    combopath = get_back_path(comboname)
    COMBO.to_pickle(combopath)
    
    return comboname


# ===== parse args =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="ictrain.yaml",
        help="Config file name under app/cfgs/"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model file name under sota/"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint (e.g., latest.ckpt, best.ckpt)"
    )
    return parser.parse_args()


# ===== set seed =====
def set_seed(seed: int = 42):
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)
    # Torch CPU / GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN reproducibility (important for CNN / LSTM)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Enforce deterministic algorithms (PyTorch 1.8+)
    # torch.use_deterministic_algorithms(True)


# optional: only needed when num_workers > 0
def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# end of app/utils/cli.py