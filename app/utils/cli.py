# app/utils/cli.py

import os
import argparse
import numpy as np
import pandas as pd
from typing import Any, List, Tuple

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
    return parser.parse_args()


# end of app/utils/cli.py