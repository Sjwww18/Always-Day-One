# app/utils/cli.py

import os
import pickle
import argparse
from turtle import mode
import numpy as np
import pandas as pd
from typing import Any, List, Tuple

from app.utils.filepath import get_back_path, get_data_path


# ===== assemble =====
def assemble(result: List[Tuple[Any, np.ndarray]], modelname: str, mode: str="eval") -> str:
    EqtyPath = "EqtyData.pkl"
    EqtyData = pd.read_pickle(get_data_path(EqtyPath))
    
    KEYS, VALS = zip(*result)
    DATES, INTES = zip(*KEYS)
    COMBO_np = np.concatenate(VALS, axis=0)
    
    idx = pd.MultiIndex.from_arrays(
        [DATES, INTES],
        names=["date", "interval"]
    )
    COMBO = pd.DataFrame(
        COMBO_np,
        index=idx,
        columns=EqtyData
    )
    
    stem = os.path.splitext(modelname)[0]
    comboname = f"{mode}_{stem}.pkl" if mode != "eval" else f"{stem}.pkl"
    
    combopath = get_back_path(comboname)
    COMBO.to_pickle(combopath)
    
    return combopath


# ===== parse args =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="ictrain.yaml",
        help="Config file name under app/cfgs/"
    )
    return parser.parse_args()


# end of app/utils/cli.py