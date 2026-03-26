# app/loader/stockdata.py

import gc
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from app.core.registry import register_loader
from app.utils.filepath import get_data_path


@register_loader("stock")
class StockLoader(Dataset):
    def __init__(
        self,
        file: str,
        label: List[str],
        features: List[str],
        fillna: str="zero",
        dffilter: Optional[str]=None,
        normalize: Optional[str]=None
    ):
        path = get_data_path(file)
        cols = ["date", "stock", "interval"] + features + label
        df = pd.read_parquet(path, engine="pyarrow", columns=cols)

        self.label = label
        self.features = features

        if dffilter is not None:
            df = df.query(dffilter)

        # ===== reshape =====
        X = df[features].to_numpy(dtype=np.float32, copy=False)
        n = len(X)
        assert n % 51 == 0
        
        S = 3
        T = 51
        F = len(features)
        N = n // T  # = D * 5171
        D = N // S  # = date num, 3 is for debug
        X = X.reshape(D, S, T, F)  # [D, S, T, F], (date, stock, interval, feature)
        X = X.transpose(0, 1, 3, 2)  # (D, S, F, T)

        # ===== zscore =====
        if normalize == "zscore":
            mean = np.nanmean(X, axis=1, keepdims=True)
            std = np.nanstd(X, axis=1, keepdims=True)
            std[(std == 0) | np.isnan(std)] = 1.0
            X -= mean
            X /= std

        # ===== fillna =====
        if fillna == "mean":
            mask = np.isnan(X)
            mean = np.nanmean(X, axis=1, keepdims=True)
            X = np.where(mask, mean, X)
            np.nan_to_num(X, nan=0.0, copy=False)
        elif fillna == "zero":
            np.nan_to_num(X, nan=0.0, copy=False)

        self.X = torch.from_numpy(X)

        # ===== y =====
        if label:
            y = df[label].to_numpy(dtype=np.float32, copy=False)
            y = y.reshape(D, S, T, -1)  # (D, S, T, 1)
            # y = y.transpose(0, 1, 3, 2)  # (D, S, F, T)

            mask = ~np.isnan(y)
            np.nan_to_num(y, nan=0.0, copy=False)

            self.y = torch.from_numpy(y)
            self.mask = torch.from_numpy(mask.astype(np.float32))
        else:
            self.y = None
            self.mask = None

        # ===== key =====
        self.keys = df["date"].to_numpy()[:: S * T]
        self.n_samples = D

        del df

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tuple, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.y is not None:
            return self.keys[idx], self.X[idx], self.y[idx], self.mask[idx]
        else:
            return self.keys[idx], self.X[idx], None, None

    def process(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(-1)


# end of app/loader/stockdata.py