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
        
        F = len(features)
        N = n // 51  # = D * 5171
        X = X.reshape(N, 51, F)  # [D * N, T, F]

        # ===== fillna =====
        if fillna == "mean":
            N_stock = 5171
            D = N // N_stock
            X4 = X.reshape(D, N_stock, 51, F)

            mask = np.isnan(X4)
            mean = np.nanmean(X4, axis=1, keepdims=True)
            X4 = np.where(mask, mean, X4)
            np.nan_to_num(X4, nan=0.0, copy=False)
            X = X4.reshape(N, 51, F)
        elif fillna == "zero":
            np.nan_to_num(X, nan=0.0, copy=False)

        # ===== zscore =====
        if normalize == "zscore":
            N_stock = 5171
            D = N // N_stock
            X4 = X.reshape(D, N_stock, 51, F)

            mean = X4.mean(axis=1, keepdims=True)
            std = X4.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            X4 = (X4 - mean) / std
            X = X4.reshape(N, 51, F)

        self.X = torch.from_numpy(X)

        # ===== y =====
        if label:
            y = df[label].to_numpy(dtype=np.float32, copy=False)
            y = y.reshape(N, 51, -1)

            mask = ~np.isnan(y)
            y = np.nan_to_num(y, nan=0.0)

            self.y = torch.from_numpy(y)
            self.mask = torch.from_numpy(mask.astype(np.float32))
        else:
            self.y = None
            self.mask = None

        # ===== key =====
        self.keys = df[["date", "stock"]].to_numpy()[:: 51]
        self.n_samples = N

        del df
        gc.collect()

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