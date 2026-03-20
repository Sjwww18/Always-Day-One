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
    """
    Stock-level dataset for CNN / sequence models.

    One sample corresponds to one (date, stock):
        X: [51, F_feature]
        y: [51, L_label] or None
        mask: [51, L_label], 1 if label valid else 0
    """
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

        Xdf = df[self.features].astype(np.float32)

        if self.fillna == "zero":
            Xdf = Xdf.fillna(np.float32(0.0))
        elif self.fillna == "mean":
            grouped = df.groupby(["date", "interval"])[self.features]
            mean_df = grouped.transform("mean").astype(np.float32)
            Xdf = Xdf.fillna(mean_df).fillna(np.float32(0.0))

        if self.normalize == "zscore":
            grouped = pd.concat([df[["date", "interval"]], Xdf], axis=1).groupby(["date", "interval"])[self.features]
            mean_df = grouped.transform("mean").astype(np.float32)
            std_df = grouped.transform("std").astype(np.float32).replace(0.0, 1.0)
            Xdf = (Xdf - mean_df) / std_df

        self.all_X = Xdf.to_numpy(dtype=np.float32, copy=False)
        self.all_y = df[self.label].to_numpy(dtype=np.float32, copy=False) if self.label else None

        key_df = df[["date", "stock"]]
        self.keys = [tuple(x) for x in key_df.to_numpy()[::51]]
        self.n_samples = len(self.keys)

        del df, Xdf, key_df
        gc.collect()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tuple, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        start = idx * 51
        end = start + 51
        key = self.keys[idx]

        X = torch.from_numpy(self.all_X[start: end])

        if self.all_y is not None:
            y = torch.from_numpy(self.all_y[start: end])
            mask = (~torch.isnan(y)).float()
        else:
            y = None
            mask = None

        return key, X, y, mask

    def process(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(-1).cpu().to(dtype=torch.float32)


# app/loader/dev.py

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
    """
    Stock-level dataset for CNN / sequence models.

    One sample corresponds to one (date, stock):
        X: [51, F_feature]
        y: [51, L_label] or None
        mask: [51, L_label], 1 if label valid else 0
    """
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

        if dffilter is not None:
            df = df.query(dffilter)

        self.features = features
        self.label = label

        # ===== X =====
        X = df[features].to_numpy(dtype=np.float32, copy=False)
        if fillna == "zero":
            np.nan_to_num(X, copy=False)
        elif fillna == "mean":
            # 按 (date, interval) 求均值再填充
            Xdf = df[features].to_numpy()
            keys, counts = np.unique(df[["date","interval"]].to_numpy(), axis=0, return_counts=True)
            for k in keys:
                mask = (df["date"]==k[0]) & (df["interval"]==k[1])
                Xdf[mask] = np.nan_to_num(Xdf[mask], nan=Xdf[mask].mean(axis=0))
            X = Xdf.astype(np.float32)

        n = len(df)
        assert n % 51 == 0, "data not divisible by 51"
        N = n // 51
        F = len(features)
        X = X.reshape(N, 51, F)

        # ===== normalize =====
        if normalize == "zscore":
            # 按 interval 截面标准化
            for t in range(51):
                interval_slice = X[:, t, :]  # shape (N, F)
                mean = np.nanmean(interval_slice, axis=0, keepdims=True)
                std = np.nanstd(interval_slice, axis=0, keepdims=True)
                std[std==0] = 1.0
                interval_slice = (interval_slice - mean) / std
                X[:, t, :] = interval_slice

        self.X = torch.from_numpy(X)

        # ===== y & mask =====
        if label:
            y = df[label].to_numpy(dtype=np.float32, copy=False).reshape(N, 51, -1)
            mask = ~np.isnan(y)
            y = np.nan_to_num(y, nan=0.0)
            self.y = torch.from_numpy(y)
            self.mask = torch.from_numpy(mask.astype(np.float32))
        else:
            self.y = None
            self.mask = None

        # ===== keys =====
        self.keys = df[["date", "stock"]].to_numpy()[::51]
        self.n_samples = N

        del df
        gc.collect()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.y is not None:
            return self.keys[idx], self.X[idx], self.y[idx], self.mask[idx]
        else:
            return self.keys[idx], self.X[idx], None, None

    def process(self, y):
        return y.squeeze(-1)

# end of app/loader/dev.py
# end of app/loader/stockdata.py