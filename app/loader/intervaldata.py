# app/loader/intervaldata.py

import gc
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from app.core.registry import register_loader
from app.utils.filepath import get_data_path


@register_loader("interval")
class IntervalLoader(Dataset):
    """
    Interval-level cross-sectional dataset.

    One batch corresponds to one (date, interval):
        X: [N_stock, F_feature]
        y: [N_stock, L_label] or None (test)
        mask: [N_stock, L_label], 1 if label valid else 0
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
        self.fillna = fillna
        self.normalize = normalize

        if dffilter is not None:
            df = df.query(dffilter)

        df[self.features] = df[self.features].astype(np.float32)

        if self.fillna == "zero":
            df[self.features] = df[self.features].fillna(np.float32(0.0))
        elif self.fillna == "mean":
            group_mean = df.groupby(["date", "interval"], sort=False)[self.features].transform("mean")
            df[self.features] = df[self.features].fillna(group_mean.astype(np.float32))
            df[self.features] = df[self.features].fillna(np.float32(0.0))

        if self.normalize == "zscore":
            group_mean = df.groupby(["date", "interval"], sort=False)[self.features].transform("mean")
            group_std = df.groupby(["date", "interval"], sort=False)[self.features].transform("std")
            group_std = group_std.replace(0.0, 1.0).fillna(1.0)

            df[self.features] = (
                (df[self.features] - group_mean) / group_std
            ).astype(np.float32)

        self.all_X = df[self.features].to_numpy(dtype=np.float32, copy=False)
        self.all_y = df[self.label].to_numpy(dtype=np.float32, copy=False) if self.label else None

        key_df = df[["date", "interval"]]
        group_id, keys = pd.factorize(list(zip(key_df["date"], key_df["interval"])), sort=False)

        order = np.argsort(group_id, kind="stable")
        group_id_sorted = group_id[order]

        group_start = np.flatnonzero(np.r_[True, group_id_sorted[1:] != group_id_sorted[:-1]])
        group_count = np.diff(np.r_[group_start, len(group_id_sorted)])

        self.keys = [keys[i] for i in range(len(keys))]
        self.row_indices = np.split(order, group_start[1:])

        del df, key_df, group_id, keys, order, group_id_sorted, group_start, group_count
        gc.collect()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[Tuple[datetime, int], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        key = self.keys[idx]
        row_ids = self.row_indices[idx]

        X = torch.from_numpy(self.all_X[row_ids])

        if self.all_y is not None:
            y = torch.from_numpy(self.all_y[row_ids])
            mask = (~torch.isnan(y)).float()
        else:
            y = None
            mask = None

        return key, X, y, mask

    def process(self, y: torch.Tensor) -> torch.Tensor:
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y
        return torch.tensor(y_np.reshape(1, -1), dtype=torch.float32)


# end of app/loader/intervaldata.py