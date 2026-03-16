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
        fillna: str = "zero",
        dffilter: Optional[str] = None,
        normalize: Optional[str] = None,
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

        X = torch.from_numpy(self.all_X[start:end])

        if self.all_y is not None:
            y = torch.from_numpy(self.all_y[start:end])
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


# end of app/loader/stockdata.py