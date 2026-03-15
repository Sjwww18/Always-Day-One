# app/loader/datedata.py

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from app.utils.helper import zscore
from app.core.registry import register_loader
from app.utils.filepath import get_data_path


@register_loader("date")
class DateLoader(Dataset):
    """
    Date-level cross-sectional dataset.

    One batch corresponds to one (date):
        X: [N_stock * M_interval, F_feature]
        y: [N_stock * M_interval, 1] or None (test)
        mask: [N_stock * M_interval, 1], 1 if label valid else 0
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

        self.data = {}
        for d, g in df.groupby(["date"], sort=False):
            X = g[self.features]

            if self.fillna == "zero":
                X = X.fillna(0.0)
            elif self.fillna == "mean":
                X = X.fillna(X.mean(axis=0))
            
            X = X.to_numpy(dtype="float32")

            if self.normalize == "zscore":
                X = zscore(X)

            if self.label:
                y = g[self.label].to_numpy(dtype="float32").reshape(-1, 1)
                mask = (~np.isnan(y)).astype("float32")
            else:
                y = None
                mask = None

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32) if y is not None else None
            mask_tensor = torch.tensor(mask, dtype=torch.float32) if mask is not None else None

            self.data[(d,)] = (X_tensor, y_tensor, mask_tensor)
        
        self.keys = list(self.data.keys())        
        del df

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[Tuple, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        key = self.keys[idx]
        X, y, mask = self.data[key]
        return (key,), X, y, mask
    
    def process(self, y: torch.Tensor) -> torch.Tensor:
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y
        return torch.tensor(y_np.reshape(-1, 51).T, dtype=torch.float32)
    
    def get_batch(self, key: Tuple[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        date = key[0]
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return self.data[(date,)]


# end of app/loader/datedata.py