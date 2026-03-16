# app/loader/intervaldata.py

import gc
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from app.utils.helper import zscore
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

        # 先获取 group 索引（用于后续按 group 标准化）
        group_indices = df.groupby(["date", "interval"], sort=False).indices
        self.keys = list(group_indices.keys())
        self.row_indices = [group_indices[k] for k in self.keys]
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}

        # 转换为 numpy 数组
        X = df[self.features].values.astype(np.float32)
        y = df[self.label].values.astype(np.float32) if self.label else None

        # 按 group 进行 fillna 和 normalize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for row_ids in self.row_indices:
                if self.fillna == "zero":
                    X[row_ids] = np.nan_to_num(X[row_ids], nan=0.0)
                elif self.fillna == "mean":
                    group_mean = np.nanmean(X[row_ids], axis=0, keepdims=True)
                    nan_mask = np.isnan(X[row_ids])
                    X[row_ids] = np.where(nan_mask, group_mean, X[row_ids])
                    X[row_ids] = np.nan_to_num(X[row_ids], nan=0.0)

                if self.normalize == "zscore":
                    X[row_ids] = zscore(X[row_ids], axis=0)

        self.all_X = X
        self.all_y = y
        
        del df
        gc.collect()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[Tuple[datetime, int], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        key = self.keys[idx]
        row_ids = self.row_indices[idx]
        
        # 按行索引取出当前 (date, interval) 的截面数据（fancy indexing 会产生拷贝）
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

    def get_batch(self, key: Tuple[str, int]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        date, interval = key
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        idx = self.key_to_idx[(date, interval)]
        row_ids = self.row_indices[idx]
        
        X = torch.from_numpy(self.all_X[row_ids])
        y = torch.from_numpy(self.all_y[row_ids]) if self.all_y is not None else None
        mask = (~torch.isnan(y)).float() if y is not None else None
        
        return X, y, mask


# end of app/loader/intervaldata.py