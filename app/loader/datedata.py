# app/loader/datedata.py

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from app.utils.helper import zscore
from app.core.registry import register_loader
from app.utils.filepath import get_data_path


@register_loader("date")
class DateLoader:
    """
    Date-level cross-sectional loader.

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
        
        # ===== label / feature handling =====
        self.label = label
        self.features = features
        self.fillna = fillna
        self.normalize = normalize

        if dffilter is not None:
            df = df.query(dffilter)

        # ===== core data structure =====
        # key: (date) -> DataFrame
        self.data = {
            (d): g
            for (d), g in df.groupby(["date"], sort=False)
        }
        self.keys = list(self.data.keys())        
        del df

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterable[Tuple[Tuple[datetime], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        for key in self.keys:
            g = self.data[key]
            X = g[self.features]
            
            # ===== nan handling =====
            if self.fillna == "zero":
                X = X.fillna(0.0).to_numpy(dtype="float32")
            elif self.fillna == "mean":
                X = X.fillna(X.mean(axis=0)).to_numpy(dtype="float32")
            else:
                X = X.to_numpy(dtype="float32")

            # ===== normalization =====
            if self.normalize == "zscore":
                X = zscore(X)

            # ===== label =====            
            if self.label:
                y = g[self.label].to_numpy(dtype="float32").reshape(-1, 1)
                mask = (~np.isnan(y)).astype("float32")
            else:
                y = None
                mask = None
            
            yield key, X, y, mask
    
    def process(self, y: np.ndarray) -> np.ndarray:
        return y.reshape(-1, 51).T  # 51 interval × 5171 stock
    
    def get_batch(self, key: Tuple[datetime]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        date = key
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")
        
        g = self.data[(date)]
        X = g[self.features].to_numpy(dtype="float32")
        
        if self.label:
            y = g[self.label].to_numpy(dtype="float32").reshape(-1, 1)
            mask = (~np.isnan(y)).astype("float32")
        else:
            y = None
            mask = None
        
        return X, y, mask


# end of app/loader/datedata.py