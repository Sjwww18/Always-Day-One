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
        self.data = {}
        for d, g in df.groupby(["date"], sort=False):
            X = g[self.features]

            # ===== nan handling =====
            if self.fillna == "zero":
                X = X.fillna(0.0)
            elif self.fillna == "mean":
                X = X.fillna(X.mean(axis=0))
            
            X = X.to_numpy(dtype="float32")

            # ===== normalization =====
            if self.normalize == "zscore":
                X = zscore(X)

            # ===== label =====
            if label:
                y = g[label].to_numpy(dtype="float32").reshape(-1, 1)
                mask = (~np.isnan(y)).astype("float32")
            else:
                y = None
                mask = None

            self.data[(d,)] = (X, y, mask)
        
        self.keys = list(self.data.keys())        
        del df

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterable[Tuple[Tuple[datetime], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        for key in self.keys:
            X, y, mask = self.data[key]
            yield key, X, y, mask
    
    def process(self, y: np.ndarray) -> np.ndarray:
        return y.reshape(-1, 51).T  # 51 interval × 5171 stock
    
    def get_batch(self, key: Tuple[str]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        date = key[0]
        # if isinstance(date, datetime):
        #     date = date.strftime("%Y-%m-%d")
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        return self.data[(date,)]


# end of app/loader/datedata.py