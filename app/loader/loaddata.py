# app/loader/loaddata.py

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Union, Iterable, Optional


class LoadData:
    """
    This loader keeps all data in memory and yields full-batch per trading day.
    """
    def __init__(self, path: str, label: List[str], features: List[str], dffilter: Optional[str]=None):        
        cols = ["date", "stock", "interval"] + features + label
        df = pd.read_parquet(path, engine="pyarrow", columns=cols)
        
        df = df.dropna(subset=label)
        df[label] = df[label].astype("float32")
        df[features] = df[features].fillna(0.0).astype("float32")
        
        if dffilter is not None:
            df = df.query(dffilter)

        self.label = label
        self.features = features
        self.data = {d: g for d, g in df.groupby("date", sort=False)}
        self.days = list(self.data.keys())
        del df

    def __len__(self) -> int:
        return len(self.days)

    def __iter__(self) -> Iterable[Tuple[datetime, np.ndarray, np.ndarray]]:
        for d, g in self.data.items():
            X = g[self.features].to_numpy()
            y = g[self.label].to_numpy().reshape(-1, 1)
            yield d, X, y
    
    def get_date(self, date: Union[str, datetime]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch data for a specific trading date.

        Parameters:
        date: str or datetime.datetime. The trading date (e.g. "2020-01-02").

        Returns:
        X and y: torch.Tensor.
        """
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")
        date_df = self.data[date]

        X = date_df[self.features].to_numpy()
        y = date_df[self.label].to_numpy().reshape(-1, 1)
        
        return X, y


# end of app/loader/loaddata.py