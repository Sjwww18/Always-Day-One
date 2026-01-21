# app/data/loaddata.py

import datetime
import pandas as pd
from typing import List, Tuple, Union, Iterable, Optional

import torch

class LoadData:
    """
    
    """
    def __init__(self, path: str, label: List[str], features: List[str], dffilter: Optional[str]=None, device: str="cpu"):
        self.label = label
        self.features = features
        self.device = device
        
        df = pd.read_parquet(path, engine="pyarrow")
        if dffilter is not None:
            df = df.query(dffilter)
        df = df.dropna(subset=label)
        df[features] = df[features].fillna(0.0)

        self.df = df
        self.days = [g for _, g in df.groupby("date", sort=False)]

    def __len__(self) -> int:
        return len(self.days)

    def __iter__(self) -> Iterable[torch.Tensor]:
        for g in self.days:
            X = torch.tensor(g[self.features].values, dtype=torch.float32, device=self.device)
            y = torch.tensor(g[self.label].values.reshape(-1, 1), dtype=torch.float32, device=self.device)
            yield X, y
    
    def get_date(self, date: Union[str, datetime.datetime]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch data for a specific trading date.

        Parameters:
        date: str or datetime.datetime. The trading date (e.g. "2020-01-02").

        Returns:
        X and y: torch.Tensor.
        """
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")
        date_df = self.df[self.df["date"] == date]

        X = torch.tensor(date_df[self.features].values, dtype=torch.float32, device=self.device)
        y = torch.tensor(date_df[self.label].values.reshape(-1, 1), dtype=torch.float32, device=self.device)
        
        return X, y


# end of app/data/loaddata.py