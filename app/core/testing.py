# app/core/testing.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from app.core.logger import setup_logger
from app.utils.filepath import get_data_path
from app.loader.loaddata import LoadData
logger = setup_logger(__name__)


class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        test_loader: LoadData,
        device: torch.device,
        writer: SummaryWriter=None,
        EqtyPath: str="EqtyData.pkl"
    ):
        self.Model = model
        self.Loss = loss_fn
        self.TestLoader = test_loader
        self.Device = device
        self.Writer = writer
        self.EqtyData = pd.read_pickle(get_data_path(EqtyPath))

    def testing(self) -> List[Tuple[datetime, np.ndarray]]:
        self.Model.eval()

        with torch.no_grad():
            Result = []
            for d, X in tqdm(self.TestLoader.data.items(), desc="推理日期进度"):
                X = X[self.TestLoader.features].to_numpy()
                X = torch.from_numpy(X).float().to(self.Device)

                ypre = self.Model(X)
                if isinstance(ypre, torch.Tensor):
                    ypre = ypre.cpu().numpy()
                ypre = ypre.reshape(51, -1)  # 51 interval × 5171 stock
                Result.append((d, ypre))

        return Result
        
    def postprocess(self, Result: List[Tuple[datetime, np.ndarray]]) -> pd.DataFrame:
        DATES_ls = [d for d, y in Result]
        COMBO_np = np.stack([y for _, y in Result], axis=0)
        COMBO_np = COMBO_np.reshape(-1, COMBO_np.shape[-1])

        DATES_np = np.repeat(DATES_ls, 51)
        INDEX_np = np.tile(np.arange(51), len(DATES_ls))
        idx = pd.MultiIndex.from_arrays([DATES_np, INDEX_np], names=["date", "interval"])

        COMBO = pd.DataFrame(COMBO_np, index=idx, columns=self.EqtyData)
        
        return COMBO


# end of app/core/testing.py