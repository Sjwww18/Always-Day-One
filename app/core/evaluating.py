# app/core/evaluating.py

import numpy as np
from tqdm import tqdm
from typing import Any, List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

# from app.core.logger import setup_logger
# logger = setup_logger(__name__)


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        eval_loader: Any,
        device: torch.device,
        writer: SummaryWriter=None
    ):
        self.Model = model
        self.Loss = loss_fn
        self.EvalLoader = eval_loader
        self.Device = device
        self.Writer = writer
    
    def evaluating(self) -> List[Tuple[Any, np.ndarray]]:
        self.Model.eval()

        with torch.no_grad():
            Result = [None] * len(self.EvalLoader)
            for i, (key, X, y, mask) in enumerate(
                tqdm(self.EvalLoader, desc="评估日期进度")
            ):
                X = X.to(self.Device)

                ypre = self.Model(X)
                if isinstance(ypre, torch.Tensor):
                    ypre = ypre.cpu().numpy()
                ypre = self.EvalLoader.process(ypre)
                Result[i] = (key, ypre)

        return Result


# end of app/core/evaluating.py