# app/core/training.py

from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from core.logger import setup_logger
from core.registry import LOSSES_REGISTRY, MODELS_REGISTRY

logger = setup_logger(__name__)


class Trainer:
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        loss_name: str,
        loss_params: Dict[str, Any],
        model_name: str,
        model_params: Dict[str, Any],
        optimizer_class: type,
        optimizer_params: Dict[str, Any],
    ):
        self.loader = loader
        self.device = device

        # ===== losses =====
        if loss_name not in LOSSES_REGISTRY:
            raise KeyError(f"Loss '{loss_name}' not registered.")

        loss_cls = LOSSES_REGISTRY[loss_name]
        self.loss = loss_cls(**loss_params)
        
        # ===== models =====
        if model_name not in MODELS_REGISTRY:
            raise KeyError(f"Model '{model_name}' not registered.")

        model_cls = MODELS_REGISTRY[model_name]
        self.model = model_cls(**model_params).to(device)

        # ===== optimizer =====
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_params
        )

    def training(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(self.loader):
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)
            loss = self.loss(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                logger.info(
                    f"[Epoch {epoch} | Step {step}] loss={loss.item():.6f}"
                )
        
        avg_loss = total_loss / len(self.loader)
        logger.info(f"[Epoch {epoch}] avg_loss={avg_loss:.6f}.")
        
        return avg_loss


# end of app/core/training.py