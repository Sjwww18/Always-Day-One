# app/losses/mse.py

import torch
import torch.nn as nn

from app.core.registry import register_losses


@register_losses("mse")
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: shape (N,) or (N, 1)
            y_true: shape (N,) or (N, 1)
        Returns:
            scalar loss tensor
        """
        return self.loss(y_pred, y_true)


# end of app/losses/mse.py