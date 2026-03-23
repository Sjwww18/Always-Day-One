# app/losses/mse.py

import torch
import torch.nn as nn

from app.core.registry import register_losses


@register_losses("mse")
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            y_pred: shape (N,) or (N, 1)
            y_true: shape (N,) or (N, 1)
            mask: shape (N,) or (N, 1), boolean mask indicating valid samples
        Returns:
            scalar loss tensor
        """
        diff = y_pred - y_true  # (B, T, 1)

        if mask is not None:
            mask = mask.bool()
            diff = diff[mask]

        if diff.numel() < 2:
            return diff.sum() * 0.0

        return (diff ** 2).mean()


# end of app/losses/mse.py