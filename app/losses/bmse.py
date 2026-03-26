# app/losses/bmse.py

import torch
import torch.nn as nn

from app.core.registry import register_losses


@register_losses("bmse")
class BatchMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Cross-sectional MSE Loss.

        Args:
            y_pred: shape (N_stock, N_interval) or (N_stock, N_interval, 1)
            y_true: shape (N_stock, N_interval) or (N_stock, N_interval, 1)
            mask: shape (N_stock, N_interval) or (N_stock, N_interval, 1), 1=valid, 0=invalid

        Returns:
            scalar loss tensor (mean MSE across intervals)
        """
        B, S, T, _ = y_pred.shape
        y_pred = y_pred.squeeze(-1).reshape(B * S, T)  # (N_stock, N_interval)
        y_true = y_true.squeeze(-1).reshape(B * S, T)  # (N_stock, N_interval)

        se = (y_pred - y_true) ** 2

        if mask is not None:
            mask = mask.squeeze(-1).reshape(B * S, T)  # (N_stock, N_interval)
            se = se * mask
            mse_per_interval = se.sum(dim=0) / (mask.sum(dim=0) + 1e-8)
        else:
            mse_per_interval = se.mean(dim=0)

        return mse_per_interval.mean()


# end of app/losses/bmse.py