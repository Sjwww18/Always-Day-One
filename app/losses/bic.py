# app/losses/bic.py

import torch
import torch.nn as nn

from app.core.registry import register_losses


@register_losses("bic")
class BatchICLoss(nn.Module):
    def __init__(self, eps: float=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Cross-sectional IC Loss.

        Args:
            y_pred: shape (N_stock, N_interval) or (N_stock, N_interval, 1)
            y_true: shape (N_stock, N_interval) or (N_stock, N_interval, 1)
            mask: shape (N_stock, N_interval) or (N_stock, N_interval, 1), 1=valid, 0=invalid

        Returns:
            scalar loss tensor (negative mean IC across intervals)
        """
        y_pred = y_pred.squeeze(-1)  # (N_stock, N_interval)
        y_true = y_true.squeeze(-1)  # (N_stock, N_interval)

        if mask is not None:
            mask = mask.squeeze(-1)  # (N_stock, N_interval)
            y_pred = y_pred * mask
            y_true = y_true * mask
            n_valid = mask.sum(dim=0)  # (N_interval,)
        else:
            n_valid = y_pred.shape[0]  # scalar

        pred_mean = y_pred.sum(dim=0) / (n_valid + self.eps)  # (N_interval,)
        true_mean = y_true.sum(dim=0) / (n_valid + self.eps)  # (N_interval,)

        pred_centered = y_pred - pred_mean  # (N_stock, N_interval)
        true_centered = y_true - true_mean  # (N_stock, N_interval)

        cov = (pred_centered * true_centered).sum(dim=0) / (n_valid + self.eps)  # (N_interval,)
        std_pred = torch.sqrt((pred_centered ** 2).sum(dim=0) / (n_valid + self.eps))  # (N_interval,)
        std_true = torch.sqrt((true_centered ** 2).sum(dim=0) / (n_valid + self.eps))  # (N_interval,)

        ic = cov / (std_pred * std_true)  # (N_interval,)
        return -ic.mean()


# end of app/losses/bic.py