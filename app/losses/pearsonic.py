# app/losses/pearsonic.py

import torch
import torch.nn as nn

from app.core.registry import register_losses


@register_losses("pearsonic")
class PearsonICLoss(nn.Module):
    def __init__(self, eps: float=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: shape (N,) or (N, 1)
            y_true: shape (N,) or (N, 1)
        Returns:
            scalar loss tensor (negative Pearson IC)
        """
        # flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # demean
        y_pred = y_pred - y_pred.mean()
        y_true = y_true - y_true.mean()

        # covariance
        cov = torch.mean(y_pred * y_true)

        # standard deviations
        std_pred = torch.sqrt(torch.mean(y_pred ** 2) + self.eps)
        std_true = torch.sqrt(torch.mean(y_true ** 2) + self.eps)

        # pearson correlation
        ic = cov / (std_pred * std_true)

        # loss: maximize IC <=> minimize -IC
        return -ic


# end of app/losses/pearsonic.py