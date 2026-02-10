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
        # flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # apply mask if provided
        if mask is not None:
            mask = mask.view(-1).bool()
            y_pred = y_pred[mask]
            y_true = y_true[mask]
        
        if y_pred.numel() < 2:
            return y_pred.sum() * 0.0

        return self.loss(y_pred, y_true)


# end of app/losses/mse.py