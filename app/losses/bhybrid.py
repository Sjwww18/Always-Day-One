# app/losses/bhybrid.py

import torch
import torch.nn as nn

from app.core.registry import register_losses
from app.losses.bmse import BatchMSELoss
from app.losses.bic import BatchICLoss


@register_losses("bhybrid")
class BatchHybridLoss(nn.Module):
    """
    Cross-batch Hybrid Loss = mse_weight * BatchMSELoss + ic_weight * BatchICLoss
    """
    def __init__(self, mse_weight: float=1.0, ic_weight: float=0.2):
        super().__init__()

        self.mse_weight = mse_weight
        self.ic_weight = ic_weight

        self.mse = BatchMSELoss()
        self.ic = BatchICLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        mse_loss = self.mse(y_pred, y_true, mask)
        ic_loss = self.ic(y_pred, y_true, mask)

        total_loss = self.mse_weight * mse_loss + self.ic_weight * ic_loss

        return total_loss


# end of app/losses/bhybrid.py