# app/metric/mse.py

import torch

from app.core.registry import register_metric


@register_metric("mse")
@torch.no_grad()
def MSEMetric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """
    Mean Squared Error metric.
    Args:
        y_pred: shape (N,) or (N, 1)
        y_true: shape (N,) or (N, 1)
        mask: shape (N,) or (N, 1), boolean mask indicating valid samples
    Returns:
        scalar MSE tensor
    """
    y_pred = y_pred.view(-1).detach()
    y_true = y_true.view(-1).detach()

    if mask is not None:
        mask = mask.view(-1).bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    if y_pred.numel() < 2:
        return torch.tensor(0.0, device=y_pred.device)

    return torch.mean((y_pred - y_true) ** 2)


# end of app/metric/mse.py