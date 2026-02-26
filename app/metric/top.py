# app/metric/top.py

import torch

from app.core.registry import register_metric


@register_metric("top")
@torch.no_grad()
def TopKMetric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor = None, k: int = 100) -> torch.Tensor:
    """
    Top-K Hit Rate: Count how many predicted top/bottom K actually hit real top/bottom K.
    Args:
        y_pred: shape (N,) or (N, 1), predicted returns (used for ranking)
        y_true: shape (N,) or (N, 1), true returns
        mask: shape (N,) or (N, 1), boolean mask indicating valid samples
        k: number of stocks to select from top and bottom
    Returns:
        scalar tensor representing hit rate (hits / k)
    """
    y_pred = y_pred.view(-1).detach()
    y_true = y_true.view(-1).detach()

    if mask is not None:
        mask = mask.view(-1).bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    n = y_pred.numel()
    if n < 2 * k:
        k = n // 2  # adjust k when sample size is insufficient

    if k <= 0 or n < 2:
        return torch.tensor(0.0, device=y_pred.device)

    _, pred_top_indices = torch.topk(y_pred, k)
    _, pred_bottom_indices = torch.topk(-y_pred, k)

    _, true_top_indices = torch.topk(y_true, k)
    _, true_bottom_indices = torch.topk(-y_true, k)

    hits_top = torch.isin(pred_top_indices, true_top_indices).sum()
    hits_bottom = torch.isin(pred_bottom_indices, true_bottom_indices).sum()

    hit_rate = (hits_top + hits_bottom).float() / (2 * k)

    return hit_rate

# end of app/metric/top.py
