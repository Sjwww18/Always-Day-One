# app/metric/pnl.py

import torch

from app.core.registry import register_metric


@register_metric("pnl")
@torch.no_grad()
def PnlMetric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None, long_k: int=100, short_k: int=100) -> torch.Tensor:
    """
    PnL (Profit and Loss) metric: Compute returns for long and short positions.
    Args:
        y_pred: shape (N,) or (N, 1), predicted returns (used for ranking)
        y_true: shape (N,) or (N, 1), true returns
        mask: shape (N,) or (N, 1), boolean mask indicating valid samples
        long_k: number of stocks to long (top predicted)
        short_k: number of stocks to short (bottom predicted)
    Returns:
        scalar tensor representing PnL (long_return - short_return)
    """
    y_pred = y_pred.view(-1).detach()
    y_true = y_true.view(-1).detach()

    if mask is not None:
        mask = mask.view(-1).bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    n = y_pred.numel()
    if n < long_k + short_k:
        long_k = n // 2
        short_k = n - long_k

    if long_k <= 0 or short_k <= 0 or n < 2:
        return torch.tensor(0.0, device=y_pred.device)

    _, long_indices = torch.topk(y_pred, long_k)
    _, short_indices = torch.topk(-y_pred, short_k)

    long_return = y_true[long_indices].mean()
    short_return = y_true[short_indices].mean()

    pnl = long_return - short_return

    return pnl


# end of app/metric/pnl.py