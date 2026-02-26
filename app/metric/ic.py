# app/metric/ic.py

import torch

from app.core.registry import register_metric


@register_metric("pearsonic")
@torch.no_grad()
def PearsonIcMetric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None, eps: float=1e-8) -> torch.Tensor:
    """
    Pearson Information Coefficient (IC) - correlation between predicted and true returns.
    Args:
        y_pred: shape (N,) or (N, 1), predicted returns
        y_true: shape (N,) or (N, 1), true returns
        mask: shape (N,) or (N, 1), boolean mask indicating valid samples
    Returns:
        scalar Pearson IC tensor
    """
    y_pred = y_pred.view(-1).detach()
    y_true = y_true.view(-1).detach()

    if mask is not None:
        mask = mask.view(-1).bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    if y_pred.numel() < 2:
        return torch.tensor(0.0, device=y_pred.device)

    y_pred = y_pred - y_pred.mean()
    y_true = y_true - y_true.mean()

    cov = torch.mean(y_pred * y_true)
    std_pred = torch.sqrt(torch.mean(y_pred ** 2) + eps)
    std_true = torch.sqrt(torch.mean(y_true ** 2) + eps)

    ic = cov / (std_pred * std_true)

    return ic


@register_metric("spearmanic")
@torch.no_grad()
def SpearmanIcMetric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor=None, eps: float=1e-8) -> torch.Tensor:
    """
    Spearman Information Coefficient (IC) - rank correlation between predicted and true returns.
    Args:
        y_pred: shape (N,) or (N, 1), predicted returns
        y_true: shape (N,) or (N, 1), true returns
        mask: shape (N,) or (N, 1), boolean mask indicating valid samples
    Returns:
        scalar Spearman IC tensor
    """
    y_pred = y_pred.view(-1).detach()
    y_true = y_true.view(-1).detach()

    if mask is not None:
        mask = mask.view(-1).bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    if y_pred.numel() < 2:
        return torch.tensor(0.0, device=y_pred.device)

    rank_pred = torch.argsort(torch.argsort(y_pred)).float() + 1.0  # double argsort to get rank, +1 for 1-based ranking
    rank_true = torch.argsort(torch.argsort(y_true)).float() + 1.0

    rank_pred = rank_pred - rank_pred.mean()
    rank_true = rank_true - rank_true.mean()

    cov = torch.mean(rank_pred * rank_true)
    std_pred = torch.sqrt(torch.mean(rank_pred ** 2) + eps)
    std_true = torch.sqrt(torch.mean(rank_true ** 2) + eps)

    ic = cov / (std_pred * std_true)

    return ic

# end of app/metric/ic.py
