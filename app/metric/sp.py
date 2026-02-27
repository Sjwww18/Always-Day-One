# app/metric/sp.py

import torch
import numpy as np


@torch.no_grad()
def SpMetric(returns: torch.Tensor, window_size: int=None, eps: float=1e-8):
    """
    Sharpe Ratio = Mean(returns) / Std(returns).
    Args:
        returns: list or tensor of return values
        window_size: if None, compute single Sharpe; if set, compute rolling Sharpe
        eps: small value to avoid division by zero
    Returns:
        If window_size is None: scalar Sharpe ratio
        If window_size is set: tensor of Sharpe ratios
    """
    if isinstance(returns, (list, np.ndarray)):
        returns = torch.tensor(returns)
    
    if returns.numel() == 0:
        return torch.tensor(0.0, device=returns.device)
    
    if window_size is None:
        mean_return = returns.mean()
        std_return = returns.std(unbiased=False)
        if std_return < eps:
            return torch.tensor(0.0, device=returns.device)
        return mean_return / std_return
    
    windows = returns.unfold(0, window_size, 1)
    means = windows.mean(dim=1)
    stds = windows.std(dim=1, unbiased=False)
    
    results = torch.where(stds < eps, torch.zeros_like(stds), means / (stds + eps))
    
    return results


# end of app/metric/sp.py