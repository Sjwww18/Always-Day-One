# app/metric/dd.py

import torch
import numpy as np


@torch.no_grad()
def DdMetric(returns: torch.Tensor, window_size: int=None) -> torch.Tensor:
    """
    Maximum Drawdown (MDD): Maximum observed loss from a peak to a trough.
    Note: Uses cumprod for Simple Returns. For Log Returns, use cumsum.
    Args:
        returns: list or tensor of return values
        window_size: if None, compute single MDD; if set, compute rolling MDD
    Returns:
        If window_size is None: scalar MDD tensor (positive value)
        If window_size is set: tensor of MDD values (positive values)
    """
    if isinstance(returns, (list, np.ndarray)):
        returns = torch.tensor(returns)
    
    if returns.numel() < 2:
        return torch.tensor(0.0, device=returns.device)
    
    if window_size is None:
        wealth = torch.cumprod(1 + returns, dim=0)
        running_max = torch.cummax(wealth, dim=0)[0]
        drawdown = (wealth - running_max) / running_max
        max_drawdown = torch.min(drawdown)
        return -max_drawdown
    
    windows = returns.unfold(0, window_size, 1)
    wealth = torch.cumprod(1 + windows, dim=1)
    running_max = torch.cummax(wealth, dim=1)[0]
    drawdown = (wealth - running_max) / running_max
    max_drawdown = torch.min(drawdown, dim=1)[0]
    
    return -max_drawdown


# end of app/metric/dd.py