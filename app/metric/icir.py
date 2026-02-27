# app/metric/icir.py

import torch
import numpy as np


@torch.no_grad()
def IcirMetric(ic_values: torch.Tensor, window_size: int=None, eps: float=1e-8) -> torch.Tensor:
    """
    Compute IC Information Ratio (ICIR) = Mean(IC) / Std(IC).
    Args:
        ic_values: list or tensor of IC values from multiple periods
        window_size: if None, compute single ICIR for all values; 
                     if set, compute sliding window ICIRs
        eps: small value to avoid division by zero
    Returns:
        If window_size is None: scalar ICIR tensor
        If window_size is set: tensor of ICIR values
    """
    if isinstance(ic_values, (list, np.ndarray)):
        ic_values = torch.tensor(ic_values)
    
    if ic_values.numel() == 0:
        return torch.tensor(0.0, device=ic_values.device)
    
    if window_size is None:
        mean_ic = ic_values.mean()
        std_ic = ic_values.std(unbiased=False)
        if std_ic < eps:
            return torch.tensor(0.0, device=ic_values.device)
        return mean_ic / std_ic
    
    windows = ic_values.unfold(0, window_size, 1)
    means = windows.mean(dim=1)
    stds = windows.std(dim=1, unbiased=False)
    
    results = torch.where(stds < eps, torch.zeros_like(stds), means / (stds + eps))
    
    return results


# end of app/metric/icir.py