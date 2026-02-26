# app/debug/metrics.py

import torch
import numpy as np

from app.metric.ic import PearsonIcMetric, SpearmanIcMetric
from app.metric.mse import MSEMetric
from app.metric.pnl import PnlMetric
from app.metric.top import TopKMetric
from app.metric.dd import DdMetric
from app.metric.sp import SpMetric
from app.metric.icir import IcirMetric


def run_case(name: str, method, *args, **kwargs) -> None:
    print(f"\n=== {name} ===")
    
    ic_result = method(*args, **kwargs)
    print(f"IC (list): {ic_result}")
    
    ic_tensor = torch.tensor(ic_result) if not isinstance(ic_result, torch.Tensor) else ic_result
    print(f"IC (tensor): {ic_tensor}")
    
    ic_array = ic_tensor.cpu().numpy() if isinstance(ic_tensor, torch.Tensor) else np.array(ic_result)
    print(f"IC (array): {ic_array}")


if __name__ == "__main__":
    ic_values_list = [0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.18, 0.22]
    ic_values_tensor = torch.tensor(ic_values_list)
    ic_values_array = np.array(ic_values_list)
    
    print("\n========== IcirMetric (list) ==========")
    run_case("ICIR (no window)", IcirMetric, ic_values_list)
    run_case("ICIR (window=3)", IcirMetric, ic_values_list, window_size=3)
    run_case("ICIR (window=5)", IcirMetric, ic_values_list, window_size=5)

    print("\n========== IcirMetric (tensor) ==========")
    run_case("ICIR (no window)", IcirMetric, ic_values_tensor)
    run_case("ICIR (window=3)", IcirMetric, ic_values_tensor, window_size=3)

    print("\n========== IcirMetric (array) ==========")
    run_case("ICIR (no window)", IcirMetric, ic_values_array)
    run_case("ICIR (window=3)", IcirMetric, ic_values_array, window_size=3)

    returns_list = [0.01, -0.02, 0.03, 0.015, -0.01, 0.02, 0.025, -0.005]
    returns_tensor = torch.tensor(returns_list)
    returns_array = np.array(returns_list)
    
    print("\n========== SpMetric (list) ==========")
    run_case("Sharpe (no window)", SpMetric, returns_list)
    run_case("Sharpe (window=3)", SpMetric, returns_list, window_size=3)
    run_case("Sharpe (window=5)", SpMetric, returns_list, window_size=5)

    print("\n========== SpMetric (tensor) ==========")
    run_case("Sharpe (no window)", SpMetric, returns_tensor)
    run_case("Sharpe (window=3)", SpMetric, returns_tensor, window_size=3)

    print("\n========== SpMetric (array) ==========")
    run_case("Sharpe (no window)", SpMetric, returns_array)
    run_case("Sharpe (window=3)", SpMetric, returns_array, window_size=3)

    print("\n========== DdMetric (list) ==========")
    run_case("Drawdown (no window)", DdMetric, returns_list)
    run_case("Drawdown (window=3)", DdMetric, returns_list, window_size=3)
    run_case("Drawdown (window=5)", DdMetric, returns_list, window_size=5)

    print("\n========== DdMetric (tensor) ==========")
    run_case("Drawdown (no window)", DdMetric, returns_tensor)
    run_case("Drawdown (window=3)", DdMetric, returns_tensor, window_size=3)

    print("\n========== DdMetric (array) ==========")
    run_case("Drawdown (no window)", DdMetric, returns_array)
    run_case("Drawdown (window=3)", DdMetric, returns_array, window_size=3)

    print("\n========== PearsonIcMetric ==========")
    y_pred = torch.tensor([2.0, 3.0, 7.0])
    y_true = torch.tensor([1.0, 2.0, 3.0])
    ic = PearsonIcMetric(y_pred, y_true)
    print(f"Pearson IC: {ic.item():.4f}")

    print("\n========== SpearmanIcMetric ==========")
    ic = SpearmanIcMetric(y_pred, y_true)
    print(f"Spearman IC: {ic.item():.4f}")

    print("\n========== PnlMetric ==========")
    y_pred = torch.tensor([2.0, 3.0, 7.0, 1.0, 5.0, 6.0, 4.0, 8.0])
    y_true = torch.tensor([1.0, 2.0, 3.0, 0.5, 5.0, 6.0, 4.0, 7.0])
    pnl = PnlMetric(y_pred, y_true, long_k=2, short_k=2)
    print(f"PnL: {pnl.item():.4f}")

    print("\n========== TopKMetric ==========")
    topk = TopKMetric(y_pred, y_true, k=3)
    print(f"TopK: {topk.item():.4f}")

    print("\n========== MSEMetric ==========")
    mse = MSEMetric(y_pred, y_true)
    print(f"MSE: {mse.item():.4f}")


# end of app/debug/metrics.py