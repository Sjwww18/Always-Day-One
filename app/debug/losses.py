# app/debug/losses.py

import torch
import torch.nn as nn

from app.losses.mse import MSELoss
from app.losses.pearsonic import PearsonICLoss


def run_case(name: str, method: nn.Module, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    print(f"\n=== {name} ===")
    y_pred = y_pred.clone().requires_grad_(True)
    y_true = y_true.clone()

    criterion = method()
    loss = criterion(y_pred, y_true)
    print(f"loss = {loss.item():.4f}.")

    loss.backward()
    print(f"grad(y_pred) = {y_pred.grad.view(-1)}.")

    return None


if __name__ == "__main__":
    # -------------------------
    # Case 1: 常数预测（std_pred ≈ 0）
    # -------------------------
    y_pred = torch.tensor([[1.0], [1.0], [1.0]])
    y_true = torch.tensor([[0.2], [-0.1], [0.3]])
    run_case("Constant prediction", PearsonICLoss, y_pred, y_true)

    # -------------------------
    # Case 2: 完全正相关
    # -------------------------
    y_true = torch.tensor([[1.0], [2.0], [3.0]])
    y_pred = y_true.clone()
    run_case("Perfect positive correlation", PearsonICLoss, y_pred, y_true)

    # -------------------------
    # Case 3: 完全负相关
    # -------------------------
    y_pred = -y_true
    run_case("Perfect negative correlation", PearsonICLoss, y_pred, y_true)

    # -------------------------
    # Case 4: 一般情况
    # -------------------------
    y_pred = torch.tensor([[2.0], [3.0], [7.0]])
    y_true = torch.tensor([[1.0], [2.0], [3.0]])
    run_case("Normal case", PearsonICLoss, y_pred, y_true)

    # -------------------------
    # MSE Loss
    # -------------------------
    y_pred = torch.tensor([[2.0], [3.0], [7.0]])
    y_true = torch.tensor([[1.0], [2.0], [3.0]])
    run_case("MSE Loss", MSELoss, y_pred, y_true)


# end of app/debug/losses.py