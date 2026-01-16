# app/debug/losses.py

import torch
from app.losses.mse import MSELoss

if __name__ == "__main__":
    criterion = MSELoss()
    y_pred = torch.tensor([[2.0], [3.0], [7.0]])
    y_true = torch.tensor([[1.0], [2.0], [3.0]])
    loss = criterion(y_pred, y_true)
    print(f"MSE Loss: {loss.item():.4f}.")


# end of app/debug/losses.py