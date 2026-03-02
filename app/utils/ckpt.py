# app/utils/ckpt.py

import os
import torch


def save_ckpt(
    path: str,
    model: torch.nn.Module,
    epoch: int=None,
    scaler: torch.cuda.amp.GradScaler=None,
    optimizer: torch.optim.Optimizer=None,
    scheduler: torch.optim.lr_scheduler._LRScheduler=None,
    best_metric: float=None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    ckpt = {
        "model_state_dict": model_state,
        "epoch": epoch,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
    }

    torch.save(ckpt, path)

    return None


def load_ckpt(
    path: str,
    model: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler=None,
    optimizer: torch.optim.Optimizer=None,
    scheduler: torch.optim.lr_scheduler._LRScheduler=None,
    device="cpu",
):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", None)

    return epoch, best_metric


# end of app/utils/ckpt.py