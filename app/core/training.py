# app/core/training.py

from tqdm import tqdm
from datetime import datetime
from typing import Any, Callable, Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

from app.core.logger import setup_logger
from app.utils.filepath import get_ckpt_path
logger = setup_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        metric_fns: List[Callable],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader: Any,
        valid_loader: Any,
        device: torch.device,
        writer: SummaryWriter=None,
        exp_name: str="default",
        early_stop_cfg: dict=None,
        checkpoint_cfg: dict=None
    ):
        self.Model = model
        self.Loss = loss_fn
        self.Metric = metric_fns or []
        self.Optimizer = optimizer
        self.Scheduler = scheduler
        self.TrainLoader = train_loader
        self.ValidLoader = valid_loader
        self.Device = device
        self.Writer = writer
        self.ExpName = exp_name
        self.EarlyStopCfg = early_stop_cfg or {}
        self.CheckpointCfg = checkpoint_cfg or {}

        # -------- Early Stop Config --------
        self.monitors = self.EarlyStopCfg.get("monitor", ["val_loss"])
        self.patience = self.EarlyStopCfg.get("patience", 10)
        
        # monitor_modes: {"val_loss": "min", "ic": "max"}
        self.monitor_modes = self.EarlyStopCfg.get(
            "monitor_modes",
            {m: "min" if m == "val_loss" else "max" for m in self.monitors}
        )
        
        self.patience_counter = 0
        self.best_monitor_vals = {}
        for m in self.monitors:
            if self.monitor_modes.get(m, "max") == "min":
                self.best_monitor_vals[m] = float("inf")
            else:
                self.best_monitor_vals[m] = float("-inf")

    def trainone(self) -> float:
        self.Model.train()
        
        batch_nums = 0
        total_loss = 0.0
        for key, X, y, mask in tqdm(self.TrainLoader, desc="Train"):
            X = X.to(self.Device)
            y = y.to(self.Device) if y is not None else None
            mask = mask.to(self.Device) if mask is not None else None
            
            self.Optimizer.zero_grad()
            ypre = self.Model(X)
            
            loss = self.Loss(ypre, y, mask=mask)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[Epoch {self.current_epoch}] batch loss is NaN or Inf, skip this batch, key: {key}.")
                continue
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Model.parameters(), max_norm=1.0)
            self.Optimizer.step()

            batch_nums += 1
            total_loss += loss.item()
            # logger.debug(f"训练批次: {batch_nums}, loss: {loss.item():.4f}, key: {key}.")

        avg_loss = total_loss / max(batch_nums, 1)

        if self.Writer is not None:
            self.Writer.add_scalar("Loss/Train", avg_loss, self.current_epoch)
            self.Writer.add_scalar("Learning_Rate", self.Optimizer.param_groups[0]["lr"], self.current_epoch)
        logger.info(f"[Epoch {self.current_epoch+1}] Train Loss: {avg_loss:.6f}.")
        
        return avg_loss

    def validate(self) -> Dict[str, float]:
        self.Model.eval()

        batch_nums = 0
        total_loss = 0.0
        
        all_pred, all_y, all_mask = [], [], []
        
        with torch.no_grad():
            for key, X, y, mask in tqdm(self.ValidLoader, desc="Valid"):
                X = X.to(self.Device)
                y = y.to(self.Device) if y is not None else None
                mask = mask.to(self.Device) if mask is not None else None
                
                ypre = self.Model(X)
                loss = self.Loss(ypre, y, mask=mask)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[Epoch {self.current_epoch}] batch loss is NaN or Inf, skip this batch, key: {key}.")
                    continue
                
                batch_nums += 1
                total_loss += loss.item()
                # logger.debug(f"验证批次: {batch_nums}, loss: {loss.item():.4f}, key: {key}.")
                
                all_pred.append(ypre.cpu())
                all_y.append(y.cpu())
                if mask is not None:
                    all_mask.append(mask.cpu())

        avg_loss = total_loss / max(batch_nums, 1)
        
        metrics = {"val_loss": avg_loss}
        
        if self.Metric and all_pred:
            all_pred = torch.cat(all_pred, dim=0)
            all_y = torch.cat(all_y, dim=0)
            all_mask = torch.cat(all_mask, dim=0) if all_mask else None
            
            for metric_fn in self.Metric:
                name = getattr(metric_fn, "name", metric_fn.__name__)
                value = metric_fn(all_pred, all_y)
                metrics[name] = value.item()
        
        if self.Writer is not None:
            for k, v in metrics.items():
                if k == "val_loss":
                    self.Writer.add_scalar("Loss/Valid", v, self.current_epoch)
                else:
                    self.Writer.add_scalar(f"Metric/{k}", v, self.current_epoch)
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        logger.info(f"[Epoch {self.current_epoch+1}] Valid: {metric_str}.")
        
        return metrics
    
    def check_improvement(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        improved = {}
        for m in self.monitors:
            current = metrics.get(m)
            best = self.best_monitor_vals.get(m)
            if current is None:
                improved[m] = False
                continue
            mode = self.monitor_modes.get(m, "max")
            if mode == "min":
                improved[m] = current < best
            else:
                improved[m] = current > best
        return improved

    def update_early_stop(self, metrics: Dict[str, float]) -> bool:
        """
        OR improvement rule:
        """
        improved_dict = self.check_improvement(metrics)
        any_improved = any(improved_dict.values())
        if any_improved:
            for m, flag in improved_dict.items():
                if flag:
                    self.best_monitor_vals[m] = metrics[m]
            self.patience_counter = 0
            logger.info(
                "Improved: " +
                ", ".join([m for m, v in improved_dict.items() if v])
            )
            return False
        self.patience_counter += 1
        logger.info(
            f"No improvement. patience: "
            f"{self.patience_counter}/{self.patience}"
        )
        if self.patience_counter >= self.patience:
            logger.info("Early stopping triggered.")
            return True
        return False

    def training(self, epochs: int) -> str:
        from app.utils.ckpt import save_ckpt
        
        if not hasattr(self, "current_epoch"):
            self.current_epoch = 0

        best_ckpt_name = None

        for epoch in tqdm(range(self.current_epoch, epochs), desc="训练进度"):
            self.current_epoch = epoch
            logger.info(f"===== Epoch {epoch+1}/{epochs} =====")

            train_loss = self.trainone()
            valid_metrics = self.validate()

            logger.info(
                f"第 {epoch+1} / {epochs} 轮训练完成 | "
                f"训练集平均损失: {train_loss:.6f}."
            )

            if self.Scheduler is not None:
                self.Scheduler.step()
            
            # -------- Always Save Latest --------
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_ckpt(
                path=get_ckpt_path(self.ExpName, "latest.ckpt"),
                model=self.Model,
                optimizer=self.Optimizer,
                scheduler=self.Scheduler,
                epoch=self.current_epoch,
                best_metric=valid_metrics,
            )

            # -------- Early Stop --------
            should_stop = self.update_early_stop(valid_metrics)
            # -------- Save Best --------
            if any(valid_metrics.get(m) == self.best_monitor_vals[m]
                   for m in self.monitors):
                if self.CheckpointCfg.get("save_best", True):
                    best_ckpt_name = f"best_{self.current_epoch+1}.ckpt"
                    save_ckpt(
                        path=get_ckpt_path(self.ExpName, best_ckpt_name),
                        model=self.Model,
                        optimizer=self.Optimizer,
                        scheduler=self.Scheduler,
                        epoch=self.current_epoch,
                        best_metric=valid_metrics,
                    )
                    logger.info("* Saved best model *")
            
            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break
        
        return best_ckpt_name

    def resume(self, path: str):
        from app.utils.ckpt import load_ckpt
        
        start_epoch, best_metric = load_ckpt(
            path=path,
            model=self.Model,
            optimizer=self.Optimizer,
            scheduler=self.Scheduler,
            device=self.Device,
        )
        
        self.current_epoch = start_epoch + 1
        self.patience_counter = 0
        if isinstance(best_metric, dict):
            for m in self.monitors:
                if m in best_metric:
                    self.best_monitor_vals[m] = best_metric[m]
        logger.info(
            f"Resumed from epoch {start_epoch}."
            f"best_monitor_vals: {self.best_monitor_vals}."
        )
        
        return start_epoch, best_metric

    def debug(self, epochs: int) -> str:
        for i in range(epochs):
            print(f"[Epoch {i}].")
        return f"Debug_{epochs}epochs.ckpt"


# end of app/core/training.py