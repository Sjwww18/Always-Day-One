# app/core/training.py

from tqdm import tqdm 
from typing import Any

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
        self.Optimizer = optimizer
        self.Scheduler = scheduler
        self.TrainLoader = train_loader
        self.ValidLoader = valid_loader
        self.Device = device
        self.Writer = writer
        self.ExpName = exp_name
        self.EarlyStopCfg = early_stop_cfg or {}
        self.CheckpointCfg = checkpoint_cfg or {}

        self.patience = self.EarlyStopCfg.get("patience", 10)
        self.monitor = self.EarlyStopCfg.get("monitor", "val_loss")
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def trainone(self) -> float:
        self.Model.train()
        
        batch_nums = 0
        total_loss = 0.0
        for key, X, y, mask in tqdm(self.TrainLoader, desc="日期进度"):
            X = torch.from_numpy(X).to(self.Device)
            y = torch.from_numpy(y).to(self.Device)
            if mask is not None:
                mask = torch.from_numpy(mask).to(self.Device)
            
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
            logger.debug(f"训练批次: {batch_nums}, loss: {loss.item():.4f}, key: {key}.")

        avg_loss = total_loss / max(batch_nums, 1)

        # TensorBoard
        if self.Writer is not None and avg_loss == avg_loss:
            self.Writer.add_scalar("Loss/Train", avg_loss, self.current_epoch)
            self.Writer.add_scalar("Learning_Rate", self.Optimizer.param_groups[0]["lr"], self.current_epoch)
        logger.info(f"[Epoch {self.current_epoch}] 训练集平均损失: {avg_loss:.4f}.")
        
        return avg_loss

    def validate(self) -> float:
        self.Model.eval()

        batch_nums = 0
        total_loss = 0.0
        with torch.no_grad():
            for key, X, y, mask in tqdm(self.ValidLoader, desc="日期进度"):
                X = torch.from_numpy(X).to(self.Device)
                y = torch.from_numpy(y).to(self.Device)
                if mask is not None:
                    mask = torch.from_numpy(mask).to(self.Device)
                
                ypre = self.Model(X)
                loss = self.Loss(ypre, y, mask=mask)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[Epoch {self.current_epoch}] batch loss is NaN or Inf, skip this batch, key: {key}.")
                    continue
                
                batch_nums += 1
                total_loss += loss.item()
                logger.debug(f"验证批次: {batch_nums}, loss: {loss.item():.4f}, key: {key}.")

        avg_loss = total_loss / max(batch_nums, 1)
        
        # TensorBoard
        if self.Writer is not None and avg_loss == avg_loss:
            self.Writer.add_scalar("Loss/Valid", avg_loss, self.current_epoch)
        logger.info(f"[Epoch {self.current_epoch}] 验证集平均损失: {avg_loss:.4f}.")
        
        return avg_loss

    def training(self, epochs: int) -> str:
        from app.utils.ckpt import save_ckpt

        self.current_epoch = 0
        best_ckpt_name = None

        for epoch in tqdm(range(epochs), desc="训练进度"):
            self.current_epoch = epoch
            logger.info(f"[Epoch {epoch+1}/{epochs}] 开始训练......")

            train_loss = self.trainone()
            valid_loss = self.validate()

            logger.info(
                f"第 {epoch+1} / {epochs} 轮训练完成 | "
                f"训练集平均损失: {train_loss:.6f} | 验证集平均损失: {valid_loss:.6f}."
            )

            if self.Scheduler is not None:
                self.Scheduler.step()

            save_latest = self.CheckpointCfg.get("save_latest", True)
            if save_latest:
                save_ckpt(
                    path=get_ckpt_path(self.ExpName, "latest.ckpt"),
                    model=self.Model,
                    optimizer=self.Optimizer,
                    scheduler=self.Scheduler,
                    epoch=self.current_epoch,
                    best_metric=valid_loss,
                )

            if valid_loss < self.best_val_loss:
                self.best_val_loss = valid_loss
                self.patience_counter = 0
                
                save_best = self.CheckpointCfg.get("save_best", True)
                if save_best:
                    best_ckpt_name = "best.ckpt"
                    save_ckpt(
                        path=get_ckpt_path(self.ExpName, best_ckpt_name),
                        model=self.Model,
                        optimizer=self.Optimizer,
                        scheduler=self.Scheduler,
                        epoch=self.current_epoch,
                        best_metric=valid_loss,
                    )
                    logger.info(f"* 保存最佳模型 (best.ckpt) *.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        
        return best_ckpt_name

    def resume(self, path):
        from app.utils.ckpt import load_ckpt
        
        start_epoch, best_metric = load_ckpt(
            path=path,
            model=self.Model,
            optimizer=self.Optimizer,
            scheduler=self.Scheduler,
            device=self.Device,
        )
        
        self.current_epoch = start_epoch + 1
        self.best_val_loss = best_metric if best_metric is not None else float("inf")
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_metric}.")
        
        return start_epoch, best_metric

    def debug(self, epochs: int) -> str:
        for i in range(epochs):
            print(f"[Epoch {i}].")
        return f"Debug_{epochs}epochs.ckpt"


# end of app/core/training.py