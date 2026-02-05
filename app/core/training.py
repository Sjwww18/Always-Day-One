# app/core/training.py

from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from app.core.logger import setup_logger
from app.loader.loaddata import LoadData
logger = setup_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: LoadData,
        valid_loader: LoadData,
        device: torch.device,
        writer: SummaryWriter=None,
        scheduler: torch.optim.lr_scheduler._LRScheduler=None,
        early_stop_cfg: dict=None
    ):
        self.Model = model
        self.Loss = loss_fn
        self.Optimizer = optimizer
        self.TrainLoader = train_loader
        self.ValidLoader = valid_loader
        self.Device = device
        self.Writer = writer
        self.Scheduler = scheduler
        self.EarlyStopCfg = early_stop_cfg or {}

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

    def savesota(self, val_loss) -> str:
        from app.utils.filepath import get_sota_path

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = getattr(self.Model, "__class__", None)
        model_name = model_name.__name__ if model_name else "Model"
        loss_name = getattr(self.Loss, "__class__", None)
        loss_name = loss_name.__name__ if loss_name else "Loss"

        filename = f"{model_name}_{loss_name}_{now}.pth"
        path = get_sota_path(filename)
        torch.save(self.Model, path)
        # torch.save(self.Model.state_dict(), path)
        logger.info(f"* 保存最优模型 -> {path} *.")

        self.best_val_loss = val_loss
        self.patience_counter = 0

        return filename

    def training(self, epochs: int) -> str:
        self.current_epoch = 0
        best_model_name = None
        for epoch in tqdm(range(epochs), desc="训练进度"):
            self.current_epoch = epoch
            logger.info(f"[Epoch {epoch+1}/{epochs}] 开始训练......")

            train_loss = self.trainone()
            valid_loss = self.validate()

            logger.info(
                f"第 {epoch+1} / {epochs} 轮训练完成 | "
                f"训练集平均损失: {train_loss:.6f} | 验证集平均损失: {valid_loss:.6f}."
            )

            # Scheduler step
            if self.Scheduler is not None:
                self.Scheduler.step()

            # Early Stop
            if valid_loss < self.best_val_loss:
                best_model_name = self.savesota(valid_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        
        return best_model_name

    def debug(self, epochs: int) -> str:
        for i in range(epochs):
            print(f"[Epoch {i}].")
        return f"Debug_{epochs}epochs.pth"


# end of app/core/training.py