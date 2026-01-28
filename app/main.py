# app/main.py

import yaml
import pickle
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from app.utils.filepath import (
    get_cfgs_path, get_data_path, get_logs_path, get_sota_path, get_test_path
    )
from app.core.logger import setup_logger
logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Config file name under app/configs/"
    )
    return parser.parse_args()


def save_config(cfg: dict, path: str) -> None:
    """Save config to file."""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    """# Fight !!!"""
    args = parse_args()

    # ========== Logger ==========
    logger.info("=" * 50)
    logger.info("Starting the application......")
    logger.info("=" * 50)
    logger.info(f"Config file: {args.config}.")

    # --------------------------------------------------
    # Resolve config path
    # --------------------------------------------------

    config_path = get_cfgs_path(args.config)
    logger.info(f"Loading config from: {config_path}.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # --------------------------------------------------
    # Load feature list
    # --------------------------------------------------
    feature_file = cfg["data"]["features"]
    feature_path = get_data_path(feature_file)
    logger.info(f"Loading features from: {feature_path}.")
    
    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    if isinstance(features, list):
        features = features
    elif hasattr(features, "to_list"):
        features = features.to_list()
    else:
        features = list(features)
    logger.info(f"Number of features: {len(features)}.")

    # --------------------------------------------------
    # Build model / loss
    # --------------------------------------------------
    import app.losses
    import app.models
    from app.core.build import build_losses, build_models

    Loss = build_losses(cfg["loss"])
    logger.info(f"Loss function: {Loss}.")
    
    Model = build_models(cfg["model"], feature_dim=len(features))
    logger.info(f"Model: {Model}.")

    # --------------------------------------------------
    # Load validation data first
    # --------------------------------------------------
    from app.loader.loaddata import LoadData
    
    logger.info("Loading validation data......")
    valid_cfg = cfg["data"]["validdata"]
    ValidLoader = LoadData(
        path=get_data_path(valid_cfg["file"]),
        label=cfg["data"]["label"],
        features=features,
        dffilter=valid_cfg["filter"]
    )

    logger.info("Loading training data......")
    train_cfg = cfg["data"]["traindata"]
    TrainLoader = LoadData(
        path=get_data_path(train_cfg["file"]),
        label=cfg["data"]["label"],
        features=features,
        dffilter=train_cfg["filter"]
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    from app.core.training import Trainer

    logger.info("Starting training loop......")
    Device = torch.device(cfg["train"]["device"])
    Writer = None
    if cfg["train"].get("record", False):
        logdir = cfg["train"].get("logdir", "tensorboard")
        Writer = SummaryWriter(log_dir=get_logs_path(logdir))
    
    Model = Model.to(Device)
    Lr = float(cfg["train"]["lr"])
    Optimizer = torch.optim.Adam(Model.parameters(), lr=Lr)

    Scheduler = cfg["train"].get("scheduler", None)
    if Scheduler is not None:
        Scheduler = torch.optim.lr_scheduler.__dict__[Scheduler["name"]](
            Optimizer, **Scheduler["params"]
        )
    EarlyStopCfg = cfg["train"].get("early_stop", {})

    trainer = Trainer(
        model=Model,
        loss_fn=Loss,
        optimizer=Optimizer,
        train_loader=TrainLoader,
        valid_loader=ValidLoader,
        device=Device,
        writer=Writer,
        scheduler=Scheduler,
        early_stop_cfg=EarlyStopCfg
    )
    
    logger.info("=" * 50)
    logger.info("Training config:")
    for k, v in cfg["train"].items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 50)
    
    # trainer.debug(epochs=cfg["train"]["epochs"])
    ModelName = trainer.training(epochs=cfg["train"]["epochs"])
    ModelPath = get_sota_path(ModelName)
    cfg["result"]["model"] = ModelName
    del ValidLoader, TrainLoader
    
    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    from app.core.testing import Tester

    logger.info("Evaluating the model......")
    logger.info("Loading testing data......") 
    test_cfg = cfg["data"]["testdata"]
    
    TestLoader = LoadData(
        path=get_data_path(test_cfg["file"]),
        label=[],
        features=features,
        dffilter=test_cfg["filter"]
    )
    
    logger.info(f"Loading sota model: {ModelPath}......")
    Model = torch.load(ModelPath).to(Device)
    
    logger.info("Testing the model......")
    tester = Tester(
        model=Model,
        loss_fn=None,
        test_loader=TestLoader,
        device=Device,
        writer=None,
        EqtyPath=cfg["data"]["eqtydata"]
    )

    COMBO = tester.postprocess(tester.testing())
    
    ComboName = ModelName.replace(".pth", ".pkl")
    ComboPath = get_test_path(ComboName)
    COMBO.to_pickle(ComboPath)
    
    cfg["result"]["combo"] = ComboName
    save_config(cfg, config_path)
    del TestLoader

    if Writer is not None:
        Writer.close()
    logger.info("Training finished.")
    # --------------------------------------------------
    # Exiting
    # --------------------------------------------------
    return None

if __name__ == "__main__":
    main()


# end of app/main.py