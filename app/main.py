# app/main.py

if __name__ == "__main__":
    import gc
    import yaml

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from app.core.logger import setup_logger
    from app.utils.cli import parse_args
    logger = setup_logger("main")
    args = parse_args()

    # ========== Logger ==========
    logger.info("=" * 50)
    logger.info("0. Starting the application......")
    logger.info(f"Config file: {args.config}.")

    # ========== Config path ==========
    from app.utils.filepath import get_cfgs_path
    
    logger.info("=" * 50)
    logger.info("1. Resolving config path......")
    
    config_path = get_cfgs_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    logger.info(f"Loading config from: {config_path}.")

    # ========== Loading features ==========
    from app.utils.helper import load_features
    from app.utils.filepath import get_data_path
    
    logger.info("=" * 50)
    logger.info("2. Loading features......")

    feature_file = cfg["data"]["features"]
    feature_path = get_data_path(feature_file)
    features = load_features(feature_path)
    label = cfg["data"]["label"]

    logger.info(f"Loading features from: {feature_path}. Number of features: {len(features)}. Label: {label}.")

    # ========== Building losses/metric/models ==========
    import app.losses
    import app.metric
    import app.models
    from app.core.build import build_losses, build_metric, build_models
    
    logger.info("=" * 50)
    logger.info("3. Building losses / metric / models ......")
        
    Loss = build_losses(cfg["loss"])
    logger.info(f"Loss function: {Loss}.")
    
    # Metric = build_metric(cfg["metric"])
    # logger.info(f"Metric: {Metric}.")
    
    Model = build_models(cfg["model"], feature_dim=len(features))
    logger.info(f"Model:\n{Model}.")

    # ========== Loading data ==========
    import app.loader
    from app.core.build import build_loader
    
    logger.info("=" * 50)
    logger.info("4.1. Loading valid data......")
    
    ValidLoader = build_loader(
        cfg["data"]["validloader"],
        features=features,
        label=label
    )

    logger.info("4.2. Loading train data......")
    
    TrainLoader = build_loader(
        cfg["data"]["trainloader"],
        features=features,
        label=label
    )
    
    # ========== Training ==========
    from app.core.training import Trainer
    from app.utils.filepath import get_logs_path

    logger.info("=" * 50)
    logger.info("5. Training the model......")
    
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
    
    logger.info("Training config:")
    for k, v in cfg["train"].items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 50)
    
    # ModelName = trainer.debug(epochs=cfg["train"]["epochs"])
    ModelName = trainer.training(epochs=cfg["train"]["epochs"])

    # ========== Clearing ==========
    logger.info("=" * 50)
    logger.info("6. Clearing GPU memory......")
    
    del trainer, TrainLoader, ValidLoader
    gc.collect()
    torch.cuda.empty_cache()

    if Writer is not None:
        Writer.close()

    # ========== Evaluating ==========
    from app.core.evaluating import Evaluator
    from app.utils.cli import assemble
    from app.utils.filepath import get_sota_path

    logger.info("=" * 50)
    logger.info("7. Evaluating the model......")
    
    logger.info("*" * 50)
    logger.info("7.1. Building metric / models ......")
    
    # Loss = build_losses(cfg["loss"])
    # logger.info(f"Loss function: {Loss}.")

    # Metric = build_metric(cfg["metric"])
    # logger.info(f"Metric: {Metric}.")

    Model = torch.load(get_sota_path(ModelName))
    logger.info(f"Model:\n{Model}.")

    logger.info("*" * 50)
    logger.info("7.2. Loading test data......") 
    
    EvalLoader = build_loader(
        cfg["data"]["evalloader"],
        features=features,
        label=label
    )

    logger.info("*" * 50)
    logger.info("7.3. Evaluating......") 

    Device = torch.device(cfg["eval"]["device"])
    Writer = None
    if cfg["eval"].get("record", False):
        logdir = cfg["eval"].get("logdir", "tensorboard")
        Writer = SummaryWriter(log_dir=get_logs_path(logdir))
    
    Model = Model.to(Device)

    evaluator = Evaluator(
        model=Model,
        loss_fn=None,
        eval_loader=EvalLoader,
        device=Device,
        writer=Writer
    )

    logger.info("Evaluating config:")
    for k, v in cfg["eval"].items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 50)
    
    Result = evaluator.evaluating()
    ComboPath = assemble(Result, ModelName, by=cfg["data"]["evalloader"]["name"], mode="eval")

    # ========== Clearing ==========
    logger.info("=" * 50)
    logger.info("8. Clearing GPU memory......")

    for name, param in Model.named_parameters():
        if param.requires_grad:
            logger.info(f"参数名称: {name}, 参数形状: {param.shape}.")
            logger.info(f"最大值: {param.data.max().item():.4f}.")
            logger.info(f"最小值: {param.data.min().item():.4f}.")
            logger.info(f"均值: {param.data.mean().item():.4f}.")
            logger.info(f"标准差: {param.data.std().item():.4f}.")
        else:
            logger.info(f"参数名称: {name}, 参数形状: {param.shape}, 未训练.")    

    del evaluator, EvalLoader
    gc.collect()
    torch.cuda.empty_cache()

    if Writer is not None:
        Writer.close()
    
    # ========== Output ==========
    logger.info("=" * 50)
    logger.info("9. Output Combo path......")

    print(ComboPath)
    
    logger.info("=" * 50)
    logger.info("10. All done!")
    logger.info("=" * 50 + "\n")


# end of app/main.py