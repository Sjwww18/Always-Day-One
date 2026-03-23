# app/main.py

if __name__ == "__main__":
    import gc
    import yaml

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from app.core.logger import setup_logger
    from app.utils.cli import parse_args, set_seed
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

    # ========== Set seed ==========
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}.")
    
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
    
    Metric = [build_metric({"name": m}) for m in cfg["metric"]]
    logger.info(f"Metric: {[fn.__name__ for fn in Metric]}.")
    
    Model = build_models(cfg["model"], feature_dim=len(features))
    logger.info(f"Model:\n{Model}.")

    # ========== Loading data ==========
    import app.loader
    from app.core.build import build_dataset, build_loader
    
    logger.info("=" * 50)
    logger.info("4.1. Loading valid data......")
    
    ValidDataset = build_dataset(
        cfg["data"]["validloader"],
        features=features,
        label=label
    )

    logger.info("4.2. Loading train data......")
    
    TrainDataset = build_dataset(
        cfg["data"]["trainloader"],
        features=features,
        label=label
    )
    
    logger.info("4.3. Building DataLoaders......")
    
    dataloader_cfg = cfg.get("dataloader", {})
    BatchSize = dataloader_cfg.get("batch_size", 1)
    Shuffle = dataloader_cfg.get("shuffle", False)
    NumWorkers = dataloader_cfg.get("num_workers", 0)
    PinMemory = dataloader_cfg.get("pin_memory", False)
    
    logger.info(f"DataLoader config: batch_size={BatchSize}, shuffle={Shuffle}, num_workers={NumWorkers}, pin_memory={PinMemory}")
    
    TrainLoader = build_loader(
        TrainDataset,
        batch_size=BatchSize,
        shuffle=Shuffle,
        num_workers=NumWorkers,
        pin_memory=PinMemory
    )
    
    ValidLoader = build_loader(
        ValidDataset,
        batch_size=BatchSize,
        shuffle=False,
        num_workers=NumWorkers,
        pin_memory=PinMemory
    )
    
    # ========== Training ==========
    from app.core.training import Trainer
    from app.utils.filepath import get_ckpt_path, get_logs_path

    logger.info("=" * 50)
    logger.info("5. Training the model......")
    
    Device = torch.device(cfg["train"]["device"])
    Writer = None
    if cfg["train"].get("record", False):
        logdir = cfg["train"].get("logdir", "tensorboard")
        Writer = SummaryWriter(log_dir=get_logs_path(logdir))
    
    if torch.cuda.device_count() > 1:
        Model = torch.nn.DataParallel(Model)
    Model = Model.to(Device)
    
    Lr = float(cfg["train"]["lr"])
    OptimizerCfg = cfg["train"].get("optimizer", {"name": "Adam", "params": {}})
    Optimizer = torch.optim.__dict__[OptimizerCfg["name"]](
        Model.parameters(), lr=Lr, **OptimizerCfg.get("params", {})
    )

    Scheduler = cfg["train"].get("scheduler", None)
    if Scheduler is not None:
        Scheduler = torch.optim.lr_scheduler.__dict__[Scheduler["name"]](
            Optimizer, **Scheduler["params"]
        )
    exp_name = args.config.replace(".yaml", "")
    EarlyStopCfg = cfg["train"].get("early_stop", {})
    CheckpointCfg = cfg["train"].get("checkpoint", {})

    trainer = Trainer(
        model=Model,
        loss_fn=Loss,
        metric_fns=Metric,
        optimizer=Optimizer,
        scheduler=Scheduler,
        train_loader=TrainLoader,
        valid_loader=ValidLoader,
        device=Device,
        writer=Writer,
        exp_name=exp_name,
        early_stop_cfg=EarlyStopCfg,
        checkpoint_cfg=CheckpointCfg
    )
    
    logger.info("Training config:")
    for k, v in cfg["train"].items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 50)
    
    total_epochs = cfg["train"]["epochs"]
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}.")
        start_epoch, _ = trainer.resume(get_ckpt_path(exp_name, args.resume))
        remaining_epochs = total_epochs - (start_epoch + 1)
        logger.info(f"Already trained {start_epoch + 1} epochs, remaining: {remaining_epochs}.")
        if remaining_epochs <= 0:
            logger.warning("Training already completed, skipping training.")
        else:
            ModelName = trainer.training(epochs=total_epochs)
    else:
        ModelName = trainer.training(epochs=total_epochs)

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
    from app.utils.ckpt import load_ckpt

    logger.info("=" * 50)
    logger.info("7. Evaluating the model......")
    
    logger.info("*" * 50)
    logger.info("7.1. Building metric / models ......")
    
    # Loss = build_losses(cfg["loss"])
    # logger.info(f"Loss function: {Loss}.")

    # Metric = build_metric(cfg["metric"])
    # logger.info(f"Metric: {Metric}.")

    Model = build_models(cfg["model"], feature_dim=len(features))
    logger.info(f"Model:\n{Model}.")

    logger.info("*" * 50)
    logger.info("7.2. Loading eval data......") 
    
    EvalDataset = build_dataset(
        cfg["data"]["evalloader"],
        features=features,
        label=label
    )
    
    logger.info("7.3. Building DataLoader......")
    
    # Set Eval Loader Config
    BatchSize = 1
    NumWorkers = 0
    PinMemory = False
    
    logger.info(f"DataLoader config: batch_size={BatchSize}, num_workers={NumWorkers}, pin_memory={PinMemory}")
    
    EvalLoader = build_loader(
        EvalDataset,
        batch_size=BatchSize,
        shuffle=False,
        num_workers=NumWorkers,
        pin_memory=PinMemory
    )

    logger.info("*" * 50)
    logger.info("7.3. Evaluating......") 

    Device = torch.device(cfg["eval"]["device"])
    Writer = None
    if cfg["eval"].get("record", False):
        logdir = cfg["eval"].get("logdir", "tensorboard")
        Writer = SummaryWriter(log_dir=get_logs_path(logdir))
    
    exp_name = args.config.replace(".yaml", "")
    ckpt_name = ModelName if ModelName else "best.ckpt"
    load_ckpt(get_ckpt_path(exp_name, ckpt_name), Model, device=Device)
    logger.info(f"Model loaded from: {ckpt_name}.")

    if torch.cuda.device_count() > 1:
        Model = torch.nn.DataParallel(Model)
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
    ComboPath = assemble(Result, ckpt_name, by=cfg["data"]["evalloader"]["name"], mode="eval")

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