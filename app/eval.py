# app/eval.py

if __name__ == "__main__":
    import gc
    import yaml
    
    import torch
    from torch.utils.tensorboard import SummaryWriter
    
    from app.core.logger import setup_logger
    from app.utils.cli import parse_args, set_seed
    logger = setup_logger("eval")
    args = parse_args()

    if args.model is None:
        logger.error("Model path is required for eval/backtest.")
    
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

    logger.info(f"Loading features from: {feature_path}. Number of features: {len(features)}.")

    # ========== Building losses/metric/models ==========
    import app.losses
    import app.metric
    import app.models
    from app.core.build import build_losses, build_metric, build_models
    from app.utils.filepath import get_ckpt_path
    
    logger.info("=" * 50)
    logger.info("3. Building metric / models ......")
    
    # Loss = build_losses(cfg["loss"])
    # logger.info(f"Loss function: {Loss}.")

    # Metric = build_metric(cfg["metric"])
    # logger.info(f"Metric: {Metric}.")

    Model = build_models(cfg["model"], feature_dim=len(features))
    logger.info(f"Model: {Model}.")
    
    # ========== Loading data ==========
    import app.loader
    from app.core.build import build_dataset, build_loader
    
    logger.info("=" * 50)
    logger.info("4. Loading eval data......")
    
    EvalDataset = build_dataset(
        cfg["data"]["evalloader"],
        features=features,
        label=label
    )
    
    logger.info("4.1. Building DataLoader......")
    
    # Set Eval Loader Config
    BatchSize = 1
    NumWorkers = 0
    PinMemory = False
    
    logger.info(f"DataLoader config: batch_size={BatchSize}, num_workers={NumWorkers}, pin_memory={PinMemory} (eval fixed)")
    
    EvalLoader = build_loader(
        EvalDataset,
        batch_size=BatchSize,
        shuffle=False,
        num_workers=NumWorkers,
        pin_memory=PinMemory
    )
    
    # ========== Evaluating ==========
    from app.core.evaluating import Evaluator
    from app.utils.cli import assemble
    from app.utils.ckpt import load_ckpt
    from app.utils.filepath import get_ckpt_path, get_logs_path

    logger.info("=" * 50)
    logger.info("5. Evaluating the model......")
    
    Device = torch.device(cfg["eval"]["device"])
    Writer = None
    if cfg["eval"].get("record", False):
        logdir = cfg["eval"].get("logdir", "tensorboard")
        Writer = SummaryWriter(log_dir=get_logs_path(logdir))
    
    exp_name = args.config.replace(".yaml", "")
    ckpt_name = args.model if args.model else "best.ckpt"
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
    logger.info("6. Clearing GPU memory......")
    
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
    logger.info("7. Output Combo path......")

    print(ComboPath)
    
    logger.info("=" * 50)
    logger.info("8. All done!")
    logger.info("=" * 50 + "\n")


# end of app/eval.py