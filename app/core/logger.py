# app/core/logger.py

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from app.utils.filepath import get_logs_path


def setup_logger(name=__name__) -> logging.Logger:
    """
    Configure and return a module-specific logger.

    - INFO+ -> module info log
    - ERROR+ -> module error log
    - DEBUG+ -> console

    Args:
        name (str): usually __name__ of the caller module

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    # 防止重复添加 handler（非常重要）
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)

    # ---------- formatter ----------
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------- path ----------
    log_dir = get_logs_path()
    os.makedirs(log_dir, exist_ok=True)

    module_name = name.split(".")[-1]
    today = datetime.now().strftime("%Y-%m-%d")

    # ---------- INFO file handler ----------
    info_log = os.path.join(log_dir, f"{today}_{module_name}_info.log")
    info_handler = RotatingFileHandler(
        info_log,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    # ---------- ERROR file handler ----------
    error_log = os.path.join(log_dir, f"{today}_{module_name}_errors.log")
    error_handler = RotatingFileHandler(
        error_log,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # ---------- Console handler ----------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # ---------- add handlers ----------
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    # 标记，防止重复配置
    logger._configured = True

    return logger


# end of app/core/logger.py