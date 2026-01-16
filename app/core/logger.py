# app/core/logger.py

import os
import logging
from logging.handlers import RotatingFileHandler

from app.utils.filepath import get_logs_path

LOG_DIR = get_logs_path()


def setup_logger(name=__name__, level=logging.INFO):
    """
    Sets up a logger with the specified name and level.
    Logs are written to a file in the LOG_DIR directory and also output to the console.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # log file handler
    log_file = os.path.join(LOG_DIR, f"{name}.log")
    fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# end of app/core/logger.py