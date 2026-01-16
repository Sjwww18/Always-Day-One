# app/debug/logger.py

from app.utils.filepath import (
    get_proj_root, get_imgs_path, get_tabs_path, get_logs_path)
from app.core.logger import setup_logger

logger = setup_logger(__name__)

logger.info(f"Proj Root Path: {get_proj_root()}.")
logger.debug(f"Imgs Path: {get_imgs_path()}.")
logger.debug(f"Tabs Path: {get_tabs_path()}.")
logger.debug(f"Logs Path: {get_logs_path()}.")


# end of app/debug/logger.py