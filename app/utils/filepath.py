# app/utils/filepath.py

import os


def get_proj_root():
    """Get the absolute path of the project root directory"""
    # Traverse up from the current file (file_handling.py) to the project root
    # Directory structure: src/project_name/file_handling.py → root is project_name/
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )


def get_data_path(subdir=""):
    """Get the path to the data directory (creates directory automatically if it doesn't exist)"""
    data_dir = os.path.join(get_proj_root(), "data", subdir)
    os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists
    return data_dir


def get_imgs_path():
    """Get the path to the images directory"""
    imgs_dir = os.path.join(get_proj_root(), "imgs")
    os.makedirs(imgs_dir, exist_ok=True)  # Ensure directory exists
    return imgs_dir


# Similarly, you can define get_tables_path, get_logs_path, etc.
def get_tabs_path():
    """Get the path to the tables directory"""
    tabs_dir = os.path.join(get_proj_root(), "tabs")
    os.makedirs(tabs_dir, exist_ok=True)  # Ensure directory exists
    return tabs_dir


def get_logs_path():
    """Get the path to the logs directory"""
    logs_dir = os.path.join(get_proj_root(), "logs")
    os.makedirs(logs_dir, exist_ok=True)  # Ensure directory exists
    return logs_dir


# end of app/utils/filepath.py