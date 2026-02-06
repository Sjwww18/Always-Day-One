# app/utils/filepath.py

import os


def get_proj_root():
    """Get the absolute path of the project root directory"""
    # Traverse up from the current file (file_handling.py) to the project root
    # Directory structure: src/project_name/file_handling.py → root is project_name/
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )


def get_back_path(filename=""):
    """Get the path to the backtest directory (creates directory automatically if it doesn't exist)"""
    back_dir = os.path.join(get_proj_root(), "back")
    os.makedirs(back_dir, exist_ok=True)  # Ensure directory exists
    return os.path.join(back_dir, "data", filename)


def get_cfgs_path(filename=""):
    """Get the path to the configs directory (creates directory automatically if it doesn't exist)"""
    cfgs_dir = os.path.join(get_proj_root(), "cfgs")
    os.makedirs(cfgs_dir, exist_ok=True)  # Ensure directory exists
    return os.path.join(cfgs_dir, filename)


def get_data_path(filename=""):
    """Get the path to the data directory (creates directory automatically if it doesn't exist)"""
    data_dir = os.path.join(get_proj_root(), "data")
    os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists
    return os.path.join(data_dir, filename)


def get_imgs_path(subdir=""):
    """Get the path to the images directory"""
    imgs_dir = os.path.join(get_proj_root(), "imgs", subdir)
    os.makedirs(imgs_dir, exist_ok=True)  # Ensure directory exists
    return imgs_dir


def get_logs_path(subdir=""):
    """Get the path to the logs directory"""
    logs_dir = os.path.join(get_proj_root(), "logs", subdir)
    os.makedirs(logs_dir, exist_ok=True)  # Ensure directory exists
    return logs_dir


def get_sota_path(filename=""):
    """Get the path to the sota directory"""
    sota_dir = os.path.join(get_proj_root(), "sota")
    os.makedirs(sota_dir, exist_ok=True)  # Ensure directory exists
    return os.path.join(sota_dir, filename)


def get_tabs_path(subdir=""):
    """Get the path to the tables directory"""
    tabs_dir = os.path.join(get_proj_root(), "tabs", subdir)
    os.makedirs(tabs_dir, exist_ok=True)  # Ensure directory exists
    return tabs_dir


def get_test_path(filename=""):
    """Get the path to the test directory"""
    test_dir = os.path.join(get_proj_root(), "test")
    os.makedirs(test_dir, exist_ok=True)  # Ensure directory exists
    if filename.endswith(".pth"):
        filename = filename.replace(".pth", ".pkl")
    return os.path.join(test_dir, filename)


# end of app/utils/filepath.py