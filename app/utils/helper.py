# app/utils/helper.py


import pickle
import numpy as np


# ===== zscore =====
def zscore(X: np.ndarray, axis: int=0, eps: float=1e-8) -> np.ndarray:
    """
    Cross-sectional zscore (per batch).
    """
    mu = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mu) / (std + eps)


# ===== load features =====
def load_features(path: str) -> list:
    with open(path, "rb") as f:
        features = pickle.load(f)
    if isinstance(features, list):
        features = features
    elif hasattr(features, "to_list"):
        features = features.to_list()
    else:
        features = list(features)
    return features


# end of app/utils/helper.py