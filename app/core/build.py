# app/core/build.py

import torch.nn as nn
from typing import Any, Callable

from app.core.registry import LOSSES_REGISTRY, METRIC_REGISTRY, MODELS_REGISTRY


def build_models(cfg: dict, **extra_kwargs) -> nn.Module:
    cls = MODELS_REGISTRY[cfg["name"]]
    params = cfg.get("params", {})
    params.update(extra_kwargs)
    return cls(**params)


def build_metric(cfg: dict) -> Callable:
    fn = METRIC_REGISTRY[cfg["name"]]
    return fn


def build_losses(cfg: dict) -> nn.Module:
    cls = LOSSES_REGISTRY[cfg["name"]]
    return cls(**cfg.get("params", {}))


# end of app/core/build.py