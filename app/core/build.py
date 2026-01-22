# app/core/build.py
# app/core/build.py

import torch.nn as nn
from typing import Any, Callable

from core.registry import LOSSES_REGISTRY, METRIC_REGISTRY, MODELS_REGISTRY


def build_models(cfg: Any) -> nn.Module:
    cls = MODELS_REGISTRY[cfg.name]
    return cls(**cfg.params)


def build_metric(cfg: Any) -> Callable:
    fn = METRIC_REGISTRY[cfg.name]
    return fn


def build_losses(cfg: Any) -> nn.Module:
    cls = LOSSES_REGISTRY[cfg.name]
    return cls(**cfg.params)


# end of app/core/build.py