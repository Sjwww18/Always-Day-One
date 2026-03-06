# app/core/build.py

import torch.nn as nn
from typing import Any, Callable, List

from app.core.registry import LOADER_REGISTRY, LOSSES_REGISTRY, METRIC_REGISTRY, MODELS_REGISTRY


def build_loader(cfg: dict, features: List[str], label: List[str]) -> Any:
    cls = LOADER_REGISTRY[cfg["name"]]
    params = cfg.get("params", {}).copy()
    params["features"] = features
    params["label"] = label
    return cls(**params)


def build_metric(cfg: dict) -> Callable:
    from functools import partial
    fn = METRIC_REGISTRY[cfg["name"]]
    params = cfg.get("params", {})
    if params:
        metric_fn = partial(fn, **params)
    else:
        metric_fn = fn
    metric_fn.name = cfg["name"]
    return metric_fn


def build_models(cfg: dict, **extra_kwargs) -> nn.Module:
    cls = MODELS_REGISTRY[cfg["name"]]
    params = cfg.get("params", {}).copy()
    params.update(extra_kwargs)
    return cls(**params)


def build_losses(cfg: dict) -> nn.Module:
    cls = LOSSES_REGISTRY[cfg["name"]]
    return cls(**cfg.get("params", {}))


# end of app/core/build.py