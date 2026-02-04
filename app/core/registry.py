# app/core/registry.py

import torch.nn as nn
from typing import Any, Type, Callable, Dict

LOADER_REGISTRY: Dict[str, Callable] = {}
LOSSES_REGISTRY: Dict[str, Callable] = {}
METRIC_REGISTRY: Dict[str, Callable] = {}
MODELS_REGISTRY: Dict[str, Callable] = {}


def register_loader(name: str) -> Callable[[Type[Any]], Type[Any]]:
    def wrapper(cls):
        if name in LOADER_REGISTRY:
            raise KeyError(f"Loader '{name}' already registered.")
        LOADER_REGISTRY[name] = cls
        return cls
    return wrapper


def register_losses(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def wrapper(cls):
        if name in LOSSES_REGISTRY:
            raise KeyError(f"Losses '{name}' already registered.")
        LOSSES_REGISTRY[name] = cls
        return cls
    return wrapper


def register_metric(name: str) -> Callable[[Callable], Callable]:
    def wrapper(fn):
        if name in METRIC_REGISTRY:
            raise KeyError(f"Metric '{name}' already registered.")
        METRIC_REGISTRY[name] = fn
        return fn
    return wrapper


def register_models(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def wrapper(cls):
        if name in MODELS_REGISTRY:
            raise KeyError(f"Models '{name}' already registered.")
        MODELS_REGISTRY[name] = cls
        return cls
    return wrapper


# end of app/core/registry.py