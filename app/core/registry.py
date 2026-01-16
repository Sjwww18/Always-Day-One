# src/core/registry.py

import torch.nn as nn
from typing import Type, Callable, Dict

LOSSES_REGISTRY: Dict[str, Callable] = {}
METRIC_REGISTRY: Dict[str, Callable] = {}
MODELS_REGISTRY: Dict[str, Callable] = {}


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


# end of src/core/registry.py