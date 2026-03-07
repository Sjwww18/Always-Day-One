# app/core/build.py

import torch.nn as nn
from typing import Any, Callable, List

import torch
from torch.utils.data import DataLoader

from app.core.registry import LOADER_REGISTRY, LOSSES_REGISTRY, METRIC_REGISTRY, MODELS_REGISTRY


def build_dataset(cfg: dict, features: List[str], label: List[str]) -> Any:
    cls = LOADER_REGISTRY[cfg["name"]]
    params = cfg.get("params", {}).copy()
    params["features"] = features
    params["label"] = label
    return cls(**params)


def build_loader(dataset: Any, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    def collate_fn(batch):
        keys, Xs, ys, masks = zip(*batch)
        X_batch = torch.stack(Xs)
        if ys[0] is not None:
            y_batch = torch.stack(ys)
        else:
            y_batch = None
        if masks[0] is not None:
            mask_batch = torch.stack(masks)
        else:
            mask_batch = None
        return keys, X_batch, y_batch, mask_batch
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    if hasattr(dataset, 'process'):
        data_loader.process = dataset.process
    
    return data_loader


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