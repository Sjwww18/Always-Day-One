import torch

from configs.default import config
from core.registry import MODEL_REGISTRY
from core.trainer import Trainer
from data.dataloader import build_dataloader


device = torch.device(config["device"])

test_loader = build_dataloader(config["data"], split="test")

model_cfg = config["model"]
model = MODEL_REGISTRY[model_cfg["name"]](**model_cfg["params"])
model.load_state_dict(torch.load(
    "logs/checkpoints/model.pt",
    map_location=device
))

trainer = Trainer(
    model=model,
    loss_fn=None,
    optimizer=None,
    device=device,
    cfg=config,
)

preds = trainer.predict(test_loader)
