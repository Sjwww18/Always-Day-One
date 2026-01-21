import torch

from app.configs.default import config
from app.core.registry import LOSSES_REGISTRY, MODELS_REGISTRY
from app.core.training import Trainer
from app.data.loaddata import LoadData
print("Training script is currently disabled.")

# device
DEVICE = torch.device(config["device"])

# # data
train_loader = LoadData(config["data"], split="train")

# # model
# model_cfg = config["model"]
# model = MODEL_REGISTRY[model_cfg["name"]](**model_cfg["params"])

# # loss
# loss_fn = LOSSES_REGISTRY[config["loss"]["name"]]()

# # optimizer
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=config["optim"]["lr"]
# )

# # trainer
# trainer = Trainer(
#     model=model,
#     loss_fn=loss_fn,
#     optimizer=optimizer,
#     device=device,
#     cfg=config,
# )

# # train
# for epoch in range(config.get("epochs", 1)):
#     trainer.train_one_epoch(train_loader, epoch)
