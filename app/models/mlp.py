# app/models/mlp.py

import torch
import torch.nn as nn

from app.core.registry import register_models


@register_models("mlp")
class MLPModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int=32, init: str="xavier"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        # initialize weights based on init
        if init == "xavier":
            nn.init.xavier_uniform_(self.mlp[1].weight)
            nn.init.zeros_(self.mlp[1].bias)
            nn.init.xavier_uniform_(self.mlp[4].weight, gain=0.1)
            nn.init.zeros_(self.mlp[4].bias)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.mlp[1].weight, nonlinearity="gelu")
            nn.init.zeros_(self.mlp[1].bias)
            nn.init.kaiming_uniform_(self.mlp[4].weight, nonlinearity="linear")
            nn.init.zeros_(self.mlp[4].bias)
        elif init == "normal":
            nn.init.normal_(self.mlp[1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mlp[1].bias)
            nn.init.normal_(self.mlp[4].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mlp[4].bias)
        else:
            raise ValueError(f"Unknown init type: {init}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# end of app/models/mlp.py