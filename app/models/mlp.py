# app/models/mlp.py

import torch
import torch.nn as nn

from core.registry import register_models


@register_models("mlp")
class MLPModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        # init
        nn.init.xavier_uniform_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)
        nn.init.xavier_uniform_(self.mlp[4].weight, gain=0.1)
        nn.init.zeros_(self.mlp[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# end of app/models/mlp.py