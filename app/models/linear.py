# app/models/linear.py

import torch
import torch.nn as nn

from app.core.registry import register_models


@register_models("linear")
class LINEAR(nn.Module):
    """
    Linear model for cross-sectional ranking (IC baseline).

    ### Input:
    x: [N_stock, F_feature]

    ### Output:
    score: [N_stock, 1]
    """
    def __init__(
        self,
        feature_dim: int,
        init: str="xavier",
    ):
        super().__init__()

        self.linear = nn.Linear(feature_dim, 1)
        self._init_weights(init)

    def _init_weights(self, init: str):
        if init == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="linear")
        elif init == "normal":
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init type: {init}.")

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# end of app/models/linear.py