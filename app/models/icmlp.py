# app/models/icmlp.py

from typing import List

import torch
import torch.nn as nn

from app.core.registry import register_models


@register_models("icmlp")
class ICMLP(nn.Module):
    """
    IC-style MLP for cross-sectional ranking.

    ### Input:
    x: Tensor of shape [N_stock, F_feature].
    (features are assumed to be cross-sectionally normalized in loader)

    ### Output:
    score: Tensor of shape [N_stock, 1].
    (ranking signal, no physical meaning)
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],
        dropout: float=0.0,
        init: str="xavier"
    ):
        super().__init__()

        assert len(hidden_dims) > 0, "hidden_dims must be a non-empty list"

        layers = []
        in_dim = feature_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        # final projection to score
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights(init)

    def _init_weights(self, init: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                elif init == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                else:
                    raise ValueError(f"Unknown init type: {init}.")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N_stock, F_feature]

        Returns:
            score: [N_stock, 1]
        """
        return self.mlp(x)


# end of app/models/icmlp.py