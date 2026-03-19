# app/models/cnn.py

from typing import List

import torch
import torch.nn as nn

from app.core.registry import register_models


@register_models("cnn2d")
class CNN2D(nn.Module):
    """
    CNN2D model for time-series (interval-level) feature extraction.

    ### Input:
    x: Tensor of shape [N_stock, F_feature, M_interval]

    ### Output:
    score: Tensor of shape [N_stock * M_interval, 1]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_channels: List[int] = [32, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        init: str = "xavier",
    ):
        super().__init__()

        assert len(hidden_channels) > 0, "hidden_channels must be non-empty"

        layers = []

        in_ch = feature_dim

        # normalize over feature dim
        self.norm = nn.LayerNorm(feature_dim)

        for ch in hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # 保持长度不变
                )
            )
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = ch

        self.conv = nn.Sequential(*layers)

        # projection to score
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, 1, kernel_size=1),
            nn.Tanh()
        )

        self._init_weights(init)

    def _init_weights(self, init: str):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
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
            x: [N_stock, F_feature, M_interval]

        Returns:
            score: [N_stock * M_interval, 1]
        """

        # LayerNorm expects last dim = feature
        # so先 transpose
        x = x.transpose(1, 2)           # [N, M, F]
        x = self.norm(x)
        x = x.transpose(1, 2)           # [N, F, M]

        x = self.conv(x)                # [N, C, M]
        x = self.head(x)                # [N, 1, M]

        x = x.transpose(1, 2)           # [N, M, 1]
        x = x.reshape(-1, 1)            # flatten

        return x


# end of app/models/cnn.py