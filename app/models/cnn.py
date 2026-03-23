# app/models/cnn.py

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.registry import register_models


class ConvBlock(nn.Module):
    """
    Causal Conv Block with Residual
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        padding_type: str
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding_type = padding_type

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.res = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = self.res(x)

        pad_left = (self.kernel_size - 1) * self.dilation
        if self.padding_type == "replicate":
            x = F.pad(x, (pad_left, 0), mode="replicate")
        else:
            x = F.pad(x, (pad_left, 0), mode="constant", value=0.0)

        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)

        return x + res


@register_models("cnn2d")
class CNN2D(nn.Module):
    """
    x: (B, T, F)
    
    score: (B, T, 1)
    """
    def __init__(
        self,
        feature_dim: int=450,
        hidden_channels: List[int]=[64],
        kernel_size: int=6,
        dropout: float=0.0,
        init: str="kaiming",
        padding_type: str="zero",
        dilation: List[int]=None
    ):
        super().__init__()

        assert len(hidden_channels) > 0
        assert init in ["kaiming", "xavier", "normal"]

        self.blocks = nn.ModuleList()

        in_ch = feature_dim
        for i, ch in enumerate(hidden_channels):
            d = dilation[i] if dilation else 1
            self.blocks.append(
                ConvBlock(
                    in_ch,
                    ch,
                    kernel_size,
                    d,
                    dropout,
                    padding_type
                )
            )
            in_ch = ch

        self.head = nn.Conv1d(in_ch, 1, kernel_size=1)

        self._init_weights(init)

    def _init_weights(self, init: str):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if init == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif init == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        """
        x = x.transpose(1, 2)  # (B, F, T)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)  # (B, 1, T)
        x = x.transpose(1, 2)  # (B, T, 1)

        return x


# end of app/models/cnn.py