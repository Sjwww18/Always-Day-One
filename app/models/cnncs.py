# app/models/cnncs.py

import torch
import torch.nn as nn

from app.models.cnn import ConvBlock
from app.models.cs import Attention, CrossSectionMLP
from app.core.registry import register_models


def get_cross_section_module(cs_type, dim, **kwargs):
    if cs_type == "attention":
        return Attention(dim=dim, **kwargs)
    elif cs_type == "mlp":
        return CrossSectionMLP(dim=dim, **kwargs)
    else:
        raise ValueError(f"Unknown cs_type: {cs_type}, choose from ['attention', 'mlp']")


@register_models("cnn1d_attention")
class CNNCS(nn.Module):
    """
    CNN + Cross-Section Module
    
    Args:
        feature_dim: input feature dimension (F)
        hidden_channels: CNN hidden channels, last one is output dim for cross-section
        cs_type: cross-section type, 'attention' or 'mlp'
        cs_kwargs: kwargs for cross-section module
    """
    def __init__(
        self,
        feature_dim: int=450,
        hidden_channels: list=[64],
        kernel_size: int=6,
        dropout: float=0.0,
        cs_type: str="attention",
        cs_kwargs: dict=None,
        init: str="xavier",
        padding_type: str="zero",
        dilation: list=None,
    ):
        super().__init__()
        
        cs_kwargs = cs_kwargs or {}
        
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
        
        cs_dim = hidden_channels[-1]
        self.cs = get_cross_section_module(cs_type, dim=cs_dim, **cs_kwargs)
        self.head = nn.Linear(cs_dim, 1)
        
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

    def forward(self, x):
        B, S, F, T = x.shape
        x = x.view(B * S, F, T)
        
        for block in self.blocks:
            x = block(x)
        
        x = x.view(B, S, -1, T)  # (B, S, C, T)
        x = x.transpose(2, 3)  # (B, S, T, C)
        
        x = self.cs(x)  # (B, S, T, C)
        x = self.head(x)  # (B, S, T, 1)
        
        return x


# end of app/models/cnncs.py