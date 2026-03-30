# app/models/cs.py

import torch
import torch.nn as nn

# from app.core.registry import register_models


class CrossSectionMLP(nn.Module):
    """
    Simple Cross-Section MLP mixing
    For each time step t, mix S stocks with MLP: (S, C) -> (S, C)
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, S, T, C = x.shape
        x = x.permute(0, 2, 1, 3)  # (B, T, S, C)
        x = x.reshape(B * T, S, C)

        x = x + self.mlp(self.norm(x))

        x = x.reshape(B, T, S, C)
        x = x.permute(0, 2, 1, 3)
        
        return x


# @register_models("attention")
class Attention(nn.Module):
    """
    Cross-Section Attention
    For each time step t, apply self-attention across S stocks
    """
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, S, T, C = x.shape  # (B, S, T, C)

        x = x.permute(0, 2, 1, 3)  # (B, T, S, C)
        x = x.reshape(B * T, S, C)

        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x = x.reshape(B, T, S, C)
        x = x.permute(0, 2, 1, 3)  # (B, S, T, C)

        return x


# end of app/models/cs.py