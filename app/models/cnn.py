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
        hidden_channels: List[int]=[64],
        kernel_size: int=3,
        dropout: float=0.1,
        init: str="xavier"
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

# app/models/dev.py

from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.registry import register_models


@register_models("cnn2d")
class CNN2D(nn.Module):
    """
    CNN1D Causal Model for time-series feature extraction.
    
    ### Input:
    x: Tensor of shape [Batch_size, M_interval, F_feature] 
       (即你说的 (Batch, 51, 450))
    
    ### Output:
    score: Tensor of shape [Batch_size * M_interval, 1]
    """
    def __init__(
        self,
        feature_dim: int = 450,
        hidden_channels: List[int] = [64],
        kernel_size: int = 6,
        dropout: float = 0.1,
        init: str = "kaiming",  # 配合 GELU，默认改成 kaiming
        padding_type: Literal["zero", "replicate"] = "zero" # 填充方式
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.padding_type = padding_type

        assert len(hidden_channels) > 0, "hidden_channels must be non-empty"
        assert init in ["kaiming", "xavier", "normal"], "Invalid init type"

        layers = []
        
        # 输入通道是 feature_dim (450)
        in_ch = feature_dim

        # 1. LayerNorm: 对 Feature 维度做归一化
        #    因为输入是 (B, T, F)，Norm 最后一维即可，无需 transpose
        self.norm = nn.LayerNorm(feature_dim)

        # 2. 堆叠因果卷积层
        for ch in hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=0, # 这里不做自动 padding，我们手动在 forward 里做因果填充
                )
            )
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = ch

        self.conv = nn.Sequential(*layers)

        # 3. 输出 Head
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, 1, kernel_size=1),
            nn.Tanh() # 如果你是做分类，记得把这里改成 Sigmoid 并去掉 Tanh
        )

        self._init_weights(init)

    def _init_weights(self, init: str):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                if init == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init == "kaiming":
                    # nonlinearity='relu' 对 GELU 也是一个很好的近似
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch_size, M_interval, F_feature] -> (B, 51, 450)
        Returns:
            score: [Batch_size * M_interval, 1]
        """
        B, T, F = x.shape
        
        # 1. 归一化
        x = self.norm(x) # (B, T, F)
        
        # 2. 调整维度为 Conv1d 要求的 (B, F, T)
        #    Conv1d expects (Batch, Channels, Length)
        x = x.transpose(1, 2) # (B, F, T)
        
        # 3. 因果填充 (Causal Padding)
        #    只在左边填充 (kernel_size - 1) 个单位
        pad_left = self.kernel_size - 1
        pad_right = 0
        
        if self.padding_type == "replicate":
            # 复制边界：即你说的“将第0行拷贝上去”
            # F.pad 格式是 (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...)
            # 我们要在最后一维 (Time) 的左边填充
            x = F.pad(x, (pad_left, pad_right), mode='replicate')
        else:
            # 默认零填充
            x = F.pad(x, (pad_left, pad_right), mode='constant', value=0.0)
        
        # 4. 卷积
        x = self.conv(x) # (B, C, T)，注意这里的 T 还是原来的长度，因为我们手动切过了
        
        # 5. 预测 Head
        x = self.head(x) # (B, 1, T)
        
        # 6. 整理输出
        x = x.transpose(1, 2) # (B, T, 1)
        x = x.reshape(-1, 1)   # (B*T, 1)
        
        return x


# end of app/models/dev.py
# end of app/models/cnn.py