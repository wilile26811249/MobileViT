from typing import Callable, Any, Optional, List

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.SiLU,
        dilation: int = 1
    ):
        super(ConvNormAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation = dilation, groups = groups, bias = norm_layer is None)

        self.norm_layer = nn.BatchNorm2d(out_channels) if norm_layer is None else norm_layer(out_channels)
        self.act = activation_layer() if activation_layer is not None else activation_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act is not None:
            x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    """
    MobileNetv2 InvertedResidual block
    """
    def __init__(self, in_channels, out_channels, stride = 1, expand_ratio = 2, act_layer = nn.SiLU):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size = 1, activation_layer = None))

        # Depth-wise convolution
        layers.append(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size = 3, stride = stride,
                        padding = 1, groups = hidden_dim, activation_layer = act_layer)
        )
        # Point-wise convolution
        layers.append(
            nn.Conv2d(hidden_dim, out_channels, kernel_size = 1, stride = 1, bias = False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileVitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, layers, feature_size):
        super(MobileVitBlock, self).__init__()
        # Local representation
        self.local_representation = nn.Sequential(
            # Encode local spatial information
            ConvNormAct(in_channels, in_channels, 3),
            # Projects the tensor to a high-diementional space
            ConvNormAct(in_channels, d_model, 1)
        )

        # Global representation
        global_representation = []
        token_size = ((feature_size - 1) // 2)**2
        global_representation.append(nn.Unfold(kernel_size = 2, stride = 2))
        global_representation.append(nn.Unfold(kernel_size = 2))

        encoder_layer = nn.TransformerEncoderLayer(token_size, nhead = 1, dim_feedforward = d_model)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = layers)
        global_representation.append(transformer_encoder)

        global_representation.append(nn.Fold((feature_size, feature_size), 2))
        self.global_representation = nn.Sequential(*global_representation)

        # Fusion block
        self.fusion_block1 = nn.Conv2d(d_model, in_channels, kernel_size = 1)
        self.fusion_block2 = nn.Conv2d(in_channels * 2, out_channels, 3, padding = 1)

    def forward(self, x):
        local_repr = self.local_representation(x)
        global_repr = self.global_representation(local_repr)

        # Fuse the local and gloval features in the concatenation tensor
        fuse_repr = self.fusion_block1(global_repr)
        result = self.fusion_block2(torch.cat([x, fuse_repr], dim = 1))
        return result
