"""
Perform classification for the segmented ROI
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from diffusers.models.resnet import Downsample2D


class ImageEmbeddingClassifier(nn.Module):
    """
    A simple CNN to perform classification given an image embedding 
    produced by SamVisionEncoder.
    Shape of SAM image embedding: (256, 64, 64)
    """

    def __init__(
        self,
        ch: int,
        ch_mults: list[int], 
        in_reso: int = 64,
        in_channels: int = 256,
        out_channels: int = 6,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_pre_norm: bool = True,
        num_blocks_per_layer: int = 1,
        downsample_padding = 1,
    ):
        super().__init__()
        resnets = []
        num_layers = len(ch_mults)
        for i in range(num_layers):
            layer = []
            in_ch = in_channels if i == 0 else ch * ch_mults[i-1]
            out_ch = ch * ch_mults[i]
            for j in range(num_blocks_per_layer):
                in_ch = in_ch if j == 0 else out_ch
                layer.append(
                    ResnetBlock2D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        eps=eps,
                        groups=groups,
                        dropout=dropout,
                        non_linearity=resnet_act_fn,
                        pre_norm=resnet_pre_norm
                    )
                )
            downsampler = Downsample2D(
                out_ch, use_conv=True, out_channels=out_ch, padding=downsample_padding, name="op"
            )
            layer.append(downsampler)
            layer = nn.Sequential(*layer)
            resnets.append(layer)

        self.resnets = nn.ModuleList(resnets)

        curr_reso = in_reso // 2 ** num_layers
        interm_ch = ch * ch_mults[0]
        self.out_conv = nn.Sequential( 
            nn.GroupNorm(num_groups=groups, num_channels=out_ch, eps=eps, affine=True),
            nn.GELU(),
            nn.Conv2d(out_ch, interm_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=interm_ch, eps=eps, affine=True),
            nn.GELU(),
            nn.Conv2d(interm_ch, out_channels, kernel_size=curr_reso)          
        )

    def forward(self, image_emb):
        x = image_emb
        for layer in self.resnets:
            x = layer(x)
        logis = self.out_conv(x).squeeze()
        return logis


# Copied and modified from diffusers.models.resnet.ResnetBlock2D
class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = torch.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


