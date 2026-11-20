# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convolution blocks of the CSPDarknet family.

Ported from the YOLOX reference implementation (``yolox/models/network_blocks.py`` of
https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0). The submodule names follow the
reference, so its checkpoints load after a key rename.
"""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float
import torch
import torch.nn as nn

BN_EPS = 1e-3
BN_MOMENTUM = 0.03


def build_activation(name: str) -> nn.Module:
    """Build the activation of a block.

    Args:
        name: ``silu``, ``relu`` or ``lrelu``.

    Returns:
        The activation module.

    Raises:
        ValueError: If the name is none of the supported activations.
    """
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unsupported activation {name!r}, expected silu, relu or lrelu.")


class ConvBNAct(nn.Module):
    """Convolution, batch normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        activation: str = "silu",
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Kernel size, padded so a stride of 1 keeps the resolution.
            stride: Convolution stride.
            groups: Convolution groups.
            activation: Activation name, see :func:`build_activation`.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.act = build_activation(activation)

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels out_height out_width"]:
        """Apply convolution, normalization and activation."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """1x1 then 3x3 convolution with an optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        activation: str = "silu",
    ) -> None:
        """Initialize the bottleneck.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            shortcut: Whether to add the input, when the channel counts allow it.
            expansion: Hidden channels as a share of the output channels.
            activation: Activation name.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = ConvBNAct(hidden_channels, out_channels, 3, 1, activation=activation)
        self.use_add = shortcut and in_channels == out_channels

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels height width"]:
        """Apply the two convolutions and the residual connection."""
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling: max pools of several kernel sizes concatenated."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
        activation: str = "silu",
    ) -> None:
        """Initialize the pooling block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_sizes: Kernel sizes of the max pools.
            activation: Activation name.
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, activation=activation)
        self.m = nn.ModuleList(
            nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            for kernel_size in kernel_sizes
        )
        self.conv2 = ConvBNAct(
            hidden_channels * (len(kernel_sizes) + 1),
            out_channels,
            1,
            1,
            activation=activation,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels height width"]:
        """Pool the reduced features at every kernel size and fuse them."""
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.m], dim=1))


class CSPLayer(nn.Module):
    """Cross stage partial layer: a stack of bottlenecks beside a 1x1 bypass."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        activation: str = "silu",
    ) -> None:
        """Initialize the layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            num_blocks: Number of bottlenecks.
            shortcut: Whether the bottlenecks are residual.
            expansion: Hidden channels as a share of the output channels.
            activation: Activation name.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = ConvBNAct(in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv3 = ConvBNAct(2 * hidden_channels, out_channels, 1, 1, activation=activation)
        self.m = nn.Sequential(
            *(
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut,
                    1.0,
                    activation=activation,
                )
                for _ in range(num_blocks)
            )
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels height width"]:
        """Fuse the bottleneck branch with the bypass branch."""
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class Focus(nn.Module):
    """Space to depth stem: every 2x2 pixel block becomes four channels, then a convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        activation: str = "silu",
    ) -> None:
        """Initialize the stem.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Kernel size of the convolution.
            stride: Stride of the convolution.
            activation: Activation name.
        """
        super().__init__()
        self.conv = ConvBNAct(
            in_channels * 4, out_channels, kernel_size, stride, activation=activation
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels half_height half_width"]:
        """Stack the four pixel phases along the channels and convolve them."""
        top_left = x[..., ::2, ::2]
        top_right = x[..., ::2, 1::2]
        bottom_left = x[..., 1::2, ::2]
        bottom_right = x[..., 1::2, 1::2]
        return self.conv(torch.cat((top_left, bottom_left, top_right, bottom_right), dim=1))
