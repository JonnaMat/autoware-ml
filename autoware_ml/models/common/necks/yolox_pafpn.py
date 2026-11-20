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

"""Path aggregation feature pyramid neck of YOLOX.

Ported from the YOLOX reference implementation (``yolox/models/yolo_pafpn.py`` of
https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0). The reference class owns its
backbone; here the neck is a module of its own that takes the three feature maps of any
backbone, so the backbone is swapped by config. A fourth, finer feature map extends both
paths by one step; the three-level neck is unchanged by it.
"""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.models.common.layers.csp import ConvBNAct, CSPLayer


class YOLOXPAFPN(nn.Module):
    """Top-down then bottom-up fusion of three or four feature maps with CSP layers.

    The output level ``i`` keeps the channel count and the stride of the input level ``i``.
    With three levels the neck is that of the reference. With four, the finest level is fused
    by one more top-down and one more bottom-up step built like the reference ones; the
    modules of the three coarser levels keep their names and their shapes.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 256, 512),
        deepen_factor: float = 0.33,
        activation: str = "silu",
    ) -> None:
        """Initialize the neck.

        Args:
            in_channels: Channels of the three or four input feature maps, shallowest first.
            deepen_factor: Multiplier of the number of bottlenecks of every CSP layer.
            activation: Activation name of every block.

        Raises:
            ValueError: If ``in_channels`` names neither three nor four levels.
        """
        super().__init__()
        if len(in_channels) not in (3, 4):
            raise ValueError(f"YOLOXPAFPN fuses three or four levels, got {len(in_channels)}.")
        self.in_channels = tuple(int(channels) for channels in in_channels)
        self.out_channels = self.in_channels
        self.num_levels = len(self.in_channels)
        c3, c4, c5 = self.in_channels[-3:]
        num_blocks = max(round(3 * deepen_factor), 1)

        self.lateral_conv0 = ConvBNAct(c5, c4, 1, 1, activation=activation)
        self.C3_p4 = CSPLayer(2 * c4, c4, num_blocks, shortcut=False, activation=activation)
        self.reduce_conv1 = ConvBNAct(c4, c3, 1, 1, activation=activation)
        self.C3_p3 = CSPLayer(2 * c3, c3, num_blocks, shortcut=False, activation=activation)
        self.bu_conv2 = ConvBNAct(c3, c3, 3, 2, activation=activation)
        self.C3_n3 = CSPLayer(2 * c3, c4, num_blocks, shortcut=False, activation=activation)
        self.bu_conv1 = ConvBNAct(c4, c4, 3, 2, activation=activation)
        self.C3_n4 = CSPLayer(2 * c4, c5, num_blocks, shortcut=False, activation=activation)
        if self.num_levels == 4:
            c2 = self.in_channels[0]
            self.reduce_conv2 = ConvBNAct(c3, c2, 1, 1, activation=activation)
            self.C3_p2 = CSPLayer(2 * c2, c2, num_blocks, shortcut=False, activation=activation)
            self.bu_conv3 = ConvBNAct(c2, c2, 3, 2, activation=activation)
            self.C3_n2 = CSPLayer(2 * c2, c3, num_blocks, shortcut=False, activation=activation)

    def forward(
        self, inputs: Sequence[Float[torch.Tensor, "batch_size channels height width"]]
    ) -> tuple[Float[torch.Tensor, "batch_size channels height width"], ...]:
        """Fuse the feature maps.

        Args:
            inputs: Feature maps at strides 8, 16 and 32, shallowest first, preceded by the
              one at stride 4 for a neck of four levels.

        Returns:
            The fused feature maps at the same strides.

        Raises:
            ValueError: If the number of feature maps is not the number of levels.
        """
        if len(inputs) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature maps, got {len(inputs)}.")
        x2, x1, x0 = inputs[-3:]

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = torch.cat([F.interpolate(fpn_out0, scale_factor=2.0, mode="nearest"), x1], dim=1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = torch.cat([F.interpolate(fpn_out1, scale_factor=2.0, mode="nearest"), x2], dim=1)
        pan_out2 = self.C3_p3(f_out1)

        if self.num_levels == 4:
            fpn_out2 = self.reduce_conv2(pan_out2)
            f_out2 = torch.cat(
                [F.interpolate(fpn_out2, scale_factor=2.0, mode="nearest"), inputs[0]], dim=1
            )
            pan_out3 = self.C3_p2(f_out2)
            p_out2 = torch.cat([self.bu_conv3(pan_out3), fpn_out2], dim=1)
            pan_out2 = self.C3_n2(p_out2)

        p_out1 = torch.cat([self.bu_conv2(pan_out2), fpn_out1], dim=1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = torch.cat([self.bu_conv1(pan_out1), fpn_out0], dim=1)
        pan_out0 = self.C3_n4(p_out0)

        if self.num_levels == 4:
            return pan_out3, pan_out2, pan_out1, pan_out0
        return pan_out2, pan_out1, pan_out0
