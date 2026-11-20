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

"""CSPDarknet backbone of YOLOX.

Ported from the YOLOX reference implementation (``yolox/models/darknet.py`` of
https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0). The backbone follows the
multi-scale tuple interface of the other image backbones: it returns the feature maps of
the selected stages, by default those at strides 8, 16 and 32.
"""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float
import torch
import torch.nn as nn

from autoware_ml.models.common.layers.csp import (
    ConvBNAct,
    CSPLayer,
    Focus,
    SPPBottleneck,
)


class CSPDarknet(nn.Module):
    """CSPDarknet with a Focus stem and four downsampling stages.

    ``deepen_factor`` and ``widen_factor`` scale the block counts and the channel widths of
    the base network: YOLOX-S is 0.33 / 0.5, M 0.67 / 0.75, L 1.0 / 1.0 and X 1.33 / 1.25.
    """

    def __init__(
        self,
        deepen_factor: float = 0.33,
        widen_factor: float = 0.5,
        in_channels: int = 3,
        out_indices: Sequence[int] = (1, 2, 3),
        activation: str = "silu",
    ) -> None:
        """Initialize the backbone.

        Args:
            deepen_factor: Multiplier of the number of bottlenecks of every stage.
            widen_factor: Multiplier of the channel width of every stage.
            in_channels: Channels of the input image.
            out_indices: Stages whose feature maps are returned, 0 (stride 4) to 3 (stride 32).
            activation: Activation name of every block.

        Raises:
            ValueError: If ``out_indices`` is empty, not increasing or outside the stages.
        """
        super().__init__()
        indices = tuple(int(index) for index in out_indices)
        if not indices or any(not 0 <= index <= 3 for index in indices):
            raise ValueError(f"out_indices must select stages in [0, 3], got {indices}.")
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(f"out_indices must be strictly increasing, got {indices}.")
        self.out_indices = indices

        base_channels = int(widen_factor * 64)
        base_depth = max(round(deepen_factor * 3), 1)
        channels = [
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.out_channels = tuple(channels[index] for index in indices)

        self.stem = Focus(in_channels, base_channels, kernel_size=3, activation=activation)
        self.dark2 = nn.Sequential(
            ConvBNAct(base_channels, channels[0], 3, 2, activation=activation),
            CSPLayer(channels[0], channels[0], num_blocks=base_depth, activation=activation),
        )
        self.dark3 = nn.Sequential(
            ConvBNAct(channels[0], channels[1], 3, 2, activation=activation),
            CSPLayer(
                channels[1],
                channels[1],
                num_blocks=base_depth * 3,
                activation=activation,
            ),
        )
        self.dark4 = nn.Sequential(
            ConvBNAct(channels[1], channels[2], 3, 2, activation=activation),
            CSPLayer(
                channels[2],
                channels[2],
                num_blocks=base_depth * 3,
                activation=activation,
            ),
        )
        self.dark5 = nn.Sequential(
            ConvBNAct(channels[2], channels[3], 3, 2, activation=activation),
            SPPBottleneck(channels[3], channels[3], activation=activation),
            CSPLayer(
                channels[3],
                channels[3],
                num_blocks=base_depth,
                shortcut=False,
                activation=activation,
            ),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> tuple[Float[torch.Tensor, "batch_size channels level_height level_width"], ...]:
        """Extract the feature maps of the stages selected by ``out_indices``.

        Args:
            x: Images, their height and width divisible by 32.

        Returns:
            One feature map per selected stage, from the shallowest to the deepest.
        """
        x = self.stem(x)
        stages = []
        for stage in (self.dark2, self.dark3, self.dark4, self.dark5):
            x = stage(x)
            stages.append(x)
        return tuple(stages[index] for index in self.out_indices)
