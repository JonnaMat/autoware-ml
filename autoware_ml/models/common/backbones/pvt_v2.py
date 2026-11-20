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

"""Pyramid Vision Transformer v2 backbone.

A hierarchical transformer with four stages at strides 4, 8, 16 and 32 whose attention reads
a spatially reduced copy of the tokens ("PVT v2: Improved Baselines with Pyramid Vision
Transformer", https://arxiv.org/abs/2106.13797). The network and its ImageNet weights come
from timm (https://github.com/huggingface/pytorch-image-models, Apache-2.0). It has no
position embedding, so it runs at any input size divisible by 32.
"""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float
import timm
import torch
import torch.nn as nn

from autoware_ml.models.common.backbones.vit import explicit_attention_in_onnx_export


class PVTv2(nn.Module):
    """PVT v2 of timm behind the multi-scale tuple interface of the image backbones."""

    def __init__(
        self,
        model_name: str = "pvt_v2_b1.in1k",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize the backbone.

        Args:
            model_name: Name of a PVT v2 variant of timm with its pretrained tag.
            pretrained: Whether to download and load the pretrained weights of timm. Not
              needed when the weights come from a checkpoint of the detector.
            out_indices: Stages whose feature maps are returned, 0 (stride 4) to 3 (stride 32).
            drop_path_rate: Stochastic depth rate of the transformer blocks.

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
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=indices,
            drop_path_rate=drop_path_rate,
        )
        self.out_channels = tuple(self.model.feature_info.channels())
        self.out_strides = tuple(self.model.feature_info.reduction())

    def forward(
        self, x: Float[torch.Tensor, "batch_size 3 height width"]
    ) -> tuple[Float[torch.Tensor, "batch_size channels level_height level_width"], ...]:
        """Extract the feature maps of the stages selected by ``out_indices``.

        Args:
            x: Images, their height and width divisible by 32, normalized as the weights
              expect.

        Returns:
            One feature map per selected stage, from the shallowest to the deepest.
        """
        with explicit_attention_in_onnx_export(self.model):
            return tuple(self.model(x))
