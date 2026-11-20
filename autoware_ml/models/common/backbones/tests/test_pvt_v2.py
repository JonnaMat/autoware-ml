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

"""Unit tests for the PVT v2 backbone: the feature levels, their widths and the stage selection."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.common.backbones.pvt_v2 import PVTv2


class TestPVTv2(unittest.TestCase):
    """Unit tests for the PVT v2 backbone."""

    def setUp(self) -> None:
        """Set up a two-image batch of 64x96 inputs."""
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 64, 96)

    def test_b1_returns_strides_8_16_32(self) -> None:
        """Test that B1 returns 128, 320 and 512 channels at strides 8, 16 and 32."""
        backbone = PVTv2(model_name="pvt_v2_b1.in1k", pretrained=False).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_channels, (128, 320, 512))
        self.assertEqual(backbone.out_strides, (8, 16, 32))
        self.assertEqual(
            [tuple(f.shape) for f in features],
            [(2, 128, 8, 12), (2, 320, 4, 6), (2, 512, 2, 3)],
        )

    def test_out_indices_select_the_stages(self) -> None:
        """Test that every stage can be returned, the first at stride 4."""
        backbone = PVTv2(
            model_name="pvt_v2_b1.in1k", pretrained=False, out_indices=[0, 1, 2, 3]
        ).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_channels, (64, 128, 320, 512))
        self.assertEqual([f.shape[2] for f in features], [16, 8, 4, 2])

    def test_out_indices_are_validated(self) -> None:
        """Test that out_indices outside the stages, unordered or empty are refused."""
        for out_indices in ([], [4], [2, 1], [1, 1]):
            with self.subTest(out_indices=out_indices), self.assertRaises(ValueError):
                PVTv2(pretrained=False, out_indices=out_indices)
