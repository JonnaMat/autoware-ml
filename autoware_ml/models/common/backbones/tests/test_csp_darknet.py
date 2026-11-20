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

"""Unit tests for the CSPDarknet backbone: the feature levels, their widths and the stage selection."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.common.backbones.csp_darknet import CSPDarknet


class TestCSPDarknet(unittest.TestCase):
    """Unit tests for the CSPDarknet backbone."""

    def setUp(self) -> None:
        """Set up a two-image batch of 128x128 inputs."""
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 128, 128)

    def test_small_variant_returns_strides_8_16_32(self) -> None:
        """Test that the YOLOX-S factors give 128, 256 and 512 channels at strides 8, 16, 32."""
        backbone = CSPDarknet(deepen_factor=0.33, widen_factor=0.5).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_channels, (128, 256, 512))
        self.assertEqual(
            [tuple(f.shape) for f in features], [(2, 128, 16, 16), (2, 256, 8, 8), (2, 512, 4, 4)]
        )

    def test_small_variant_has_the_parameter_count_of_the_reference(self) -> None:
        """Test that the YOLOX-S backbone holds the 4.21 M parameters of the reference backbone."""
        backbone = CSPDarknet(deepen_factor=0.33, widen_factor=0.5)

        num_parameters = sum(parameter.numel() for parameter in backbone.parameters())

        self.assertEqual(num_parameters, 4_212_672)

    def test_widen_factor_scales_the_channels(self) -> None:
        """Test that the widen factor scales the channels of every returned level."""
        backbone = CSPDarknet(
            deepen_factor=0.33, widen_factor=0.25, out_indices=[0, 1, 2, 3]
        ).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual([f.shape[1] for f in features], [32, 64, 128, 256])
        self.assertEqual([f.shape[2] for f in features], [32, 16, 8, 4])

    def test_out_indices_are_validated(self) -> None:
        """Test that out_indices outside the stages, unordered or empty are refused."""
        for out_indices in ([], [4], [2, 1], [1, 1]):
            with self.subTest(out_indices=out_indices), self.assertRaises(ValueError):
                CSPDarknet(out_indices=out_indices)
