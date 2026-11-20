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

"""Unit tests for the YOLOX PAFPN neck: it keeps the channels and strides of its inputs."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.common.backbones.csp_darknet import CSPDarknet
from autoware_ml.models.common.backbones.resnet import ResNet18MultiScale
from autoware_ml.models.common.necks.yolox_pafpn import YOLOXPAFPN


class TestYOLOXPAFPN(unittest.TestCase):
    """Unit tests for the YOLOX PAFPN neck."""

    def test_output_levels_keep_channels_and_sizes(self) -> None:
        """Test that every output level has the channels and the size of its input level."""
        neck = YOLOXPAFPN(in_channels=[32, 64, 128]).eval()
        inputs = [torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8), torch.randn(2, 128, 4, 4)]

        with torch.no_grad():
            outputs = neck(inputs)

        self.assertEqual([tuple(o.shape) for o in outputs], [tuple(i.shape) for i in inputs])
        self.assertEqual(neck.out_channels, (32, 64, 128))

    def test_neck_takes_the_features_of_either_backbone(self) -> None:
        """Test that the neck fuses the features of the CSPDarknet and of a ResNet-18 alike."""
        images = torch.randn(1, 3, 64, 64)
        backbones = {
            "csp_darknet": CSPDarknet(deepen_factor=0.33, widen_factor=0.5),
            "resnet18": ResNet18MultiScale(in_channels=3, out_indices=[1, 2, 3]),
        }
        for name, backbone in backbones.items():
            with self.subTest(backbone=name), torch.no_grad():
                outputs = YOLOXPAFPN(in_channels=[128, 256, 512]).eval()(backbone.eval()(images))
                self.assertEqual(
                    [tuple(o.shape[1:]) for o in outputs], [(128, 8, 8), (256, 4, 4), (512, 2, 2)]
                )

    def test_number_of_levels_is_checked(self) -> None:
        """Test that another number of levels than three is refused, at build and at call."""
        with self.assertRaises(ValueError):
            YOLOXPAFPN(in_channels=[32, 64])
        with self.assertRaises(ValueError):
            YOLOXPAFPN(in_channels=[32, 64, 128])([torch.randn(1, 32, 8, 8)])
