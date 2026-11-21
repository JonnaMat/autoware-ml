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

"""Unit tests for the YOLOX PAFPN neck: it keeps the channels and strides of its inputs, and
its fourth level leaves the three-level neck as it is."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from autoware_ml.models.common.backbones.csp_darknet import CSPDarknet
from autoware_ml.models.common.backbones.resnet import ResNet18MultiScale, ResNet50MultiScale
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

    def test_neck_takes_the_features_of_the_wider_backbones(self) -> None:
        """Test that the neck fuses the features of the YOLOX-M CSPDarknet and of a ResNet-50.

        Check: the output levels keep the widths of the backbone, 192, 384 and 768 channels
        or 512, 1024 and 2048.
        """
        images = torch.randn(1, 3, 64, 64)
        backbones = {
            "csp_darknet_m": CSPDarknet(deepen_factor=0.67, widen_factor=0.75),
            "resnet50": ResNet50MultiScale(in_channels=3, out_indices=[1, 2, 3]),
        }
        widths = {"csp_darknet_m": (192, 384, 768), "resnet50": (512, 1024, 2048)}
        for name, backbone in backbones.items():
            with self.subTest(backbone=name), torch.no_grad():
                neck = YOLOXPAFPN(in_channels=widths[name], deepen_factor=0.67).eval()
                outputs = neck(backbone.eval()(images))
                self.assertEqual(
                    [tuple(o.shape[1:]) for o in outputs],
                    [(widths[name][0], 8, 8), (widths[name][1], 4, 4), (widths[name][2], 2, 2)],
                )

    def test_medium_neck_has_the_depth_and_parameter_count_of_the_reference(self) -> None:
        """Test that the YOLOX-M neck holds two bottlenecks per CSP layer and 8.68 M parameters."""
        neck = YOLOXPAFPN(in_channels=[192, 384, 768], deepen_factor=0.67)

        num_bottlenecks = [
            len(layer.m) for layer in (neck.C3_p4, neck.C3_p3, neck.C3_n3, neck.C3_n4)
        ]

        self.assertEqual(num_bottlenecks, [2, 2, 2, 2])
        self.assertEqual(sum(p.numel() for p in neck.parameters()), 8_679_168)

    def test_number_of_levels_is_checked(self) -> None:
        """Test that fewer than three or more than four levels are refused, at build and at call."""
        with self.assertRaises(ValueError):
            YOLOXPAFPN(in_channels=[32, 64])
        with self.assertRaises(ValueError):
            YOLOXPAFPN(in_channels=[8, 16, 32, 64, 128])
        with self.assertRaises(ValueError):
            YOLOXPAFPN(in_channels=[32, 64, 128])([torch.randn(1, 32, 8, 8)])
        with self.assertRaises(ValueError):
            YOLOXPAFPN(in_channels=[16, 32, 64, 128])(
                [torch.randn(1, 32, 16, 16), torch.randn(1, 64, 8, 8), torch.randn(1, 128, 4, 4)]
            )


# Parameter names of the three-level neck, those of the reference checkpoints.
_THREE_LEVEL_MODULES = (
    "lateral_conv0",
    "C3_p4",
    "reduce_conv1",
    "C3_p3",
    "bu_conv2",
    "C3_n3",
    "bu_conv1",
    "C3_n4",
)
_FOURTH_LEVEL_MODULES = ("reduce_conv2", "C3_p2", "bu_conv3", "C3_n2")


def _reference_three_level_forward(
    neck: YOLOXPAFPN, inputs: list[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Fuse three feature maps as the reference neck does, with the modules of ``neck``."""
    x2, x1, x0 = inputs
    fpn_out0 = neck.lateral_conv0(x0)
    f_out0 = neck.C3_p4(torch.cat([F.interpolate(fpn_out0, scale_factor=2.0), x1], dim=1))
    fpn_out1 = neck.reduce_conv1(f_out0)
    pan_out2 = neck.C3_p3(torch.cat([F.interpolate(fpn_out1, scale_factor=2.0), x2], dim=1))
    pan_out1 = neck.C3_n3(torch.cat([neck.bu_conv2(pan_out2), fpn_out1], dim=1))
    pan_out0 = neck.C3_n4(torch.cat([neck.bu_conv1(pan_out1), fpn_out0], dim=1))
    return pan_out2, pan_out1, pan_out0


class TestYOLOXPAFPNLevels(unittest.TestCase):
    """The fourth level extends the neck without changing the three-level one."""

    def setUp(self) -> None:
        """Set up the feature maps of a 128x128 image at strides 4, 8, 16 and 32."""
        torch.manual_seed(0)
        self.inputs = [
            torch.randn(2, 16, 32, 32),
            torch.randn(2, 32, 16, 16),
            torch.randn(2, 64, 8, 8),
            torch.randn(2, 128, 4, 4),
        ]

    def test_three_level_neck_keeps_the_modules_of_the_reference(self) -> None:
        """Test that the three-level neck holds exactly the modules of the reference neck."""
        neck = YOLOXPAFPN(in_channels=[32, 64, 128])

        self.assertEqual(tuple(name for name, _ in neck.named_children()), _THREE_LEVEL_MODULES)
        self.assertEqual(
            {key.split(".")[0] for key in neck.state_dict()}, set(_THREE_LEVEL_MODULES)
        )
        self.assertEqual(sum(p.numel() for p in YOLOXPAFPN().parameters()), 2_834_688)

    def test_three_level_neck_computes_the_reference_fusion(self) -> None:
        """Test that the three-level output equals the fusion of the reference, bit for bit."""
        neck = YOLOXPAFPN(in_channels=[32, 64, 128]).eval()

        with torch.no_grad():
            outputs = neck(self.inputs[1:])
            expected = _reference_three_level_forward(neck, self.inputs[1:])

        for output, reference in zip(outputs, expected, strict=True):
            self.assertTrue(torch.equal(output, reference))

    def test_three_level_initialization_does_not_depend_on_the_extension(self) -> None:
        """Test that a seed draws the same three-level weights as before the extension.

        Check: the first and the last module built draw the values the seed gives a neck
        that builds only the reference modules in the reference order.
        """
        torch.manual_seed(3)
        neck = YOLOXPAFPN(in_channels=[32, 64, 128])
        torch.manual_seed(3)
        first = torch.nn.Conv2d(128, 64, 1, bias=False).weight

        self.assertTrue(torch.equal(neck.lateral_conv0.conv.weight, first))

    def test_four_level_neck_adds_one_step_to_each_path(self) -> None:
        """Test that the four-level neck adds four modules and keeps the shapes of the others."""
        three = YOLOXPAFPN(in_channels=[32, 64, 128])
        four = YOLOXPAFPN(in_channels=[16, 32, 64, 128])

        self.assertEqual(
            tuple(name for name, _ in four.named_children()),
            _THREE_LEVEL_MODULES + _FOURTH_LEVEL_MODULES,
        )
        for key, value in three.state_dict().items():
            self.assertEqual(four.state_dict()[key].shape, value.shape)
        # A three-level checkpoint initializes the coarser levels of the four-level neck.
        result = four.load_state_dict(three.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(
            all(key.split(".")[0] in _FOURTH_LEVEL_MODULES for key in result.missing_keys)
        )

    def test_four_level_output_keeps_channels_and_sizes(self) -> None:
        """Test that every output level has the channels and the size of its input level."""
        neck = YOLOXPAFPN(in_channels=[16, 32, 64, 128]).eval()

        with torch.no_grad():
            outputs = neck(self.inputs)

        self.assertEqual(neck.out_channels, (16, 32, 64, 128))
        self.assertEqual([tuple(o.shape) for o in outputs], [tuple(i.shape) for i in self.inputs])

    def test_finest_level_reaches_every_output(self) -> None:
        """Test that the stride 4 input changes every output level of the four-level neck."""
        neck = YOLOXPAFPN(in_channels=[16, 32, 64, 128]).eval()
        changed = [self.inputs[0] + 1.0, *self.inputs[1:]]

        with torch.no_grad():
            outputs = neck(self.inputs)
            outputs_changed = neck(changed)

        for output, other in zip(outputs, outputs_changed, strict=True):
            self.assertFalse(torch.equal(output, other))

    def test_neck_takes_four_levels_of_either_backbone(self) -> None:
        """Test that the four-level neck fuses the stages of the CSPDarknet and of a ResNet-18."""
        images = torch.randn(1, 3, 64, 64)
        backbones = {
            "csp_darknet": CSPDarknet(
                deepen_factor=0.33, widen_factor=0.5, out_indices=[0, 1, 2, 3]
            ),
            "resnet18": ResNet18MultiScale(in_channels=3, out_indices=[0, 1, 2, 3]),
        }
        for name, backbone in backbones.items():
            with self.subTest(backbone=name), torch.no_grad():
                neck = YOLOXPAFPN(in_channels=[64, 128, 256, 512]).eval()
                outputs = neck(backbone.eval()(images))
                self.assertEqual(
                    [tuple(o.shape[1:]) for o in outputs],
                    [(64, 16, 16), (128, 8, 8), (256, 4, 4), (512, 2, 2)],
                )
