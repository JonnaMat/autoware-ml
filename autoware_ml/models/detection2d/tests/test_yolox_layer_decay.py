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

"""Unit tests for the YOLOX detector with layer-wise learning rates of its backbone."""

from __future__ import annotations

from functools import partial
import unittest

import torch
import torch.nn as nn

from autoware_ml.models.common.backbones.csp_darknet import CSPDarknet
from autoware_ml.models.common.backbones.vit import ViTSimplePyramid
from autoware_ml.models.common.necks.yolox_pafpn import YOLOXPAFPN
from autoware_ml.models.detection2d.heads.yolox import YOLOXHead
from autoware_ml.models.detection2d.yolox_layer_decay import YOLOXLayerDecayDetector
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


def build_detector(backbone: nn.Module, **kwargs: object) -> YOLOXLayerDecayDetector:
    """Build a detector with a narrow head around a backbone, optimized by AdamW at 1e-3."""
    return YOLOXLayerDecayDetector(
        backbone=backbone,
        neck=YOLOXPAFPN(in_channels=backbone.out_channels),
        head=YOLOXHead(num_classes=2, in_channels=backbone.out_channels, feat_channels=16),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=0.05),
        optimizer_group_overrides={"no_decay": {"weight_decay": 0.0}},
        use_ema=True,
        **kwargs,
    )


def build_vit() -> ViTSimplePyramid:
    """Build a randomly initialized transformer backbone with narrow pyramid levels."""
    return ViTSimplePyramid(
        image_size=[64, 64], pretrained=False, window_size=2, out_channels=[16, 32, 64]
    )


class TestYOLOXLayerDecayDetector(unittest.TestCase):
    """Unit tests for the optimizer groups of the detector."""

    def test_every_trained_parameter_is_in_one_group(self) -> None:
        """Test that the groups hold each parameter of the network once and none of the average."""
        detector = build_detector(build_vit(), layer_decay=0.8)

        groups = detector.build_optimizer_groups()

        grouped = [id(parameter) for parameters in groups.values() for parameter in parameters]
        expected = {id(parameter) for parameter in detector.network.parameters()}
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), expected)
        self.assertEqual(
            {name for name in groups if name.startswith("backbone_layer_")},
            {f"backbone_layer_{layer}_{kind}" for layer in range(14) for kind in ("no_decay",)}
            | {f"backbone_layer_{layer}_decay" for layer in range(13)},
        )

    def test_embeddings_and_pyramid_are_grouped_by_their_role(self) -> None:
        """Test that the embeddings are not decayed and the pyramid trains with the head."""
        detector = build_detector(build_vit(), layer_decay=0.8)
        backbone = detector.network.backbone

        groups = detector.build_optimizer_groups()

        def group_of(parameter: nn.Parameter) -> str:
            return next(
                name for name, members in groups.items() if any(parameter is m for m in members)
            )

        self.assertEqual(group_of(backbone.vit.pos_embed), "backbone_layer_0_no_decay")
        self.assertEqual(group_of(backbone.vit.cls_token), "backbone_layer_0_no_decay")
        self.assertEqual(group_of(backbone.vit.patch_embed.proj.weight), "backbone_layer_0_decay")
        self.assertEqual(group_of(backbone.vit.blocks[3].attn.qkv.weight), "backbone_layer_4_decay")
        self.assertEqual(group_of(backbone.vit.blocks[3].ls1.gamma), "backbone_layer_4_no_decay")
        self.assertEqual(group_of(backbone.vit.norm.weight), "backbone_layer_13_no_decay")
        self.assertEqual(group_of(backbone.pyramid.level3[0].conv.weight), "decay")
        self.assertEqual(group_of(backbone.pyramid.level3[0].bn.weight), "no_decay")
        self.assertEqual(group_of(detector.network.head.obj_preds[0].bias), "no_decay")

    def test_learning_rates_decay_towards_the_embeddings(self) -> None:
        """Test the rate and the weight decay of the optimizer groups."""
        detector = build_detector(build_vit(), backbone_lr_factor=0.5, layer_decay=0.8)

        optimizer = detector.configure_optimizers()

        names = list(detector.build_optimizer_groups())
        by_name = dict(zip(names, optimizer.param_groups, strict=True))
        self.assertAlmostEqual(by_name["decay"]["lr"], 1e-3)
        self.assertAlmostEqual(by_name["no_decay"]["lr"], 1e-3)
        self.assertAlmostEqual(by_name["backbone_layer_13_no_decay"]["lr"], 0.5e-3)
        self.assertAlmostEqual(by_name["backbone_layer_12_decay"]["lr"], 0.5e-3 * 0.8)
        self.assertAlmostEqual(by_name["backbone_layer_0_decay"]["lr"], 0.5e-3 * 0.8**13)
        self.assertEqual(by_name["backbone_layer_0_decay"]["weight_decay"], 0.05)
        self.assertEqual(by_name["backbone_layer_0_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(by_name["no_decay"]["weight_decay"], 0.0)
        # Building the optimizer again gives the same groups.
        self.assertEqual(len(detector.configure_optimizers().param_groups), len(names))

    def test_default_factors_keep_one_learning_rate(self) -> None:
        """Test that without a factor and a decay every group trains at the optimizer rate."""
        detector = build_detector(build_vit())

        optimizer = detector.configure_optimizers()

        self.assertEqual({group["lr"] for group in optimizer.param_groups}, {1e-3})

    def test_backbone_without_layers_is_one_layer(self) -> None:
        """Test that a backbone that tells no layers trains at the backbone factor as a whole."""
        detector = build_detector(
            CSPDarknet(widen_factor=0.125), backbone_lr_factor=0.1, layer_decay=0.5
        )

        optimizer = detector.configure_optimizers()

        names = list(detector.build_optimizer_groups())
        self.assertEqual(
            names, ["decay", "no_decay", "backbone_layer_0_decay", "backbone_layer_0_no_decay"]
        )
        by_name = dict(zip(names, optimizer.param_groups, strict=True))
        self.assertAlmostEqual(by_name["backbone_layer_0_decay"]["lr"], 1e-4)
        self.assertAlmostEqual(by_name["backbone_layer_0_no_decay"]["lr"], 1e-4)
        self.assertAlmostEqual(by_name["decay"]["lr"], 1e-3)
        backbone_parameters = sum(
            len(by_name[name]["params"]) for name in names if name.startswith("backbone_")
        )
        self.assertEqual(backbone_parameters, len(list(detector.network.backbone.parameters())))

    def test_factors_are_validated(self) -> None:
        """Test that factors outside (0, 1] are refused."""
        for arguments in ({"backbone_lr_factor": 0.0}, {"layer_decay": 1.5}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_detector(CSPDarknet(widen_factor=0.125), **arguments)

    def test_optimizer_without_a_learning_rate_is_refused(self) -> None:
        """Test that the rates of the layers need the rate of the optimizer."""
        detector = build_detector(CSPDarknet(widen_factor=0.125))
        detector.optimizer_partial = partial(torch.optim.AdamW)

        with self.assertRaises(ValueError):
            detector.configure_optimizers()
