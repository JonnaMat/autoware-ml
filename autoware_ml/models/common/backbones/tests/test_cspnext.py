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

"""Unit tests for the CSPNeXt backbone: the feature levels and their widths, the parameter
names shared with MMDetection and the loading of classifier and detector checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle
import tempfile
import unittest

import torch

from autoware_ml.models.common.backbones.cspnext import CSPNeXt, load_backbone_state_dict


class TrainingLog:
    """Stands for the framework objects a training checkpoint holds next to its weights."""

    def __init__(self) -> None:
        """Hold some state a weights-only load refuses to rebuild."""
        self.history = [1.0, 2.0]


class TestCSPNeXt(unittest.TestCase):
    """Unit tests for the feature maps of the CSPNeXt backbone."""

    def setUp(self) -> None:
        """Set up a two-image batch of 128x128 inputs."""
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 128, 128)

    def test_small_variant_returns_strides_8_16_32(self) -> None:
        """Test that the RTMDet-S factors give 128, 256 and 512 channels at strides 8, 16, 32."""
        backbone = CSPNeXt(deepen_factor=0.33, widen_factor=0.5).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_channels, (128, 256, 512))
        self.assertEqual(
            [tuple(f.shape) for f in features], [(2, 128, 16, 16), (2, 256, 8, 8), (2, 512, 4, 4)]
        )

    def test_small_variant_has_the_parameter_count_of_the_reference(self) -> None:
        """Test that the RTMDet-S backbone holds the 4.38 M parameters of the reference."""
        backbone = CSPNeXt(deepen_factor=0.33, widen_factor=0.5)

        num_parameters = sum(parameter.numel() for parameter in backbone.parameters())

        self.assertEqual(num_parameters, 4_378_320)
        self.assertEqual(len(backbone.state_dict()), 242)

    def test_four_levels_add_the_stride_4_map(self) -> None:
        """Test that ``out_indices=[0, 1, 2, 3]`` adds the 64 channel map at stride 4."""
        backbone = CSPNeXt(out_indices=[0, 1, 2, 3]).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual([f.shape[1] for f in features], [64, 128, 256, 512])
        self.assertEqual([f.shape[2] for f in features], [32, 16, 8, 4])

    def test_parameter_names_follow_the_reference(self) -> None:
        """Test that the stem, block, attention and pooling names are those of MMDetection."""
        keys = set(CSPNeXt().state_dict())

        for key in (
            "stem.0.conv.weight",
            "stem.2.bn.running_var",
            "stage1.0.conv.weight",
            "stage1.1.main_conv.conv.weight",
            "stage2.1.blocks.1.conv2.depthwise_conv.conv.weight",
            "stage2.1.blocks.1.conv2.pointwise_conv.bn.weight",
            "stage3.1.attention.fc.bias",
            "stage4.1.conv2.conv.weight",
            "stage4.2.final_conv.conv.weight",
        ):
            self.assertIn(key, keys)

    def test_last_stage_is_not_residual(self) -> None:
        """Test that only the blocks of the first three stages add their input."""
        backbone = CSPNeXt()

        self.assertTrue(backbone.stage1[1].blocks[0].add_identity)
        self.assertTrue(backbone.stage3[1].blocks[1].add_identity)
        self.assertFalse(backbone.stage4[2].blocks[0].add_identity)

    def test_out_indices_are_validated(self) -> None:
        """Test that out_indices outside the stages, unordered or empty are refused."""
        for out_indices in ([], [4], [2, 1], [1, 1]):
            with self.subTest(out_indices=out_indices), self.assertRaises(ValueError):
                CSPNeXt(out_indices=out_indices)


class TestCSPNeXtWeights(unittest.TestCase):
    """The backbone starts from a classifier or a detector checkpoint when asked."""

    def setUp(self) -> None:
        """Build a reference backbone and a directory for the checkpoint files."""
        torch.manual_seed(0)
        self.reference = CSPNeXt()
        self.prefixed = OrderedDict(
            (f"backbone.{key}", value) for key, value in self.reference.state_dict().items()
        )
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        """Remove the temporary files."""
        self.directory.cleanup()

    def assert_loaded(self, backbone: CSPNeXt) -> None:
        """Check that every tensor of the backbone equals the reference's."""
        for key, value in self.reference.state_dict().items():
            torch.testing.assert_close(backbone.state_dict()[key], value)

    def test_classifier_checkpoint_loads_the_backbone_and_drops_the_head(self) -> None:
        """
        Input: a checkpoint with ``backbone.`` and ``head.`` keys under ``state_dict``.
        Expected: the backbone tensors load, the classification head is ignored.
        Check: all tensors match the reference.
        """
        state_dict = OrderedDict(self.prefixed)
        state_dict["head.fc.weight"] = torch.zeros(1000, 512)
        path = self.root / "classifier.pth"
        torch.save({"meta": {"epoch": 600}, "state_dict": state_dict}, path)

        self.assert_loaded(CSPNeXt(weights=path))

    def test_detector_checkpoint_with_framework_objects_loads_the_backbone(self) -> None:
        """
        Input: a detector checkpoint whose logging state a weights-only load refuses.
        Expected: the backbone tensors load, the neck, the head and the logging state are
            dropped without being rebuilt.
        Check: all tensors match the reference and the loader returns backbone keys only.
        """
        state_dict = OrderedDict(self.prefixed)
        state_dict["neck.reduce_layers.0.conv.weight"] = torch.zeros(4, 4, 1, 1)
        state_dict["bbox_head.rtm_cls.0.weight"] = torch.zeros(4, 4, 1, 1)
        path = self.root / "detector.pth"
        torch.save({"message_hub": {"log": TrainingLog()}, "state_dict": state_dict}, path)
        with self.assertRaises(pickle.UnpicklingError):
            torch.load(path, map_location="cpu", weights_only=True)

        self.assert_loaded(CSPNeXt(weights=path))
        self.assertEqual(set(load_backbone_state_dict(path)), set(self.reference.state_dict()))

    def test_plain_state_dict_loads(self) -> None:
        """
        Input: the state dict of a backbone, without a prefix or a wrapper.
        Expected: loaded as it is.
        Check: all tensors match the reference.
        """
        path = self.root / "backbone.pth"
        torch.save(self.reference.state_dict(), path)

        self.assert_loaded(CSPNeXt(weights=path))

    def test_other_input_width_keeps_its_own_stem(self) -> None:
        """
        Input: a three-channel checkpoint into a one-channel backbone.
        Expected: the stages load but the stem convolution is skipped.
        Check: a stage weight matches and the stem keeps one input channel.
        """
        path = self.root / "backbone.pth"
        torch.save(self.reference.state_dict(), path)

        backbone = CSPNeXt(in_channels=1, weights=path)

        torch.testing.assert_close(
            backbone.stage2[0].conv.weight, self.reference.stage2[0].conv.weight
        )
        self.assertEqual(backbone.stem[0].conv.weight.shape[1], 1)

    def test_truncated_checkpoint_is_refused(self) -> None:
        """
        Input: a checkpoint without the last stage.
        Expected: refused instead of leaving that stage at its random draw.
        Check: ValueError naming the file and a missing key.
        """
        truncated = {key: value for key, value in self.prefixed.items() if ".stage4." not in key}
        path = self.root / "truncated.pth"
        torch.save({"state_dict": truncated}, path)

        with self.assertRaisesRegex(ValueError, "(?s)truncated.pth.*stage4.0.conv.weight"):
            CSPNeXt(weights=path)

    def test_checkpoint_of_another_width_is_refused(self) -> None:
        """
        Input: the checkpoint of the S backbone into a tiny backbone.
        Expected: refused, the widths differ.
        Check: ValueError.
        """
        path = self.root / "backbone.pth"
        torch.save(self.reference.state_dict(), path)

        with self.assertRaises(ValueError):
            CSPNeXt(deepen_factor=0.167, widen_factor=0.375, weights=path)

    def test_file_without_a_state_dict_is_refused(self) -> None:
        """
        Input: a file holding a list.
        Expected: refused with the file name.
        Check: ValueError naming the file.
        """
        path = self.root / "list.pth"
        torch.save([1, 2, 3], path)

        with self.assertRaisesRegex(ValueError, "list.pth"):
            CSPNeXt(weights=path)

    def test_unknown_weight_name_is_refused_with_the_known_names(self) -> None:
        """
        Input: a name that is neither a file nor a published checkpoint.
        Expected: refused with the available names, nothing is downloaded.
        Check: ValueError naming both published checkpoints.
        """
        with self.assertRaisesRegex(ValueError, "(?s)cspnext_s_imagenet_600e.*rtmdet_s_coco_300e"):
            CSPNeXt(weights="NOT_A_CHECKPOINT")


if __name__ == "__main__":
    unittest.main()
