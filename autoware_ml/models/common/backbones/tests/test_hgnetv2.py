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

"""Unit tests for the HGNetV2 backbone: the feature levels and their widths, the projection
blocks, the parameter names shared with timm and the loading of a timm state dict."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from autoware_ml.models.common.backbones.hgnetv2 import HGNetV2


class TestHGNetV2(unittest.TestCase):
    """Unit tests for the feature maps of the HGNetV2 backbone."""

    def setUp(self) -> None:
        """Set up a two-image batch of 128x128 inputs."""
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 128, 128)

    def test_variants_return_strides_8_16_32(self) -> None:
        """Test that B0 and B1 give 256, 512 and 1024 channels at strides 8, 16 and 32."""
        for variant in ("b0", "b1"):
            with self.subTest(variant=variant):
                backbone = HGNetV2(variant).eval()

                with torch.no_grad():
                    features = backbone(self.images)

                self.assertEqual(backbone.out_channels, (256, 512, 1024))
                self.assertEqual(
                    [tuple(f.shape) for f in features],
                    [(2, 256, 16, 16), (2, 512, 8, 8), (2, 1024, 4, 4)],
                )

    def test_four_levels_add_the_stride_4_map(self) -> None:
        """Test that ``out_indices=[0, 1, 2, 3]`` adds the 64 channel map at stride 4."""
        backbone = HGNetV2("b0", out_indices=[0, 1, 2, 3]).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_channels, (64, 256, 512, 1024))
        self.assertEqual([f.shape[2] for f in features], [32, 16, 8, 4])

    def test_variants_have_the_parameter_count_of_the_reference(self) -> None:
        """Test that the stem and the stages hold the parameters of the timm networks."""
        for variant, expected in (("b0", 1_850_396), ("b1", 2_197_004)):
            with self.subTest(variant=variant):
                backbone = HGNetV2(variant)

                num_parameters = sum(parameter.numel() for parameter in backbone.parameters())

                self.assertEqual(num_parameters, expected)

    def test_odd_feature_sizes_keep_the_strides(self) -> None:
        """Test that an input of 96x160 pixels gives maps of an eighth of its size and below."""
        backbone = HGNetV2("b0").eval()

        with torch.no_grad():
            features = backbone(torch.randn(1, 3, 96, 160))

        self.assertEqual([tuple(f.shape[2:]) for f in features], [(12, 20), (6, 10), (3, 5)])

    def test_projection_sets_the_output_widths(self) -> None:
        """Test that the projection blocks give the requested widths at the same strides."""
        backbone = HGNetV2("b0", projection_channels=[128, 256, 512]).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_channels, (128, 256, 512))
        self.assertEqual(
            [tuple(f.shape) for f in features],
            [(2, 128, 16, 16), (2, 256, 8, 8), (2, 512, 4, 4)],
        )

    def test_projection_leaves_the_stages_unchanged(self) -> None:
        """Test that the stage parameters keep their names when the projections are added."""
        plain = HGNetV2("b0")
        projected = HGNetV2("b0", projection_channels=[128, 256, 512])

        added = set(projected.state_dict()) - set(plain.state_dict())

        self.assertTrue(set(plain.state_dict()) <= set(projected.state_dict()))
        self.assertTrue(all(key.startswith("projections.") for key in added))

    def test_arguments_are_validated(self) -> None:
        """Test that unknown variants, bad stage selections and projection counts are refused."""
        with self.assertRaises(ValueError):
            HGNetV2("b9")
        for out_indices in ([], [4], [2, 1], [1, 1]):
            with self.subTest(out_indices=out_indices), self.assertRaises(ValueError):
                HGNetV2("b0", out_indices=out_indices)
        with self.assertRaises(ValueError):
            HGNetV2("b0", projection_channels=[128, 256])


class TestHGNetV2Weights(unittest.TestCase):
    """The backbone starts from a timm state dict when asked."""

    def setUp(self) -> None:
        """Save a state dict in the timm layout, classification head included, to a file."""
        torch.manual_seed(0)
        self.reference = HGNetV2("b0")
        state_dict = dict(self.reference.state_dict())
        state_dict["head.fc.weight"] = torch.zeros(1000, 2048)
        state_dict["head.fc.bias"] = torch.zeros(1000)
        self.state_dict = state_dict
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "hgnetv2_b0.bin"
        torch.save(state_dict, self.path)

    def tearDown(self) -> None:
        """Remove the temporary file."""
        self.directory.cleanup()

    def test_parameter_names_follow_timm(self) -> None:
        """Test that the stem, plain stage, light stage and downsampling names are those of timm."""
        keys = set(self.reference.state_dict())

        for key in (
            "stem.stem1.conv.weight",
            "stem.stem2a.lab.scale",
            "stages.0.blocks.0.layers.0.conv.weight",
            "stages.0.blocks.0.aggregation.1.lab.bias",
            "stages.1.downsample.bn.running_mean",
            "stages.2.blocks.1.layers.2.conv2.lab.scale",
            "stages.3.blocks.0.layers.0.conv1.conv.weight",
        ):
            self.assertIn(key, keys)
        self.assertNotIn("stages.0.downsample.conv.weight", keys)
        self.assertNotIn("stages.2.blocks.0.layers.0.conv1.lab.scale", keys)
        self.assertEqual(len(keys), 312)

    def test_state_dict_file_loads_the_stages_and_drops_the_head(self) -> None:
        """
        Input: a timm layout state dict file with its classification head.
        Expected: every stem and stage tensor equals the file's, the head is ignored and the
            projection blocks are left alone.
        Check: all tensors of the reference match and the outputs agree.
        """
        backbone = HGNetV2("b0", weights=self.path).eval()
        projected = HGNetV2("b0", projection_channels=[128, 256, 512], weights=self.path)

        for key, value in self.reference.state_dict().items():
            torch.testing.assert_close(backbone.state_dict()[key], value)
            torch.testing.assert_close(projected.state_dict()[key], value)
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            for own, expected in zip(backbone(images), self.reference.eval()(images)):
                torch.testing.assert_close(own, expected)

    def test_other_input_width_keeps_its_own_stem(self) -> None:
        """
        Input: the same file into a one-channel backbone.
        Expected: the stages load but the three-channel stem convolution is skipped.
        Check: a stage weight matches and the stem keeps one input channel.
        """
        backbone = HGNetV2("b0", in_channels=1, weights=self.path)

        torch.testing.assert_close(
            backbone.stages[1].blocks[0].layers[0].conv.weight,
            self.reference.stages[1].blocks[0].layers[0].conv.weight,
        )
        self.assertEqual(backbone.stem.stem1.conv.weight.shape[1], 1)

    def test_truncated_state_dict_is_refused(self) -> None:
        """
        Input: the state dict without its last stage, saved to a file.
        Expected: refused instead of leaving that stage at its random draw.
        Check: ValueError naming the file and a missing key.
        """
        truncated = {
            key: value for key, value in self.state_dict.items() if not key.startswith("stages.3.")
        }
        path = Path(self.directory.name) / "truncated.bin"
        torch.save(truncated, path)

        with self.assertRaisesRegex(ValueError, "(?s)truncated.bin.*stages.3.downsample"):
            HGNetV2("b0", weights=path)

    def test_state_dict_of_another_variant_is_refused(self) -> None:
        """
        Input: the B0 file into a B1 backbone.
        Expected: refused, the widths differ.
        Check: ValueError naming the variant.
        """
        with self.assertRaisesRegex(ValueError, "HGNetV2-B1"):
            HGNetV2("b1", weights=self.path)

    def test_unknown_weight_set_is_refused_with_the_known_names(self) -> None:
        """
        Input: a name that is neither a file nor a weight set.
        Expected: refused with the available names, nothing is downloaded.
        Check: ValueError naming ssld_stage2_ft_in1k.
        """
        with self.assertRaisesRegex(ValueError, "ssld_stage2_ft_in1k"):
            HGNetV2("b0", weights="NOT_A_WEIGHT_SET")


if __name__ == "__main__":
    unittest.main()
