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

"""Unit tests for the plain vision transformer backbone: the feature levels, the windowed
attention, the loading of pretrained weights of another patch and input size, and the layers
of the layer-wise learning rate decay."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest

import onnx
import timm
import torch

from autoware_ml.models.common.backbones.vit import SimpleFeaturePyramid, ViTSimplePyramid

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
REGISTER_MODEL_NAME = "vit_small_patch14_reg4_dinov2.lvd142m"


def build_backbone(**kwargs: object) -> ViTSimplePyramid:
    """Build a randomly initialized backbone for 64x64 inputs, a 4x4 patch grid."""
    arguments = {"model_name": MODEL_NAME, "image_size": [64, 64], "pretrained": False}
    return ViTSimplePyramid(**{**arguments, **kwargs}).eval()


class TestSimpleFeaturePyramid(unittest.TestCase):
    """Unit tests for the simple feature pyramid."""

    def test_levels_are_at_twice_once_and_half_the_resolution(self) -> None:
        """Test that the three maps have the configured widths at strides 8, 16 and 32."""
        pyramid = SimpleFeaturePyramid(in_channels=32, out_channels=[8, 16, 24]).eval()

        with torch.no_grad():
            features = pyramid(torch.randn(2, 32, 6, 4))

        self.assertEqual(pyramid.out_channels, (8, 16, 24))
        self.assertEqual(
            [tuple(f.shape) for f in features], [(2, 8, 12, 8), (2, 16, 6, 4), (2, 24, 3, 2)]
        )

    def test_number_of_levels_is_validated(self) -> None:
        """Test that widths of other than three levels are refused."""
        with self.assertRaises(ValueError):
            SimpleFeaturePyramid(in_channels=32, out_channels=[8, 16])


class TestViTSimplePyramid(unittest.TestCase):
    """Unit tests for the vision transformer backbone."""

    def setUp(self) -> None:
        """Set up a two-image batch of 64x64 inputs."""
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 64, 64)

    def test_returns_strides_8_16_32(self) -> None:
        """Test that the backbone returns 128, 256 and 512 channels at strides 8, 16 and 32."""
        backbone = build_backbone(window_size=2)

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_channels, (128, 256, 512))
        self.assertEqual(backbone.out_strides, (8, 16, 32))
        self.assertEqual(
            [tuple(f.shape) for f in features], [(2, 128, 8, 8), (2, 256, 4, 4), (2, 512, 2, 2)]
        )

    def test_global_attention_reproduces_the_timm_network(self) -> None:
        """Test that without windows the token map is the patch tokens of the timm network."""
        backbone = build_backbone()

        with torch.no_grad():
            tokens = backbone.forward_tokens(self.images)
            reference = backbone.vit.forward_features(self.images)[:, 1:]

        self.assertEqual(backbone.global_block_indices, tuple(range(12)))
        torch.testing.assert_close(tokens.flatten(2).transpose(1, 2), reference)

    def test_one_window_over_the_grid_is_global_attention(self) -> None:
        """Test that a window as large as the patch grid changes nothing."""
        for model_name in (MODEL_NAME, REGISTER_MODEL_NAME):
            with self.subTest(model_name=model_name):
                torch.manual_seed(0)
                windowed = build_backbone(model_name=model_name, window_size=4)
                plain = build_backbone(model_name=model_name)
                plain.load_state_dict(windowed.state_dict())

                with torch.no_grad():
                    torch.testing.assert_close(
                        windowed.forward_tokens(self.images),
                        plain.forward_tokens(self.images),
                        atol=1e-5,
                        rtol=1e-5,
                    )

    def test_windows_split_and_join_the_tokens(self) -> None:
        """Test that joining the windows restores the token sequence, prefix tokens included."""
        for model_name, num_prefix_tokens in ((MODEL_NAME, 1), (REGISTER_MODEL_NAME, 5)):
            with self.subTest(model_name=model_name):
                backbone = build_backbone(model_name=model_name, window_size=2)
                tokens = torch.randn(2, num_prefix_tokens + 16, 384)

                windows = backbone._to_windows(tokens)

                self.assertEqual(backbone.num_prefix_tokens, num_prefix_tokens)
                self.assertEqual(tuple(windows.shape), (8, num_prefix_tokens + 4, 384))
                # The second window of the first image holds the top right 2x2 tokens.
                torch.testing.assert_close(
                    windows[1, num_prefix_tokens:],
                    tokens[0, num_prefix_tokens:].reshape(4, 4, 384)[:2, 2:].reshape(4, 384),
                )
                torch.testing.assert_close(backbone._from_windows(windows), tokens)

    def test_windowed_blocks_do_not_mix_windows(self) -> None:
        """Test that without a global block a change of the image stays within its window."""
        backbone = build_backbone(window_size=2, global_block_indices=[])
        changed = self.images.clone()
        changed[:, :, :32, :32] += 1.0

        with torch.no_grad():
            difference = (
                backbone.forward_tokens(changed) - backbone.forward_tokens(self.images)
            ).abs()

        self.assertGreater(difference[:, :, :2, :2].max().item(), 1e-3)
        self.assertEqual(difference[:, :, 2:, :].max().item(), 0.0)
        self.assertEqual(difference[:, :, :, 2:].max().item(), 0.0)

    def test_global_blocks_mix_windows(self) -> None:
        """Test that a global block spreads a change of the image over every window."""
        backbone = build_backbone(window_size=2, global_block_indices=[11])
        changed = self.images.clone()
        changed[:, :, :32, :32] += 1.0

        with torch.no_grad():
            difference = (
                backbone.forward_tokens(changed) - backbone.forward_tokens(self.images)
            ).abs()

        self.assertGreater(difference[:, :, 2:, 2:].max().item(), 1e-8)

    def test_weights_of_another_patch_and_input_size_load(self) -> None:
        """Test that weights of patch 14 at 518x518 load at patch 16 and another input size."""
        torch.manual_seed(0)
        native = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        self.assertEqual(tuple(native.patch_embed.proj.weight.shape), (384, 3, 14, 14))
        self.assertEqual(tuple(native.pos_embed.shape), (1, 1 + 37 * 37, 384))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save(native.state_dict(), path)
            backbone = ViTSimplePyramid(
                model_name=MODEL_NAME, image_size=[64, 96], pretrained=False
            )
            loaded = timm.create_model(
                MODEL_NAME,
                pretrained=True,
                pretrained_cfg_overlay={"file": str(path)},
                img_size=backbone.image_size,
                patch_size=backbone.patch_size,
                num_classes=0,
            )
            report = backbone.vit.load_state_dict(loaded.state_dict())

        self.assertEqual(report.missing_keys, [])
        self.assertEqual(report.unexpected_keys, [])
        self.assertEqual(tuple(backbone.vit.patch_embed.proj.weight.shape), (384, 3, 16, 16))
        self.assertEqual(tuple(backbone.vit.pos_embed.shape), (1, 1 + 4 * 6, 384))
        torch.testing.assert_close(
            backbone.vit.blocks[5].mlp.fc1.weight, native.blocks[5].mlp.fc1.weight
        )
        with torch.no_grad():
            features = backbone.eval()(torch.randn(1, 3, 64, 96))
        self.assertEqual([tuple(f.shape[2:]) for f in features], [(8, 12), (4, 6), (2, 3)])

    def test_export_has_constant_shapes_and_explicit_attention(self) -> None:
        """Test that the ONNX graph of one image has no shape arithmetic and one node per norm."""
        backbone = build_backbone(window_size=2)
        buffer = io.BytesIO()

        torch.onnx.export(
            backbone, (self.images[:1],), buffer, opset_version=17, dynamo=False, verbose=False
        )

        graph = onnx.load_from_string(buffer.getvalue()).graph
        consumed = {name for node in graph.node for name in node.input}
        op_types = {
            node.op_type for node in graph.node if any(name in consumed for name in node.output)
        }
        self.assertLessEqual({"LayerNormalization", "Softmax", "MatMul", "ConvTranspose"}, op_types)
        self.assertEqual(
            op_types & {"Shape", "Expand", "Where", "Cast", "Sqrt", "Gather", "Einsum"}, set()
        )
        # The attention is switched to its explicit form for the export only.
        self.assertTrue(all(block.attn.fused_attn for block in backbone.vit.blocks))

    def test_images_of_another_size_are_refused(self) -> None:
        """Test that the backbone refuses images it was not built for."""
        backbone = build_backbone()

        with self.assertRaises(ValueError):
            backbone(torch.randn(1, 3, 96, 64))

    def test_sizes_are_validated(self) -> None:
        """Test that an input, a window or a global block that does not fit is refused."""
        for arguments in (
            {"image_size": [64, 80]},
            {"window_size": 3},
            {"window_size": 0},
            {"window_size": 2, "global_block_indices": [12]},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_backbone(**arguments)

    def test_every_transformer_parameter_has_a_layer(self) -> None:
        """Test that the layers run from the embeddings over the blocks to the final norm."""
        backbone = build_backbone()

        layer_ids, num_layers = backbone.lr_layer_ids()

        names = {name for name, _ in backbone.named_parameters()}
        self.assertEqual(num_layers, 13)
        self.assertEqual({name for name in names if name.startswith("vit.")}, set(layer_ids))
        self.assertEqual(layer_ids["vit.pos_embed"], 0)
        self.assertEqual(layer_ids["vit.cls_token"], 0)
        self.assertEqual(layer_ids["vit.patch_embed.proj.weight"], 0)
        self.assertEqual(layer_ids["vit.blocks.0.attn.qkv.weight"], 1)
        self.assertEqual(layer_ids["vit.blocks.11.mlp.fc2.bias"], 12)
        self.assertEqual(layer_ids["vit.norm.weight"], 13)
        self.assertLessEqual({"vit.pos_embed", "vit.cls_token"}, backbone.no_weight_decay())
