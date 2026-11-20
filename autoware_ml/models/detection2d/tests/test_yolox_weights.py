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

"""Unit tests for the conversion of YOLOX weights: reference checkpoints and folded ONNX files."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from autoware_ml.models.common.necks.yolox_pafpn import YOLOXPAFPN
from autoware_ml.models.detection2d.tests.yolox_fixtures import build_detector
from autoware_ml.models.detection2d.yolox import YOLOXExportModule
from autoware_ml.models.detection2d.yolox_weights import (
    convert_folded_onnx,
    convert_reference_checkpoint,
    rename_reference_key,
)
from autoware_ml.scripts.tools.yolox.convert_weights import convert
from autoware_ml.utils.checkpoints import apply_matching_weights


def _reference_state_dict(detector: torch.nn.Module) -> dict[str, torch.Tensor]:
    """State dict of the network of a detector under the key names of the reference."""
    renamed = {}
    for key, value in detector.network.state_dict().items():
        stage, rest = key.split(".", 1)
        prefix = {"backbone": "backbone.backbone.", "neck": "backbone.", "head": "head."}[stage]
        renamed[prefix + rest] = value.clone()
    return renamed


class _ReferencePAFPN(YOLOXPAFPN):
    """Neck laid out as in the reference implementation, which owns and calls its backbone."""

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the nested backbone, then fuse its feature maps."""
        return super().forward(self.backbone(images))


class _ReferenceModel(torch.nn.Module):
    """Detector under the module names of the reference: ``backbone.backbone`` and ``head``."""

    def __init__(self, network: torch.nn.Module) -> None:
        """Rearrange a copy of the network of a detector."""
        super().__init__()
        network = copy.deepcopy(network)
        self.backbone = network.neck
        self.backbone.__class__ = _ReferencePAFPN
        self.backbone.backbone = network.backbone
        self.head = network.head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict the output tensor the reference exports without its decoding."""
        return self.head.export_outputs(self.head(self.backbone(images)))


class TestReferenceCheckpoint(unittest.TestCase):
    """Unit tests for the conversion of a checkpoint of the reference implementation."""

    def test_keys_are_renamed_by_stage(self) -> None:
        """Test that the nested backbone, the neck and the head keys land in their stages."""
        self.assertEqual(
            rename_reference_key("backbone.backbone.dark2.0.conv.weight"),
            "backbone.dark2.0.conv.weight",
        )
        self.assertEqual(
            rename_reference_key("backbone.C3_p4.conv1.bn.bias"), "neck.C3_p4.conv1.bn.bias"
        )
        self.assertEqual(rename_reference_key("head.cls_preds.0.bias"), "head.cls_preds.0.bias")
        with self.assertRaises(ValueError):
            rename_reference_key("neck.unknown.weight")

    def test_converted_checkpoint_covers_the_detector(self) -> None:
        """Test that a converted checkpoint loads into the network and its average without a gap.

        Check: no key is missing or unexpected under a strict load, and the weights and the
        average both hold the source weights.
        """
        torch.manual_seed(0)
        source = build_detector()
        target = build_detector()

        state_dict = convert_reference_checkpoint({"model": _reference_state_dict(source)})
        target.load_state_dict(state_dict, strict=True)

        for name, value in source.network.state_dict().items():
            torch.testing.assert_close(target.network.state_dict()[name], value)
            torch.testing.assert_close(target.ema.module.state_dict()[name], value)

    def test_class_outputs_of_another_class_count_are_the_only_skipped_tensors(self) -> None:
        """Test that weights of another class count load except for the class outputs.

        Setup: a converted checkpoint of an 80 class detector applied to a 2 class detector
        through the weight loading of the CLI, with mismatched shapes skipped.
        Check: nothing is missing or unexpected, and the skipped tensors are exactly the
        class output weights and biases of the three levels, of the weights and the average.
        """
        source = build_detector(num_classes=80)
        target = build_detector(num_classes=2)
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.pth"
            converted_path = Path(directory) / "converted.ckpt"
            torch.save({"model": _reference_state_dict(source)}, reference_path)
            convert(reference_path, converted_path)

            (report,) = apply_matching_weights(target, converted_path, skip_mismatched_shapes=True)

        expected_skipped = {
            f"{prefix}head.cls_preds.{level}.{name}"
            for prefix in ("network.", "ema.module.")
            for level in range(3)
            for name in ("weight", "bias")
        }
        self.assertEqual(set(report.shape_mismatched_keys), expected_skipped)
        self.assertEqual(report.unexpected_keys, ())
        self.assertEqual(set(report.not_loaded_model_keys), expected_skipped)


class TestFoldedONNX(unittest.TestCase):
    """Unit tests for the conversion of an ONNX file with folded batch normalizations."""

    def test_converted_weights_reproduce_the_exported_graph(self) -> None:
        """Test that a detector loaded from a folded ONNX export predicts what the source does.

        Setup: a detector with non-trivial normalization statistics exported the way the
        reference implementation exports, which folds every batch normalization.
        Check: the converted state loads strictly and the outputs of both detectors agree.
        """
        torch.manual_seed(0)
        source = build_detector(use_ema=False)
        source.train()
        with torch.no_grad():
            for _ in range(3):
                source.network(torch.rand(2, 3, 64, 64) * 255.0)
        source.eval()
        images = torch.rand(1, 3, 64, 64) * 255.0
        with tempfile.TemporaryDirectory() as directory:
            onnx_path = Path(directory) / "reference.onnx"
            torch.onnx.export(
                _ReferenceModel(source.network).eval(),
                (images,),
                str(onnx_path),
                opset_version=11,
                dynamo=False,
            )
            state_dict = convert_folded_onnx(onnx_path, with_ema=False)

        target = build_detector(use_ema=False).eval()
        target.load_state_dict(state_dict, strict=True)
        with torch.no_grad():
            expected = YOLOXExportModule(source.network)(images)
            actual = YOLOXExportModule(target.network)(images)
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
