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

"""Unit tests for the ONNX backed YOLOX detector used to score deployed graphs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from autoware_ml.models.detection2d.tests.yolox_fixtures import build_batch_inputs, build_detector
from autoware_ml.models.detection2d.yolox_onnx import ONNXYOLOXDetector
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


class TestONNXYOLOXDetector(unittest.TestCase):
    """Unit tests for the ONNX backed detector."""

    def setUp(self) -> None:
        """Export a narrow detector with a background class score to a temporary ONNX file."""
        torch.manual_seed(0)
        self.detector = build_detector(export_background_class=True, score_threshold=0.0).eval()
        self.batch = build_batch_inputs(
            images=torch.rand(2, 3, 64, 64) * 255.0,
            gt_boxes=[torch.tensor([[8.0, 8.0, 40.0, 40.0]]), torch.zeros(0, 4)],
            gt_labels=[torch.tensor([0]), torch.zeros(0, dtype=torch.long)],
        )
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.onnx_path = Path(self.directory.name) / "detector.onnx"
        spec = self.detector.build_export_spec(self.batch)
        torch.onnx.export(
            spec.module,
            (spec.args[0][:1],),
            str(self.onnx_path),
            input_names=spec.input_names,
            output_names=spec.output_names,
            opset_version=17,
            dynamo=False,
        )

    def build_onnx_detector(self, **kwargs: object) -> ONNXYOLOXDetector:
        """Build the ONNX backed detector over the exported file."""
        return ONNXYOLOXDetector(
            onnx_path=self.onnx_path,
            num_classes=2,
            providers=["CPUExecutionProvider"],
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            score_threshold=0.0,
            **kwargs,
        ).eval()

    def test_graph_scores_like_the_detector_it_was_exported_from(self) -> None:
        """Test that the ONNX backed detector reproduces the detector on a batch of two images.

        Check: with the classes read behind the background entry, the box regression and the
        probabilities equal those of the detector, and the eval output handed to the metrics
        holds as many detections and the same ground truth.
        """
        onnx_detector = self.build_onnx_detector(class_indices=[1, 2])

        with torch.no_grad():
            expected_outputs = self.detector(self.batch)
            outputs = onnx_detector(self.batch)

        expected = expected_outputs.detection2d().predictions
        actual = outputs.detection2d().predictions
        torch.testing.assert_close(actual[..., :4], expected[..., :4], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            actual[..., 4:].sigmoid(), expected[..., 4:].sigmoid(), atol=1e-4, rtol=1e-4
        )
        expected_frames = self.detector.build_eval_output(
            self.batch, expected_outputs
        ).detection2d.frames
        frames = onnx_detector.build_eval_output(self.batch, outputs).detection2d.frames
        for frame, expected_frame in zip(frames, expected_frames, strict=True):
            self.assertEqual(
                frame.predictions.bboxes_2d.shape, expected_frame.predictions.bboxes_2d.shape
            )
            torch.testing.assert_close(frame.gt_boxes, expected_frame.gt_boxes)

    def test_default_class_indices_read_the_first_outputs(self) -> None:
        """Test that without class indices the first class outputs are read: here the background."""
        onnx_detector = self.build_onnx_detector()

        with torch.no_grad():
            predictions = onnx_detector(self.batch).detection2d().predictions

        self.assertEqual(onnx_detector.class_indices, [0, 1])
        # The background score of the graph is zero, a probability the wrapper clips.
        self.assertLess(float(predictions[..., 5].sigmoid().max()), 1e-6)

    def test_mismatching_graphs_and_inputs_are_refused(self) -> None:
        """Test that a wrong class count, image size or stride set is refused with an error."""
        with self.assertRaises(ValueError):
            self.build_onnx_detector(class_indices=[1])
        larger = build_batch_inputs(
            images=torch.rand(1, 3, 96, 96),
            gt_boxes=[torch.zeros(0, 4)],
            gt_labels=[torch.zeros(0, dtype=torch.long)],
        )
        with self.assertRaises(ValueError):
            self.build_onnx_detector()(larger)
        with self.assertRaises(ValueError):
            self.build_onnx_detector(strides=[8, 16])(self.batch)
