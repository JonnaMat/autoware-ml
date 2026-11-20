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

"""Unit tests for the YOLOX detector with a fourth level at stride 4: the priors, the loss,
the decoding and the deployed graph."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
import torch

from autoware_ml.models.common.backbones.csp_darknet import CSPDarknet
from autoware_ml.models.common.necks.yolox_pafpn import YOLOXPAFPN
from autoware_ml.models.detection2d.heads.yolox import YOLOXHead
from autoware_ml.models.detection2d.tests.yolox_fixtures import build_batch_inputs
from autoware_ml.models.detection2d.yolox import YOLOXDetector
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


def build_four_level_detector(**kwargs: object) -> YOLOXDetector:
    """Build a narrow YOLOX detector predicting at strides 4, 8, 16 and 32."""
    backbone = CSPDarknet(deepen_factor=0.33, widen_factor=0.125, out_indices=[0, 1, 2, 3])
    return YOLOXDetector(
        backbone=backbone,
        neck=YOLOXPAFPN(in_channels=backbone.out_channels),
        head=YOLOXHead(
            num_classes=2,
            in_channels=backbone.out_channels,
            feat_channels=16,
            strides=[4, 8, 16, 32],
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        **kwargs,
    )


class TestFourLevelYOLOXDetector(unittest.TestCase):
    """Unit tests for the detector with a stride 4 level."""

    def setUp(self) -> None:
        """Build the detector and a batch of two 64x64 images, one of them without a box."""
        torch.manual_seed(0)
        self.detector = build_four_level_detector()
        self.batch = build_batch_inputs(
            images=torch.rand(2, 3, 64, 64) * 255.0,
            gt_boxes=[
                torch.tensor([[8.0, 8.0, 14.0, 20.0], [30.0, 30.0, 60.0, 50.0]]),
                torch.zeros(0, 4),
            ],
            gt_labels=[torch.tensor([0, 1]), torch.zeros(0, dtype=torch.long)],
        )

    def test_priors_cover_the_four_levels_finest_first(self) -> None:
        """Test that a 64x64 image has 16² + 8² + 4² + 2² priors, the stride 4 ones first."""
        outputs = self.detector(self.batch).detection2d()

        self.assertEqual(outputs.predictions.shape, (2, 256 + 64 + 16 + 4, 7))
        self.assertEqual(outputs.priors[:256, 2].unique().tolist(), [4.0])
        self.assertEqual(outputs.priors[256:320, 2].unique().tolist(), [8.0])
        self.assertEqual(outputs.priors[-4:, 2].unique().tolist(), [32.0])

    def test_input_of_960_pixels_has_76500_priors(self) -> None:
        """Test that the levels of a 960x960 input hold 240² + 120² + 60² + 30² priors."""
        priors = self.detector.head.build_priors(
            [(240, 240), (120, 120), (60, 60), (30, 30)], torch.device("cpu")
        )

        self.assertEqual(priors.shape, (76_500, 3))

    def test_training_step_reaches_the_modules_of_the_fourth_level(self) -> None:
        """Test that the loss is finite and its gradient reaches the stride 4 modules."""
        self.detector.train()

        loss = self.detector.training_step(self.batch, 0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        network = self.detector.network
        for parameter in (
            network.neck.C3_p2.conv3.conv.weight,
            network.neck.C3_n2.conv3.conv.weight,
            network.head.obj_preds[0].weight,
            network.head.obj_preds[3].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_small_box_is_assigned_to_the_stride_4_priors_that_predict_it(self) -> None:
        """Test that a 4x4 pixel box predicted by the stride 4 level only is assigned there.

        Check: the foreground priors are stride 4 priors within 1.5 strides of the box center.
        """
        head = self.detector.head
        priors = head.build_priors([(16, 16), (8, 8), (4, 4), (2, 2)], torch.device("cpu"))
        box = torch.tensor([[22.0, 22.0, 4.0, 4.0]])
        pred_boxes = torch.tensor([[50.0, 50.0, 4.0, 4.0]]).repeat(priors.shape[0], 1)
        pred_boxes[:256] = box
        assignment = head.assigner.assign(
            pred_boxes,
            torch.full((priors.shape[0], 1), 4.0),
            torch.full((priors.shape[0], 2), 4.0),
            priors,
            box,
            torch.tensor([0]),
        )

        foreground = priors[assignment.foreground_mask]
        self.assertGreater(foreground.shape[0], 0)
        self.assertEqual(foreground[:, 2].unique().tolist(), [4.0])
        centers = (foreground[:, :2] + 0.5) * 4.0
        self.assertTrue(((centers - 22.0).abs() < 6.0).all())

    def test_detections_are_decoded_from_every_level(self) -> None:
        """Test that the eval output holds detections in the image for both images."""
        detector = build_four_level_detector(score_threshold=0.0, max_detections=500).eval()

        predictions = detector.decode_outputs(self.batch, detector(self.batch))

        for detections in predictions.detection2d_predictions:
            self.assertGreater(detections.bboxes_2d.shape[0], 0)
            self.assertTrue(torch.isfinite(detections.bboxes_2d).all())

    def test_exported_graph_matches_the_detector_in_onnx_runtime(self) -> None:
        """Test that the deployed graph outputs one row per prior of the four levels."""
        detector = build_four_level_detector(export_background_class=True).eval()
        spec = detector.build_export_spec(self.batch)
        images = spec.args[0][:1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.onnx"
            torch.onnx.export(
                spec.module,
                (images,),
                str(path),
                input_names=spec.input_names,
                output_names=spec.output_names,
                opset_version=17,
                dynamo=False,
            )
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            (result,) = session.run(None, {"images": images.numpy()})
        with torch.no_grad():
            expected = spec.module(images).numpy()

        self.assertEqual(result.shape, (1, 340, 8))
        np.testing.assert_allclose(result, expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
