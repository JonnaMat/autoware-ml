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

"""Unit tests for the YOLOX detector: the training and evaluation paths, the averaged weights,
the optimizer groups, the mapping of detections back onto the raw image and the export."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

from autoware_ml.dataclasses.models.detection2d.head_outputs import YOLOXHeadOutputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.models.common.backbones.resnet import ResNet18MultiScale
from autoware_ml.models.common.necks.yolox_pafpn import YOLOXPAFPN
from autoware_ml.models.detection2d.heads.yolox import YOLOXHead
from autoware_ml.models.detection2d.tests.yolox_fixtures import build_batch_inputs, build_detector
from autoware_ml.models.detection2d.yolox import YOLOXDetector
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec


def _run_exported_graph(spec: ExportSpec, images: torch.Tensor) -> tuple[bytes, np.ndarray]:
    """Export a graph the way the deployment does and run it in ONNX Runtime.

    Args:
        spec: Export specification of a detector.
        images: Images the graph is given.

    Returns:
        The serialized graph and its output for the images.
    """
    with tempfile.TemporaryDirectory() as directory:
        onnx_path = Path(directory) / "detector.onnx"
        torch.onnx.export(
            spec.module,
            spec.args,
            str(onnx_path),
            input_names=spec.input_names,
            output_names=spec.output_names,
            opset_version=17,
            dynamo=False,
        )
        graph = onnx.load(str(onnx_path)).graph.SerializeToString(deterministic=True)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return graph, session.run(None, {"images": images.numpy()})[0]


class TestYOLOXDetector(unittest.TestCase):
    """Unit tests for the YOLOX detector."""

    def setUp(self) -> None:
        """Set up a narrow detector and a batch of two 64x64 images, one without boxes."""
        torch.manual_seed(0)
        self.detector = build_detector()
        self.batch = build_batch_inputs(
            images=torch.rand(2, 3, 64, 64) * 255.0,
            gt_boxes=[
                torch.tensor([[8.0, 8.0, 40.0, 40.0], [30.0, 10.0, 50.0, 28.0]]),
                torch.zeros(0, 4),
            ],
            gt_labels=[torch.tensor([0, 1]), torch.zeros(0, dtype=torch.long)],
        )

    def test_training_step_returns_a_finite_loss_with_gradients(self) -> None:
        """Test that a training forward pass gives a finite loss that trains the network only."""
        self.detector.train()

        outputs = self.detector(self.batch)
        metrics = self.detector.compute_metrics(self.batch, outputs)
        metrics["loss"].backward()

        self.assertEqual(tuple(outputs.detection2d().predictions.shape), (2, 84, 7))
        self.assertTrue(torch.isfinite(metrics["loss"]))
        self.assertTrue(all(p.grad is not None for p in self.detector.network.parameters()))
        self.assertTrue(all(p.grad is None for p in self.detector.ema.parameters()))

    def test_images_reach_the_backbone_in_the_configured_channel_order(self) -> None:
        """Test that the RGB images of the batch are flipped to BGR unless RGB is configured."""
        rgb = self.batch.image_data.images[:, 0]

        torch.testing.assert_close(self.detector.prepare_images(self.batch), rgb.flip(1))
        torch.testing.assert_close(
            build_detector(input_channel_order="rgb").prepare_images(self.batch), rgb
        )
        with self.assertRaises(ValueError):
            build_detector(input_channel_order="gbr")

    def test_input_normalization_is_part_of_the_network(self) -> None:
        """Test that the mean and the divisor are applied inside the network.

        Check: a detector normalizing by a mean and a divisor predicts on an image what the
        same weights without normalization predict on the normalized image.
        """
        plain = build_detector(use_ema=False).eval()
        normalizing = build_detector(
            use_ema=False, input_mean=[10.0, 20.0, 30.0], input_std=[2.0, 4.0, 8.0]
        ).eval()
        normalizing.network.load_state_dict(plain.network.state_dict())
        images = torch.rand(1, 3, 64, 64) * 255.0
        mean = torch.tensor([10.0, 20.0, 30.0]).view(1, 3, 1, 1)
        std = torch.tensor([2.0, 4.0, 8.0]).view(1, 3, 1, 1)

        with torch.no_grad():
            expected = plain.network((images - mean) / std)
            actual = normalizing.network(images)

        for expected_level, actual_level in zip(expected, actual, strict=True):
            torch.testing.assert_close(actual_level, expected_level, atol=1e-5, rtol=1e-5)

    def test_average_follows_the_weights_and_predicts_in_eval_mode(self) -> None:
        """Test the life cycle of the averaged weights.

        Check: the fit start copies the weights into the average, an update moves the average
        towards changed weights by the ramped decay, the average predicts in eval mode and
        the weights in training mode, and the average is part of the state dict.
        """
        self.detector.on_fit_start()
        for averaged, trained in zip(
            self.detector.ema.module.parameters(), self.detector.network.parameters(), strict=True
        ):
            torch.testing.assert_close(averaged, trained)

        with torch.no_grad():
            for parameter in self.detector.network.parameters():
                parameter.add_(1.0)
        self.detector.on_train_batch_end(None, None, 0)

        # The first update follows the weights almost fully: the decay ramps up from 0.
        decay = 0.9998 * (1.0 - math.exp(-1 / 2000))
        trained = next(self.detector.network.parameters())
        averaged = next(self.detector.ema.module.parameters())
        self.assertAlmostEqual((trained - averaged).abs().mean().item(), decay, places=5)
        self.assertEqual(int(self.detector.ema.num_updates), 1)
        self.assertIs(self.detector.train().inference_network(), self.detector.network)
        self.assertIs(self.detector.eval().inference_network(), self.detector.ema.module)
        self.assertFalse(self.detector.train().ema.module.training)
        self.assertIn("ema.module.backbone.stem.conv.conv.weight", self.detector.state_dict())
        self.assertIn("ema.num_updates", self.detector.state_dict())

    def test_detector_without_average_predicts_with_its_weights(self) -> None:
        """Test that a detector without an average has none in its state and evaluates its weights."""
        detector = build_detector(use_ema=False).eval()

        self.assertIsNone(detector.ema)
        self.assertIs(detector.inference_network(), detector.network)
        self.assertFalse(any(key.startswith("ema.") for key in detector.state_dict()))

    def test_optimizer_groups_split_by_weight_decay(self) -> None:
        """Test that biases and normalization weights are not decayed and the average is left out."""
        groups = self.detector.build_optimizer_groups()

        num_trained = sum(1 for _ in self.detector.network.parameters())
        self.assertEqual(len(groups["decay"]) + len(groups["no_decay"]), num_trained)
        self.assertTrue(all(parameter.ndim >= 2 for parameter in groups["decay"]))
        self.assertTrue(all(parameter.ndim == 1 for parameter in groups["no_decay"]))

    def test_strong_augmentation_switch_controls_the_l1_loss(self) -> None:
        """Test that the L1 loss is on exactly while the strong augmentation is off."""
        self.detector.set_strong_augmentation(False)
        self.assertTrue(self.detector.head.use_l1)
        self.detector.set_strong_augmentation(True)
        self.assertFalse(self.detector.head.use_l1)

    def _outputs_with_one_detection(self) -> ModelOutputs:
        """Raw outputs in which one prior of the first image detects a 16 px box of class 1."""
        priors = self.detector.head.build_priors([(8, 8), (4, 4), (2, 2)], torch.device("cpu"))
        predictions = torch.full((2, 84, 7), -12.0)
        predictions[..., :4] = 0.0
        # Prior 10: cell (2, 1) of the stride 8 level, a box centered on (20, 12).
        predictions[0, 10] = torch.tensor([0.5, 0.5, 0.6931472, 0.6931472, 12.0, -12.0, 12.0])
        return ModelOutputs(
            detection2d_head_outputs=YOLOXHeadOutputs(predictions=predictions, priors=priors)
        )

    def test_detections_and_ground_truth_are_mapped_onto_the_raw_image(self) -> None:
        """Test that the eval output is in pixels of the raw image.

        Setup: a batch whose test pipeline shrank the raw images by 0.25, one ground truth
        box and one detection of a 16 px box at (12, 4) of the network input.
        Check: the detection and the ground truth box are four times as large in the eval
        output, and the predict step returns the same detection.
        """
        batch = build_batch_inputs(
            images=torch.rand(2, 3, 64, 64),
            gt_boxes=[torch.tensor([[12.0, 4.0, 28.0, 20.0]]), torch.zeros(0, 4)],
            gt_labels=[torch.tensor([1]), torch.zeros(0, dtype=torch.long)],
            scale=0.25,
        )
        outputs = self._outputs_with_one_detection()

        eval_output = self.detector.build_eval_output(batch, outputs)
        predictions = self.detector.decode_outputs(batch, outputs)

        frames = eval_output.detection2d.frames
        expected = torch.tensor([[48.0, 16.0, 112.0, 80.0]])
        torch.testing.assert_close(frames[0].predictions.bboxes_2d, expected, atol=1e-3, rtol=0)
        torch.testing.assert_close(frames[0].gt_boxes, expected)
        self.assertEqual(frames[0].predictions.labels_2d.tolist(), [1])
        self.assertEqual(frames[0].gt_labels.tolist(), [1])
        self.assertEqual(frames[1].predictions.bboxes_2d.shape[0], 0)
        self.assertEqual(frames[1].gt_boxes.shape[0], 0)
        torch.testing.assert_close(
            predictions.detection2d_predictions[0].bboxes_2d, expected, atol=1e-3, rtol=0
        )

    def test_batch_of_several_cameras_is_refused(self) -> None:
        """Test that a batch with more than one camera per sample is refused."""
        image_data = self.batch.image_data
        two_cameras = self.batch.replace(
            image_data=image_data._replace(images=image_data.images.repeat(1, 2, 1, 1, 1))
        )

        with self.assertRaises(ValueError):
            self.detector(two_cameras)

    def test_resnet_backbone_drops_in(self) -> None:
        """Test that a ResNet-18 backbone trains under the same neck and head."""
        detector = YOLOXDetector(
            backbone=ResNet18MultiScale(in_channels=3, out_indices=[1, 2, 3]),
            neck=YOLOXPAFPN(in_channels=[128, 256, 512]),
            head=YOLOXHead(num_classes=2, in_channels=[128, 256, 512], feat_channels=16),
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            input_channel_order="rgb",
            input_mean=[123.675, 116.28, 103.53],
            input_std=[58.395, 57.12, 57.375],
        ).train()

        metrics = detector.compute_metrics(self.batch, detector(self.batch))

        self.assertTrue(torch.isfinite(metrics["loss"]))


class TestYOLOXExport(unittest.TestCase):
    """Unit tests for the deployed graph of the detector."""

    def setUp(self) -> None:
        """Set up a detector in eval mode whose average differs from its weights."""
        torch.manual_seed(0)
        self.detector = build_detector(export_background_class=True)
        with torch.no_grad():
            for parameter in self.detector.ema.module.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        self.detector.eval()
        self.batch = build_batch_inputs(
            images=torch.rand(1, 3, 64, 64) * 255.0,
            gt_boxes=[torch.tensor([[8.0, 8.0, 40.0, 40.0]])],
            gt_labels=[torch.tensor([0])],
        )

    def test_exported_graph_reads_the_configured_channel_order(self) -> None:
        """Test that a graph deployed for BGR input feeds an RGB backbone the reversed channels.

        Check: the graph of an RGB detector exported for BGR, given the BGR image, equals the
        graph exported in the order of the backbone given the RGB image; the example input of
        the export is the BGR image.
        """
        torch.manual_seed(0)
        detector = build_detector(input_channel_order="rgb", use_ema=False).eval()
        for_runtime = build_detector(
            input_channel_order="rgb", export_input_channel_order="bgr", use_ema=False
        ).eval()
        for_runtime.load_state_dict(detector.state_dict())
        rgb = self.batch.image_data.images[:, 0]

        native_spec = detector.build_export_spec(self.batch)
        runtime_spec = for_runtime.build_export_spec(self.batch)

        torch.testing.assert_close(native_spec.args[0], rgb)
        torch.testing.assert_close(runtime_spec.args[0], rgb.flip(1))
        with torch.no_grad():
            torch.testing.assert_close(
                runtime_spec.module(runtime_spec.args[0]), native_spec.module(native_spec.args[0])
            )
        with self.assertRaises(ValueError):
            build_detector(export_input_channel_order="gbr")

    def test_onnx_graph_of_an_rgb_backbone_takes_the_bgr_image_of_the_runtime(self) -> None:
        """Test the deployed graph of a normalizing RGB backbone against the detector.

        Check: ONNX Runtime, given the BGR image in [0, 255] a YOLOX runtime delivers, returns
        what the detector predicts for the same image served as RGB by the data pipeline.
        """
        torch.manual_seed(0)
        detector = build_detector(
            input_channel_order="rgb",
            input_mean=[123.675, 116.28, 103.53],
            input_std=[58.395, 57.12, 57.375],
            export_input_channel_order="bgr",
        ).eval()
        bgr = self.batch.image_data.images[:, 0].flip(1)

        onnx_output = _run_exported_graph(detector.build_export_spec(self.batch), bgr)[1]

        with torch.no_grad():
            raw = detector(self.batch).detection2d().predictions
        expected = torch.cat([raw[..., :4], raw[..., 4:].sigmoid()], dim=-1)
        np.testing.assert_allclose(onnx_output, expected.numpy(), atol=1e-5, rtol=1e-4)

    def test_onnx_graph_of_a_bgr_backbone_is_the_same_for_a_bgr_runtime(self) -> None:
        """Test that deploying a BGR backbone for a BGR runtime adds nothing to its graph.

        Check: the graph exported with the BGR export order is node for node and weight for
        weight the graph exported in the order of the backbone, and returns the same output.
        """
        for_runtime = build_detector(
            export_background_class=True, export_input_channel_order="bgr"
        ).eval()
        for_runtime.load_state_dict(self.detector.state_dict())
        bgr = self.batch.image_data.images[:, 0].flip(1)

        graph, output = _run_exported_graph(self.detector.build_export_spec(self.batch), bgr)
        runtime_graph, runtime_output = _run_exported_graph(
            for_runtime.build_export_spec(self.batch), bgr
        )

        self.assertEqual(runtime_graph, graph)
        np.testing.assert_array_equal(runtime_output, output)

    def test_export_spec_names_the_tensors_of_the_runtime(self) -> None:
        """Test the export contract: tensor names, the BGR input and the output layout.

        Check: the graph takes ``images`` in BGR and returns ``output`` with one row per
        prior holding the raw box regression, the objectness and, behind a background score
        of zero, the class scores of the averaged weights.
        """
        spec = self.detector.build_export_spec(self.batch)

        with torch.no_grad():
            output = spec.module(*spec.args)
            raw = self.detector(self.batch).detection2d().predictions

        self.assertEqual(spec.input_names, ["images"])
        self.assertEqual(spec.output_names, ["output"])
        self.assertIsNone(spec.dynamic_axes)
        torch.testing.assert_close(spec.args[0], self.batch.image_data.images[:, 0].flip(1))
        self.assertEqual(tuple(output.shape), (1, 84, 8))
        torch.testing.assert_close(output[..., :4], raw[..., :4])
        torch.testing.assert_close(output[..., 4], raw[..., 4].sigmoid())
        self.assertEqual(float(output[..., 5].abs().max()), 0.0)
        torch.testing.assert_close(output[..., 6:], raw[..., 5:].sigmoid())

    def test_exported_graph_matches_the_detector_in_onnx_runtime(self) -> None:
        """Test that the exported ONNX file reproduces the export module in ONNX Runtime.

        Check: the graph has a static single image input, ONNX Runtime returns the output of
        the export module, and the export leaves the averaged network in eval mode.
        """
        spec = self.detector.build_export_spec(self.batch)
        with tempfile.TemporaryDirectory() as directory:
            onnx_path = Path(directory) / "detector.onnx"
            torch.onnx.export(
                spec.module,
                spec.args,
                str(onnx_path),
                input_names=spec.input_names,
                output_names=spec.output_names,
                opset_version=17,
                dynamo=False,
            )
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            onnx_output = session.run(None, {"images": spec.args[0].numpy()})[0]

        with torch.no_grad():
            expected = spec.module(*spec.args).numpy()
        self.assertEqual(session.get_inputs()[0].shape, [1, 3, 64, 64])
        self.assertFalse(self.detector.ema.module.training)
        np.testing.assert_allclose(onnx_output, expected, atol=1e-4, rtol=1e-4)
