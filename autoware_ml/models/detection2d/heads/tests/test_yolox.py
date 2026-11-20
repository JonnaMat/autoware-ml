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

"""Unit tests for the YOLOX head: output layout, priors, decoding, losses and postprocessing."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.models.detection2d.head_outputs import YOLOXHeadOutputs
from autoware_ml.models.detection2d.heads.yolox import (
    YOLOXHead,
    decode_boxes,
    postprocess_detections,
)


class TestYOLOXHead(unittest.TestCase):
    """Unit tests for the YOLOX head."""

    def setUp(self) -> None:
        """Set up a two-class head over three levels of a 64x64 input."""
        torch.manual_seed(0)
        self.head = YOLOXHead(num_classes=2, in_channels=[16, 32, 64], feat_channels=16)
        self.features = [
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 32, 4, 4),
            torch.randn(2, 64, 2, 2),
        ]

    def test_forward_returns_one_map_per_level(self) -> None:
        """Test that every level predicts 4 box values, 1 objectness and the class logits."""
        outputs = self.head(self.features)

        self.assertEqual(
            [tuple(o.shape) for o in outputs], [(2, 7, 8, 8), (2, 7, 4, 4), (2, 7, 2, 2)]
        )

    def test_flattened_outputs_follow_the_priors(self) -> None:
        """Test that the flattened predictions and the priors are ordered level by level, row by row."""
        level_outputs = self.head(self.features)

        outputs = self.head.flatten_outputs(level_outputs)

        self.assertEqual(tuple(outputs.predictions.shape), (2, 84, 7))
        self.assertEqual(tuple(outputs.priors.shape), (84, 3))
        self.assertEqual(outputs.priors[:64, 2].unique().tolist(), [8.0])
        self.assertEqual(outputs.priors[64:80, 2].unique().tolist(), [16.0])
        # Prior 10 of the first level is column 2 of row 1.
        self.assertEqual(outputs.priors[10].tolist(), [2.0, 1.0, 8.0])
        torch.testing.assert_close(outputs.predictions[1, 10], level_outputs[0][1, :, 1, 2])

    def test_decoding_adds_the_grid_and_scales_by_the_stride(self) -> None:
        """Test that a zero regression decodes to a box of one stride at the corner of its cell."""
        priors = torch.tensor([[2.0, 1.0, 8.0], [0.0, 3.0, 16.0]])
        predictions = torch.zeros(2, 7)
        predictions[1, :4] = torch.tensor([0.5, 0.25, 1.0, 0.0])

        boxes = decode_boxes(predictions, priors)

        torch.testing.assert_close(boxes[0], torch.tensor([16.0, 8.0, 8.0, 8.0]))
        torch.testing.assert_close(boxes[1], torch.tensor([8.0, 52.0, 16.0 * torch.e, 16.0]))

    def test_export_outputs_keep_the_regression_raw_and_squash_the_scores(self) -> None:
        """Test the deployed layout: raw box regression, sigmoid objectness and class scores."""
        level_outputs = self.head(self.features)
        raw = self.head.flatten_outputs(level_outputs).predictions

        exported = self.head.export_outputs(level_outputs)

        torch.testing.assert_close(exported[..., :4], raw[..., :4])
        torch.testing.assert_close(exported[..., 4:], raw[..., 4:].sigmoid())

    def test_biases_start_at_the_prior_probability(self) -> None:
        """Test that the objectness and class outputs start at a probability of 0.01."""
        for conv in [*self.head.cls_preds, *self.head.obj_preds]:
            torch.testing.assert_close(conv.bias.sigmoid(), torch.full_like(conv.bias, 0.01))

    def test_loss_is_finite_and_reaches_every_output(self) -> None:
        """Test that the loss of a batch with and without boxes is finite and trains the head."""
        outputs = self.head.flatten_outputs(self.head(self.features))
        gt_boxes = [
            torch.tensor([[8.0, 8.0, 40.0, 40.0], [30.0, 10.0, 50.0, 28.0]]),
            torch.zeros(0, 4),
        ]
        gt_labels = [torch.tensor([0, 1]), torch.zeros(0, dtype=torch.long)]

        losses = self.head.loss(outputs, gt_boxes, gt_labels)
        losses["loss"].backward()

        for name in ("loss", "loss_iou", "loss_obj", "loss_cls", "loss_l1", "num_fg"):
            self.assertTrue(torch.isfinite(losses[name]), name)
        self.assertEqual(float(losses["loss_l1"]), 0.0)
        self.assertGreater(float(losses["num_fg"]), 0.0)
        for name, parameter in self.head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_l1_loss_joins_the_total_when_switched_on(self) -> None:
        """Test that ``use_l1`` adds the L1 loss of the raw regression to the total loss."""
        outputs = self.head.flatten_outputs(self.head(self.features))
        gt_boxes = [
            torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
            torch.tensor([[20.0, 20.0, 36.0, 44.0]]),
        ]
        gt_labels = [torch.tensor([0]), torch.tensor([1])]

        without_l1 = self.head.loss(outputs, gt_boxes, gt_labels)
        self.head.use_l1 = True
        with_l1 = self.head.loss(outputs, gt_boxes, gt_labels)

        self.assertGreater(float(with_l1["loss_l1"].detach()), 0.0)
        self.assertAlmostEqual(
            float(with_l1["loss"].detach()),
            float((without_l1["loss"] + with_l1["loss_l1"]).detach()),
            places=4,
        )

    def test_loss_of_a_perfect_prediction_is_small(self) -> None:
        """Test that predictions matching the ground truth give nearly no loss.

        Setup: one 16 px box on cell (2, 1) of the first level. Its prior predicts the box
        exactly with a confident objectness and class, every other prior a low objectness.
        Check: the prior is the only foreground one and the IoU and L1 losses vanish.
        """
        priors = self.head.build_priors([(8, 8), (4, 4), (2, 2)], torch.device("cpu"))
        predictions = torch.zeros(1, 84, 7)
        # Every other prior predicts a box too small to overlap the ground truth.
        predictions[..., 2:4] = -6.0
        predictions[..., 4] = -12.0
        predictions[..., 5:] = -12.0
        predictions[0, 10] = torch.tensor([0.5, 0.5, 0.6931472, 0.6931472, 12.0, 12.0, -12.0])
        outputs = YOLOXHeadOutputs(predictions=predictions, priors=priors)
        self.head.use_l1 = True

        losses = self.head.loss(
            outputs, [torch.tensor([[12.0, 4.0, 28.0, 20.0]])], [torch.tensor([0])]
        )

        self.assertAlmostEqual(float(losses["num_fg"]), 1.0)
        self.assertLess(float(losses["loss_iou"]), 1e-5)
        self.assertLess(float(losses["loss_l1"]), 1e-5)
        self.assertLess(float(losses["loss"]), 1e-2)

    def test_number_of_levels_is_checked(self) -> None:
        """Test that a stride count differing from the level count is refused."""
        with self.assertRaises(ValueError):
            YOLOXHead(num_classes=2, in_channels=[16, 32], strides=[8, 16, 32])


class TestPostprocessDetections(unittest.TestCase):
    """Unit tests for the decoding of dense predictions into detections."""

    def setUp(self) -> None:
        """Set up three priors: two overlapping boxes and one apart."""
        self.boxes = torch.tensor(
            [[[20.0, 20.0, 20.0, 20.0], [21.0, 20.0, 20.0, 20.0], [70.0, 70.0, 10.0, 10.0]]]
        )
        self.objectness = torch.tensor([[0.9, 0.8, 0.5]])

    def test_overlapping_boxes_of_one_class_are_suppressed(self) -> None:
        """Test that the lower scored of two overlapping boxes of a class is dropped."""
        class_scores = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.8]]])

        detections = postprocess_detections(self.boxes, self.objectness, class_scores, 0.1, 0.5)[0]

        self.assertEqual(detections.labels_2d.tolist(), [0, 1])
        torch.testing.assert_close(detections.scores_2d, torch.tensor([0.9, 0.4]))
        torch.testing.assert_close(detections.bboxes_2d[0], torch.tensor([10.0, 10.0, 30.0, 30.0]))

    def test_classes_only_suppress_each_other_when_class_agnostic(self) -> None:
        """Test that overlapping boxes of two classes survive unless the NMS is class agnostic."""
        class_scores = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.8]]])

        per_class = postprocess_detections(self.boxes, self.objectness, class_scores, 0.1, 0.5)[0]
        agnostic = postprocess_detections(
            self.boxes, self.objectness, class_scores, 0.1, 0.5, class_agnostic_nms=True
        )[0]

        self.assertEqual(per_class.labels_2d.tolist(), [0, 1, 1])
        self.assertEqual(agnostic.labels_2d.tolist(), [0, 1])

    def test_score_threshold_and_detection_cap(self) -> None:
        """Test that low scores are dropped and only the highest scored detections are kept."""
        class_scores = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.8]]])

        thresholded = postprocess_detections(self.boxes, self.objectness, class_scores, 0.5, 0.5)[0]
        capped = postprocess_detections(
            self.boxes, self.objectness, class_scores, 0.1, 0.5, max_detections=1
        )[0]

        torch.testing.assert_close(thresholded.scores_2d, torch.tensor([0.9, 0.8]))
        torch.testing.assert_close(capped.scores_2d, torch.tensor([0.9]))
