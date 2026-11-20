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

"""Unit tests for the 2D detection metrics: the COCO evaluation on a hand-computed case, its
size bins, the operating point counts, the agreement with pycocotools and the metric suite."""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import pytest
import torch

from autoware_ml.dataclasses.models.detection2d.predictions import Detection2DSamplePredictions
from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection2d.average_precision import COCOAveragePrecision
from autoware_ml.metrics.detection2d.coco_evaluation import (
    COCOEvaluation,
    Detection2DSample,
    SizeBin,
)
from autoware_ml.metrics.detection2d.suite import Detection2DMetricSuite
from autoware_ml.metrics.eval_output import Detection2DEvalFrame, Detection2DEvalOutput, EvalOutput

IOU_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def _hand_computed_sample() -> Detection2DSample:
    """One image with two ground truth boxes and three detections of one class.

    Ground truth: A, 10x10 px (size 10), and B, 40x40 px (size 40).
    Detections by decreasing score: d1 on A exactly (IoU 1.0), d2 a 12x12 px box on nothing,
    d3 covering 62 % of B (IoU 0.62).
    """
    return Detection2DSample(
        pred_boxes=np.array(
            [[0.0, 0.0, 10.0, 10.0], [200.0, 200.0, 212.0, 212.0], [100.0, 100.0, 140.0, 124.8]]
        ),
        pred_scores=np.array([0.9, 0.8, 0.7]),
        pred_labels=np.array([0, 0, 0]),
        gt_boxes=np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 140.0, 140.0]]),
        gt_labels=np.array([0, 0]),
    )


class TestCOCOEvaluation(unittest.TestCase):
    """Unit tests for the COCO evaluation."""

    def setUp(self) -> None:
        """Evaluate the hand-computed sample with a small and a large size bin."""
        self.evaluation = COCOEvaluation(
            [_hand_computed_sample()],
            num_classes=1,
            size_bins=[SizeBin("small", 0.0, 16.0), SizeBin("large", 16.0, None)],
            iou_thresholds=IOU_THRESHOLDS,
        )

    def test_average_precision_matches_the_hand_computation(self) -> None:
        """Test the average precision of every IoU threshold against a hand computation.

        Up to IoU 0.6 the detections are TP, FP, TP: the precision is 1 up to a recall of 0.5
        (51 of the 101 recall levels) and 2/3 above (50 levels), so AP = (51 + 50 * 2/3) / 101.
        From IoU 0.65 on d3 misses B: TP, FP, FP, a recall of 0.5 at most, so AP = 51 / 101.
        """
        average_precisions = self.evaluation.average_precision(0)

        expected_low = (51 + 50 * 2 / 3) / 101
        expected_high = 51 / 101
        np.testing.assert_allclose(average_precisions[:3], expected_low, atol=1e-9)
        np.testing.assert_allclose(average_precisions[3:], expected_high, atol=1e-9)
        self.assertAlmostEqual(
            float(average_precisions.mean()), (3 * expected_low + 7 * expected_high) / 10
        )

    def test_size_bins_ignore_what_lies_outside(self) -> None:
        """Test the average precision of each size bin against a hand computation.

        Small bin: A is its only ground truth box. d1 hits it, d2 is a small false positive
        and d3, matched to the large B or large and unmatched, is ignored: precision 1 at
        every recall level at every threshold.
        Large bin: B is its only ground truth box. d1, matched to the small A, and the small
        unmatched d2 are ignored. d3 hits B up to IoU 0.6 (AP 1) and is a false positive
        above (AP 0), which averages to 0.3.
        """
        small = self.evaluation.average_precision(0, "small")
        large = self.evaluation.average_precision(0, "large")

        np.testing.assert_allclose(small, 1.0)
        np.testing.assert_allclose(large, [1.0] * 3 + [0.0] * 7)
        self.assertEqual(self.evaluation.num_gt(0, "small"), 1)
        self.assertEqual(self.evaluation.num_gt(0, "large"), 1)
        self.assertEqual(self.evaluation.num_gt(0), 2)

    def test_counts_at_a_score_threshold(self) -> None:
        """Test the true and false positives of the detections scoring at least a threshold."""
        self.assertEqual(self.evaluation.counts_at_score(0, 0.75, 0.5), (1, 1, 2))
        self.assertEqual(self.evaluation.counts_at_score(0, 0.5, 0.5), (2, 1, 2))
        self.assertEqual(self.evaluation.counts_at_score(0, 0.5, 0.75), (1, 2, 2))
        self.assertEqual(self.evaluation.counts_at_score(0, 0.5, 0.5, "large"), (1, 0, 1))
        with self.assertRaises(ValueError):
            self.evaluation.counts_at_score(0, 0.5, 0.42)

    def test_class_without_ground_truth_has_no_average_precision(self) -> None:
        """Test that a class without a ground truth box reports no average precision."""
        evaluation = COCOEvaluation([_hand_computed_sample()], num_classes=2)

        self.assertIsNone(evaluation.average_precision(1))
        self.assertIsNotNone(evaluation.average_precision(0))

    def test_only_the_highest_scored_detections_are_evaluated(self) -> None:
        """Test that detections beyond ``max_detections`` per image and class are left out."""
        evaluation = COCOEvaluation([_hand_computed_sample()], num_classes=1, max_detections=2)

        self.assertEqual(evaluation.counts_at_score(0, 0.0, 0.5), (1, 1, 2))

    def test_reserved_and_duplicate_bin_names_are_refused(self) -> None:
        """Test that a bin named ``all`` or two bins of one name are refused."""
        for names in (["all"], ["small", "small"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                COCOEvaluation([], num_classes=1, size_bins=[SizeBin(name) for name in names])

    def test_agrees_with_pycocotools(self) -> None:
        """Test that random detections score as pycocotools scores them, per class and size bin.

        Setup: 40 images with up to 6 ground truth boxes of two classes and jittered, missing
        and spurious detections. The size bins are the COCO area ranges as box sizes.
        Check: the per-class average precision of every bin equals the one pycocotools
        computes with the same area ranges.
        """
        coco_module = pytest.importorskip("pycocotools.coco")
        cocoeval_module = pytest.importorskip("pycocotools.cocoeval")
        rng = np.random.RandomState(0)
        samples = []
        for _ in range(40):
            num_gt = rng.randint(0, 7)
            top_left = rng.uniform(0, 400, (num_gt, 2))
            sizes = rng.uniform(4, 150, (num_gt, 2))
            gt_boxes = np.concatenate([top_left, top_left + sizes], axis=1)
            gt_labels = rng.randint(0, 2, num_gt)
            kept = rng.rand(num_gt) < 0.8
            jitter = rng.normal(0, 0.08, (int(kept.sum()), 4)) * np.tile(sizes[kept], 2)
            num_spurious = rng.randint(0, 4)
            spurious_top_left = rng.uniform(0, 400, (num_spurious, 2))
            spurious = np.concatenate(
                [spurious_top_left, spurious_top_left + rng.uniform(4, 150, (num_spurious, 2))],
                axis=1,
            )
            pred_boxes = np.concatenate([gt_boxes[kept] + jitter, spurious])
            pred_labels = np.concatenate([gt_labels[kept], rng.randint(0, 2, num_spurious)])
            samples.append(
                Detection2DSample(
                    pred_boxes=pred_boxes,
                    # Distinct scores: the two implementations may order ties differently.
                    pred_scores=rng.permutation(len(pred_boxes)) / 100.0 + rng.uniform(0, 0.005),
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )
            )
        bins = [
            SizeBin("small", 0.0, 32.0),
            SizeBin("medium", 32.0, 96.0),
            SizeBin("large", 96.0, None),
        ]
        evaluation = COCOEvaluation(
            samples, num_classes=2, size_bins=bins, iou_thresholds=IOU_THRESHOLDS
        )

        images, annotations, results = [], [], []
        for image_id, sample in enumerate(samples):
            images.append({"id": image_id, "width": 600, "height": 600})
            for box, label in zip(sample.gt_boxes, sample.gt_labels, strict=True):
                width, height = box[2] - box[0], box[3] - box[1]
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [box[0], box[1], width, height],
                        "area": width * height,
                        "iscrowd": 0,
                    }
                )
            for box, score, label in zip(
                sample.pred_boxes, sample.pred_scores, sample.pred_labels, strict=True
            ):
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": float(score),
                    }
                )
        with contextlib.redirect_stdout(io.StringIO()):
            ground_truth = coco_module.COCO()
            ground_truth.dataset = {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 0}, {"id": 1}],
            }
            ground_truth.createIndex()
            coco_eval = cocoeval_module.COCOeval(
                ground_truth, ground_truth.loadRes(results), "bbox"
            )
            # pycocotools includes both ends of an area range, a bin here excludes its upper end.
            coco_eval.params.areaRng = [
                [0, 1e10],
                [0, 32**2 - 1e-6],
                [32**2, 96**2 - 1e-6],
                [96**2, 1e10],
            ]
            coco_eval.params.areaRngLbl = ["all", "small", "medium", "large"]
            coco_eval.evaluate()
            coco_eval.accumulate()
        precision = coco_eval.eval["precision"]  # [thresholds, recall levels, classes, areas, dets]

        for label in range(2):
            for area_index, name in enumerate(["all", "small", "medium", "large"]):
                with self.subTest(label=label, size_bin=name):
                    expected = precision[:, :, label, area_index, -1].mean(axis=1)
                    np.testing.assert_allclose(
                        evaluation.average_precision(label, name), expected, atol=1e-9
                    )


class TestCOCOAveragePrecision(unittest.TestCase):
    """Unit tests for the average precision metric component."""

    def _suite(self, **kwargs: object) -> Detection2DMetricSuite:
        """Build a suite of two classes with the average precision component."""
        component = COCOAveragePrecision(
            size_bins={"tiny": [0, 16], "big": [16, None]}, small_bin="tiny", **kwargs
        )
        return Detection2DMetricSuite(components=[component], class_names=["light", "sign"])

    def _eval_output(self) -> EvalOutput:
        """Wrap the hand-computed sample as the eval output of a batch of one image."""
        sample = _hand_computed_sample()
        frame = Detection2DEvalFrame(
            predictions=Detection2DSamplePredictions(
                bboxes_2d=torch.from_numpy(sample.pred_boxes).float(),
                scores_2d=torch.from_numpy(sample.pred_scores).float(),
                labels_2d=torch.from_numpy(sample.pred_labels).long(),
            ),
            gt_boxes=torch.from_numpy(sample.gt_boxes).float(),
            gt_labels=torch.from_numpy(sample.gt_labels).long(),
        )
        return EvalOutput(detection2d=Detection2DEvalOutput(frames=[frame]))

    def test_suite_reports_the_documented_values(self) -> None:
        """Test the values of the suite on the hand-computed sample.

        Check: the mean leaves out the class without ground truth, ``ap_small`` is the average
        precision of the bin ``small_bin`` names, and the operating point at a score of 0.75
        counts d1 and d2.
        """
        suite = self._suite(operating_score_threshold=0.75)
        suite.update(self._eval_output())

        report = suite.result(EvalStage.TEST).to_log_dict("test")

        expected_low = (51 + 50 * 2 / 3) / 101
        expected_ap = (3 * expected_low + 7 * 51 / 101) / 10
        self.assertAlmostEqual(report["test/ap"], expected_ap, places=6)
        self.assertAlmostEqual(report["test/ap_light"], expected_ap, places=6)
        self.assertEqual(report["test/ap_sign"], -1.0)
        self.assertAlmostEqual(report["test/ap50"], expected_low, places=6)
        self.assertAlmostEqual(report["test/ap75"], 51 / 101, places=6)
        self.assertAlmostEqual(report["test/ap_size_tiny"], 1.0, places=6)
        self.assertAlmostEqual(report["test/ap_small"], 1.0, places=6)
        self.assertAlmostEqual(report["test/ap_size_big"], 0.3, places=6)
        self.assertAlmostEqual(report["test/recall"], 0.5)
        self.assertAlmostEqual(report["test/precision"], 0.5)
        self.assertAlmostEqual(report["test/recall_size_tiny"], 1.0)
        self.assertAlmostEqual(report["test/recall_size_big"], 0.0)
        self.assertEqual(report["test/num_gt"], 2.0)
        self.assertEqual(report["test/num_gt_size_tiny"], 1.0)

    def test_small_bin_follows_the_config(self) -> None:
        """Test that ``ap_small`` reports the bin its config key names."""
        component = COCOAveragePrecision(
            size_bins={"tiny": [0, 16], "big": [16, None]}, small_bin="big"
        )
        suite = Detection2DMetricSuite(components=[component], class_names=["light", "sign"])
        suite.update(self._eval_output())

        report = suite.result(EvalStage.TEST).to_log_dict("test")

        self.assertAlmostEqual(report["test/ap_small"], 0.3, places=6)

    def test_padding_frames_are_left_out(self) -> None:
        """Test that a frame flagged as padding does not change the values."""
        suite = self._suite()
        eval_output = self._eval_output()
        padding = eval_output.detection2d.frames[0].model_copy(update={"is_padding": True})
        suite.update(eval_output)
        suite.update(EvalOutput(detection2d=Detection2DEvalOutput(frames=[padding])))

        report = suite.result(EvalStage.TEST).to_log_dict("test")

        self.assertEqual(report["test/num_gt"], 2.0)

    def test_invalid_configurations_are_refused(self) -> None:
        """Test that an unknown small bin, a malformed bin and missing thresholds are refused."""
        with self.assertRaises(ValueError):
            COCOAveragePrecision(size_bins={"tiny": [0, 16]}, small_bin="small")
        with self.assertRaises(ValueError):
            COCOAveragePrecision(size_bins={"tiny": [0, 16, 32]}, small_bin="tiny")
        with self.assertRaises(ValueError):
            COCOAveragePrecision(
                size_bins={"tiny": [0, 16]}, small_bin="tiny", iou_thresholds=[0.5]
            )

    def test_labels_outside_the_classes_are_refused(self) -> None:
        """Test that a ground truth label outside the classes fails the update."""
        suite = self._suite()
        frame = (
            self._eval_output()
            .detection2d.frames[0]
            .model_copy(update={"gt_labels": torch.tensor([0, 5])})
        )

        with self.assertRaises(ValueError):
            suite.update(EvalOutput(detection2d=Detection2DEvalOutput(frames=[frame])))
