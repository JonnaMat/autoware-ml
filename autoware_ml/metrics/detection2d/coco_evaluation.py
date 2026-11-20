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

"""COCO style evaluation of 2D detections, with box size bins of free choice.

The protocol is the one of the COCO detection evaluation (``pycocotools.cocoeval``): per class
and per image the detections are matched greedily in score order to the ground truth box they
overlap most, the precision is made monotonic and averaged over 101 recall levels, and the
average precision is the mean over IoU thresholds and classes. The size bins differ from
COCO: a bin is a range of the square root of the box area in pixels, set by the caller.
Ground truth outside a bin is ignored when the bin is evaluated, as is an unmatched
detection outside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Bool, Float, Int
import numpy as np

RECALL_LEVELS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class SizeBin:
    """Range of box sizes, the size of a box being the square root of its area in pixels.

    Attributes:
        name: Name of the bin, part of the metric names.
        min_size: Smallest size of the bin, included.
        max_size: Size the bin ends before, None for a bin without an upper end.
    """

    name: str
    min_size: float = 0.0
    max_size: float | None = None

    def contains(self, sizes: Float[np.ndarray, " num_boxes"]) -> Bool[np.ndarray, " num_boxes"]:
        """Tell which box sizes fall into the bin.

        Args:
            sizes: Square root of the area of every box.

        Returns:
            Mask of the boxes of the bin.
        """
        inside = sizes >= self.min_size
        if self.max_size is not None:
            inside &= sizes < self.max_size
        return inside


@dataclass(frozen=True)
class Detection2DSample:
    """Detections and ground truth of one image, in pixels of the same image.

    Attributes:
        pred_boxes: Detected boxes ``(x_min, y_min, x_max, y_max)``.
        pred_scores: Score of every detection.
        pred_labels: Class index of every detection.
        gt_boxes: Ground truth boxes ``(x_min, y_min, x_max, y_max)``.
        gt_labels: Class index of every ground truth box.
    """

    pred_boxes: Float[np.ndarray, "num_predictions 4"]
    pred_scores: Float[np.ndarray, " num_predictions"]
    pred_labels: Int[np.ndarray, " num_predictions"]
    gt_boxes: Float[np.ndarray, "num_gt 4"]
    gt_labels: Int[np.ndarray, " num_gt"]


def box_sizes(boxes: Float[np.ndarray, "num_boxes 4"]) -> Float[np.ndarray, " num_boxes"]:
    """Square root of the area of every box.

    Args:
        boxes: Boxes ``(x_min, y_min, x_max, y_max)``.

    Returns:
        The size of every box, 0 for a box without an area.
    """
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return np.sqrt(widths * heights)


def pairwise_iou(
    boxes: Float[np.ndarray, "num_boxes 4"], others: Float[np.ndarray, "num_others 4"]
) -> Float[np.ndarray, "num_boxes num_others"]:
    """IoU of every box with every other box.

    Args:
        boxes: Boxes ``(x_min, y_min, x_max, y_max)``.
        others: Boxes in the same layout.

    Returns:
        The IoU of every pair, 0 for a pair without a union.
    """
    left_top = np.maximum(boxes[:, None, :2], others[None, :, :2])
    right_bottom = np.minimum(boxes[:, None, 2:], others[None, :, 2:])
    intersection = np.prod(np.clip(right_bottom - left_top, 0.0, None), axis=2)
    areas = box_sizes(boxes) ** 2
    other_areas = box_sizes(others) ** 2
    union = areas[:, None] + other_areas[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


@dataclass(frozen=True)
class _ClassMatches:
    """Matching of the detections of one class over every image, per size bin.

    Attributes:
        scores: Score of every detection, images concatenated.
        matched: Whether a detection is matched to a ground truth box, per bin and IoU
          threshold.
        ignored: Whether a detection counts neither as true nor as false positive.
        num_gt: Number of ground truth boxes of every bin.
    """

    scores: Float[np.ndarray, " num_detections"]
    matched: Bool[np.ndarray, "num_bins num_thresholds num_detections"]
    ignored: Bool[np.ndarray, "num_bins num_thresholds num_detections"]
    num_gt: Int[np.ndarray, " num_bins"]


class COCOEvaluation:
    """Detections of a set of images matched to their ground truth, ready to be scored.

    The first size bin is always ``all``, the bin of every box; the bins of the caller follow.
    """

    def __init__(
        self,
        samples: Sequence[Detection2DSample],
        num_classes: int,
        size_bins: Sequence[SizeBin] = (),
        iou_thresholds: Sequence[float] = tuple(np.linspace(0.5, 0.95, 10)),
        max_detections: int = 100,
    ) -> None:
        """Match the detections of every image.

        Args:
            samples: Detections and ground truth of every image.
            num_classes: Number of classes.
            size_bins: Size bins evaluated besides the bin of every box.
            iou_thresholds: IoU thresholds a match is evaluated at.
            max_detections: Number of highest scored detections per image and class that are
              evaluated.

        Raises:
            ValueError: If a bin is named ``all`` or two bins share a name.
        """
        names = [size_bin.name for size_bin in size_bins]
        if "all" in names or len(set(names)) != len(names):
            raise ValueError(f"Size bin names must be unique and not 'all', got {names}.")
        self.size_bins = (SizeBin("all"), *size_bins)
        self.iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64)
        self.num_classes = num_classes
        self.max_detections = max_detections
        self._matches = [self._match_class(samples, label) for label in range(num_classes)]

    def _match_class(self, samples: Sequence[Detection2DSample], label: int) -> _ClassMatches:
        """Match the detections of one class in every image."""
        num_bins, num_thresholds = len(self.size_bins), len(self.iou_thresholds)
        scores, matched, ignored = [], [], []
        num_gt = np.zeros(num_bins, dtype=np.int64)
        for sample in samples:
            gt_boxes = sample.gt_boxes[sample.gt_labels == label]
            of_class = sample.pred_labels == label
            order = np.argsort(-sample.pred_scores[of_class], kind="stable")[: self.max_detections]
            pred_boxes = sample.pred_boxes[of_class][order]
            gt_sizes = box_sizes(gt_boxes)
            pred_sizes = box_sizes(pred_boxes)
            gt_in_bin = np.stack([size_bin.contains(gt_sizes) for size_bin in self.size_bins])
            pred_in_bin = np.stack([size_bin.contains(pred_sizes) for size_bin in self.size_bins])
            num_gt += gt_in_bin.sum(axis=1)
            image_matched, image_ignored = self._match_image(
                pairwise_iou(pred_boxes, gt_boxes), gt_in_bin, pred_in_bin
            )
            scores.append(sample.pred_scores[of_class][order])
            matched.append(image_matched)
            ignored.append(image_ignored)
        empty = np.zeros((num_bins, num_thresholds, 0), dtype=bool)
        return _ClassMatches(
            scores=np.concatenate(scores) if scores else np.zeros(0),
            matched=np.concatenate(matched, axis=2) if matched else empty,
            ignored=np.concatenate(ignored, axis=2) if ignored else empty,
            num_gt=num_gt,
        )

    def _match_image(
        self,
        ious: Float[np.ndarray, "num_predictions num_gt"],
        gt_in_bin: Bool[np.ndarray, "num_bins num_gt"],
        pred_in_bin: Bool[np.ndarray, "num_bins num_predictions"],
    ) -> tuple[
        Bool[np.ndarray, "num_bins num_thresholds num_predictions"],
        Bool[np.ndarray, "num_bins num_thresholds num_predictions"],
    ]:
        """Match the detections of one class of one image, in decreasing score order.

        A detection takes the unmatched ground truth box of its bin it overlaps most, when the
        IoU reaches the threshold. Only when no such box is left it may take a ground truth
        box outside the bin, which makes the detection ignored instead of a false positive.
        An unmatched detection outside the bin is ignored too.

        Args:
            ious: IoU of every detection with every ground truth box of the class.
            gt_in_bin: Which ground truth boxes belong to every size bin.
            pred_in_bin: Which detections belong to every size bin.

        Returns:
            Whether every detection is matched and whether it is ignored, per bin and IoU
            threshold.
        """
        num_bins, num_thresholds = len(self.size_bins), len(self.iou_thresholds)
        num_predictions, num_gt = ious.shape
        matched = np.zeros((num_bins, num_thresholds, num_predictions), dtype=bool)
        ignored = np.repeat(~pred_in_bin[:, None, :], num_thresholds, axis=1)
        if num_gt == 0 or num_predictions == 0:
            return matched, ignored
        gt_taken = np.zeros((num_bins, num_thresholds, num_gt), dtype=bool)
        # A box of the bin always wins over a box outside it: IoU is at most 1.
        preference = 2.0 * gt_in_bin[:, None, :]
        bin_index, threshold_index = np.indices((num_bins, num_thresholds))
        for prediction in range(num_predictions):
            candidates = (ious[prediction][None, None, :] >= self.iou_thresholds[None, :, None]) & (
                ~gt_taken
            )
            value = np.where(candidates, ious[prediction][None, None, :] + preference, -1.0)
            # The last of equally good boxes wins, as in the COCO evaluation.
            best = num_gt - 1 - np.argmax(value[:, :, ::-1], axis=2)
            found = np.take_along_axis(value, best[:, :, None], axis=2)[:, :, 0] >= 0.0
            gt_taken[bin_index[found], threshold_index[found], best[found]] = True
            matched[:, :, prediction] = found
            matched_outside_bin = ~gt_in_bin[bin_index, best]
            ignored[:, :, prediction] = np.where(
                found, matched_outside_bin, ignored[:, :, prediction]
            )
        return matched, ignored

    def _bin_index(self, size_bin: str) -> int:
        """Index of a size bin by name."""
        for index, candidate in enumerate(self.size_bins):
            if candidate.name == size_bin:
                return index
        raise KeyError(f"Unknown size bin {size_bin!r}.")

    def num_gt(self, label: int, size_bin: str = "all") -> int:
        """Number of ground truth boxes of a class in a size bin.

        Args:
            label: Class index.
            size_bin: Name of the size bin.

        Returns:
            The number of boxes.
        """
        return int(self._matches[label].num_gt[self._bin_index(size_bin)])

    def average_precision(
        self, label: int, size_bin: str = "all"
    ) -> Float[np.ndarray, " num_thresholds"] | None:
        """Average precision of a class in a size bin at every IoU threshold.

        Args:
            label: Class index.
            size_bin: Name of the size bin.

        Returns:
            The average precision per IoU threshold, None when the class has no ground truth
            box in the bin.
        """
        matches = self._matches[label]
        bin_index = self._bin_index(size_bin)
        num_gt = int(matches.num_gt[bin_index])
        if num_gt == 0:
            return None
        order = np.argsort(-matches.scores, kind="mergesort")
        matched = matches.matched[bin_index][:, order]
        counted = ~matches.ignored[bin_index][:, order]
        true_positives = np.cumsum(matched & counted, axis=1, dtype=np.float64)
        false_positives = np.cumsum(~matched & counted, axis=1, dtype=np.float64)
        average_precisions = np.zeros(len(self.iou_thresholds))
        for index, (tp, fp) in enumerate(zip(true_positives, false_positives, strict=True)):
            recall = tp / num_gt
            precision = tp / np.maximum(tp + fp, np.spacing(1))
            # Precision at a recall level is the best precision at any higher recall.
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            positions = np.searchsorted(recall, RECALL_LEVELS, side="left")
            sampled = np.zeros(len(RECALL_LEVELS))
            reached = positions < len(precision)
            sampled[reached] = precision[positions[reached]]
            average_precisions[index] = sampled.mean()
        return average_precisions

    def counts_at_score(
        self, label: int, score_threshold: float, iou_threshold: float, size_bin: str = "all"
    ) -> tuple[int, int, int]:
        """Count the detections of a class at a fixed score threshold, as a deployed detector.

        Args:
            label: Class index.
            score_threshold: Lowest score of a counted detection.
            iou_threshold: IoU threshold of a match, one of the evaluated thresholds.
            size_bin: Name of the size bin.

        Returns:
            The true positives, the false positives and the ground truth boxes of the bin.

        Raises:
            ValueError: If the IoU threshold is not one of the evaluated thresholds.
        """
        close = np.isclose(self.iou_thresholds, iou_threshold)
        if not close.any():
            raise ValueError(
                f"IoU threshold {iou_threshold} is not evaluated, the thresholds are "
                f"{self.iou_thresholds.tolist()}."
            )
        matches = self._matches[label]
        bin_index = self._bin_index(size_bin)
        threshold_index = int(np.argmax(close))
        counted = (matches.scores >= score_threshold) & ~matches.ignored[bin_index, threshold_index]
        matched = matches.matched[bin_index, threshold_index]
        return (
            int((matched & counted).sum()),
            int((~matched & counted).sum()),
            int(matches.num_gt[bin_index]),
        )
