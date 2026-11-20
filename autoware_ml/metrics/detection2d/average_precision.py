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

"""COCO style average precision of 2D detections, overall, per class and per box size."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection2d.coco_evaluation import COCOEvaluation, SizeBin
from autoware_ml.metrics.detection2d.suite import Detection2DState
from autoware_ml.metrics.detection3d.naming import metric_token
from autoware_ml.metrics.report import MetricReport


class COCOAveragePrecision(Metric[Detection2DState]):
    """Average precision by the COCO protocol, sliced by box size.

    The size of a box is the square root of its area in pixels of the original image. With
    ``<class>`` a class name and ``<bin>`` the name of a size bin, the metric reports:

    * ``ap``, ``ap50``, ``ap75``: mean over the classes of the average precision over the IoU
      thresholds, at IoU 0.5 and at IoU 0.75;
    * ``ap_<class>``, ``ap50_<class>``, ``ap75_<class>``: the same per class;
    * ``ap_size_<bin>``, ``ap50_size_<bin>``: the same for the boxes of a size bin, and
      ``ap_size_<bin>_<class>`` per class;
    * ``ap_small``: ``ap_size_<bin>`` of the bin ``small_bin`` names, the one number a run is
      selected by, so what counts as small is a config value;
    * ``recall``, ``precision`` and ``recall_size_<bin>``: recall and precision at IoU
      ``operating_iou_threshold`` of the detections scoring at least
      ``operating_score_threshold``, the operating point of a deployed detector, pooled over
      the classes;
    * ``num_gt`` and ``num_gt_size_<bin>``: number of ground truth boxes behind the values.

    A class without a ground truth box in a slice is left out of the mean of that slice; a
    slice without any ground truth box reports -1, as the COCO evaluation does.
    """

    range_invariant = True

    def __init__(
        self,
        size_bins: Mapping[str, Sequence[float | None]],
        small_bin: str,
        iou_thresholds: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
        max_detections: int = 100,
        operating_score_threshold: float = 0.35,
        operating_iou_threshold: float = 0.5,
        stages: tuple[str, ...] | list[str] = ("val", "test"),
    ) -> None:
        """Initialize the metric.

        Args:
            size_bins: Size bins by name, each ``[min_size, max_size]`` on the square root of
              the box area in pixels, the upper end excluded and None for no upper end.
            small_bin: Name of the size bin reported as ``ap_small``.
            iou_thresholds: IoU thresholds averaged into the average precision. They must
              include 0.5, 0.75 and ``operating_iou_threshold``.
            max_detections: Number of highest scored detections per image and class that are
              evaluated.
            operating_score_threshold: Score threshold of the recall and the precision.
            operating_iou_threshold: IoU threshold of the recall and the precision.
            stages: Stages the metric reports at.

        Raises:
            ValueError: If ``small_bin`` names no size bin, a bin is not a ``[min, max]``
              pair, or a required IoU threshold is missing.
        """
        super().__init__(stages=stages)
        self.size_bins = []
        for name, bounds in size_bins.items():
            if len(bounds) != 2:
                raise ValueError(f"Size bin {name!r} must be [min_size, max_size], got {bounds}.")
            self.size_bins.append(
                SizeBin(
                    name=str(name),
                    min_size=float(bounds[0] or 0.0),
                    max_size=None if bounds[1] is None else float(bounds[1]),
                )
            )
        if small_bin not in size_bins:
            raise ValueError(
                f"small_bin {small_bin!r} names none of the size bins {list(size_bins)}."
            )
        self.small_bin = small_bin
        self.iou_thresholds = tuple(float(threshold) for threshold in iou_thresholds)
        for required in (0.5, 0.75, operating_iou_threshold):
            if not np.isclose(self.iou_thresholds, required).any():
                raise ValueError(
                    f"iou_thresholds must include {required}, got {list(self.iou_thresholds)}."
                )
        self.max_detections = max_detections
        self.operating_score_threshold = operating_score_threshold
        self.operating_iou_threshold = operating_iou_threshold

    def _threshold_index(self, iou_threshold: float) -> int:
        """Index of an IoU threshold."""
        return int(np.argmax(np.isclose(self.iou_thresholds, iou_threshold)))

    def evaluate(self, state: Detection2DState, stage: EvalStage) -> MetricReport:
        """Compute the average precision values from the accumulated frames.

        Args:
            state: Detections and ground truth of every evaluated image.
            stage: Stage the values are reported for.

        Returns:
            The report of every value the class docstring lists.
        """
        num_classes = len(state.class_names)
        evaluation = COCOEvaluation(
            state.samples,
            num_classes=num_classes,
            size_bins=self.size_bins,
            iou_thresholds=self.iou_thresholds,
            max_detections=self.max_detections,
        )
        index_50 = self._threshold_index(0.5)
        index_75 = self._threshold_index(0.75)
        report = MetricReport()

        def mean_or_missing(values: list[float]) -> float:
            """Mean over the classes that have ground truth, -1 when none has."""
            return float(np.mean(values)) if values else -1.0

        for size_bin in ("all", *(size_bin.name for size_bin in self.size_bins)):
            suffix = "" if size_bin == "all" else f"_size_{metric_token(size_bin)}"
            per_class = [
                evaluation.average_precision(label, size_bin) for label in range(num_classes)
            ]
            valid = [values for values in per_class if values is not None]
            report.record(f"ap{suffix}", mean_or_missing([values.mean() for values in valid]))
            report.record(f"ap50{suffix}", mean_or_missing([values[index_50] for values in valid]))
            if size_bin == "all":
                report.record("ap75", mean_or_missing([values[index_75] for values in valid]))
            for class_name, values in zip(state.class_names, per_class, strict=True):
                token = metric_token(class_name)
                report.record(
                    f"ap{suffix}_{token}", float(values.mean()) if values is not None else -1.0
                )
                if size_bin == "all":
                    report.record(
                        f"ap50_{token}", float(values[index_50]) if values is not None else -1.0
                    )
                    report.record(
                        f"ap75_{token}", float(values[index_75]) if values is not None else -1.0
                    )
            if size_bin == self.small_bin:
                report.record("ap_small", mean_or_missing([values.mean() for values in valid]))

            counts = np.sum(
                [
                    evaluation.counts_at_score(
                        label,
                        self.operating_score_threshold,
                        self.operating_iou_threshold,
                        size_bin,
                    )
                    for label in range(num_classes)
                ],
                axis=0,
            )
            true_positives, false_positives, num_gt = (int(count) for count in counts)
            report.record(f"recall{suffix}", true_positives / num_gt if num_gt else -1.0)
            if size_bin == "all":
                num_counted = true_positives + false_positives
                report.record("precision", true_positives / num_counted if num_counted else -1.0)
            report.record(f"num_gt{suffix}", float(num_gt))
        return report
