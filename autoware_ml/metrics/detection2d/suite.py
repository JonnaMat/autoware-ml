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

"""The 2D detection metric suite: accumulates the frames and hands them to its metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from autoware_ml.metrics.base import IDENTITY, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.metrics.detection2d.coco_evaluation import Detection2DSample
from autoware_ml.metrics.eval_output import EvalOutput, MetricRequirements


@dataclass(frozen=True)
class Detection2DState:
    """What the 2D detection metrics consume.

    Attributes:
        samples: Detections and ground truth of every evaluated image.
        class_names: Names of the classes, in index order.
    """

    samples: list[Detection2DSample]
    class_names: tuple[str, ...]


class Detection2DMetricSuite(MetricSuite[Detection2DState]):
    """Accumulate the 2D detections and ground truth boxes of every image of an epoch.

    The suite has no distance ranges and its metrics take no filter: a 2D metric slices the
    boxes by their size in the image, which the metric itself configures.
    """

    prefix = "det2d"

    def __init__(
        self,
        components: list[Metric[Detection2DState]],
        class_names: tuple[str, ...] | list[str],
        **kwargs: Any,
    ) -> None:
        """Register the accumulation states.

        Args:
            components: Metrics computed from the accumulated frames.
            class_names: Names of the classes, in index order.
            **kwargs: Arguments of :class:`MetricSuite`.

        Raises:
            ValueError: If a component carries a filter.
        """
        super().__init__(components=components, ranges=(), **kwargs)
        self.class_names = tuple(class_names)
        for component in self.components:
            if component.filter is not IDENTITY:
                raise ValueError(
                    f"{type(component).__name__} carries the filter {component.filter.name!r}, "
                    "the 2D detection suite supports none."
                )
        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("pred_scores", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("gt_boxes", default=[], dist_reduce_fx=None)
        self.add_state("gt_labels", default=[], dist_reduce_fx=None)
        # Per-frame flag of the frames that only pad a streaming lane: gathered like every
        # other frame, so the ranks hold as many entries, and left out of every metric.
        self.add_state("padding", default=[], dist_reduce_fx=None)

    def requirements(self) -> MetricRequirements:
        """The suite reads the 2D detection frames."""
        return MetricRequirements(detection2d=True)

    def update(self, eval_output: EvalOutput) -> None:
        """Accumulate the frames of one batch.

        Args:
            eval_output: Eval output of the model for one batch.

        Raises:
            ValueError: If the eval output carries no 2D detection frames, or a ground truth
              label is outside the classes.
        """
        output = eval_output.detection2d
        if output is None:
            raise ValueError(
                "The 2D detection suite needs the detection2d frames, the eval output carries none."
            )
        for index, frame in enumerate(output.frames):
            gt_labels = frame.gt_labels.detach().to(dtype=torch.long)
            if bool(((gt_labels < 0) | (gt_labels >= len(self.class_names))).any()):
                raise ValueError(
                    f"Frame {index} carries ground truth labels outside the "
                    f"{len(self.class_names)} classes."
                )
            self._note_frame()
            self.pred_boxes.append(frame.predictions.bboxes_2d.detach().to(dtype=torch.float32))
            self.pred_scores.append(frame.predictions.scores_2d.detach().to(dtype=torch.float32))
            self.pred_labels.append(frame.predictions.labels_2d.detach().to(dtype=torch.long))
            self.gt_boxes.append(frame.gt_boxes.detach().to(dtype=torch.float32))
            self.gt_labels.append(gt_labels)
            self.padding.append(torch.tensor([frame.is_padding], device=gt_labels.device))

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter = IDENTITY
    ) -> Detection2DState:
        """Build the state the metrics consume.

        Args:
            metric_range: Always None, the suite has no ranges.
            metric_filter: Always the identity, the suite has no filters.

        Returns:
            The accumulated frames, those that only pad a streaming lane left out.
        """
        samples = [
            Detection2DSample(
                pred_boxes=self.pred_boxes[index].cpu().numpy().astype("float64"),
                pred_scores=self.pred_scores[index].cpu().numpy().astype("float64"),
                pred_labels=self.pred_labels[index].cpu().numpy(),
                gt_boxes=self.gt_boxes[index].cpu().numpy().astype("float64"),
                gt_labels=self.gt_labels[index].cpu().numpy(),
            )
            for index in range(len(self.gt_boxes))
            if not bool(self.padding[index].item())
        ]
        return Detection2DState(samples=samples, class_names=self.class_names)
