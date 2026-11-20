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

"""SimOTA label assignment of YOLOX.

Ported from the YOLOX reference implementation (``get_assignments``, ``get_geometry_constraint``
and ``simota_matching`` of ``yolox/models/yolo_head.py`` of
https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0).
"""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Bool, Float32, Int64
import torch
from torch import Tensor
import torch.nn.functional as F

from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D


class SimOTAAssignment(NamedTuple):
    """Result of the assignment of one image.

    Attributes:
        foreground_mask: Priors assigned to a ground truth box.
        matched_gt_indices: Index of the ground truth box of every foreground prior.
        matched_ious: IoU between the predicted box of every foreground prior and its ground
          truth box.
    """

    foreground_mask: Bool[Tensor, " num_priors"]
    matched_gt_indices: Int64[Tensor, " num_foreground"]
    matched_ious: Float32[Tensor, " num_foreground"]


class SimOTAAssigner:
    """Assign priors to ground truth boxes by a dynamic top-k over a joint cost.

    A prior is a candidate of a ground truth box when its center lies within
    ``center_radius`` strides of the box center. The cost of a pair is the classification
    cost plus ``iou_weight`` times ``-log(IoU)``. Every ground truth box takes its ``k``
    cheapest candidates, ``k`` being the sum of its ``candidate_topk`` largest IoUs (at least
    1), and a prior taken by several boxes goes to the cheapest one.
    """

    def __init__(
        self,
        center_radius: float = 1.5,
        candidate_topk: int = 10,
        iou_weight: float = 3.0,
    ) -> None:
        """Initialize the assigner.

        Args:
            center_radius: Half side of the candidate square around a box center, in strides.
            candidate_topk: Number of largest IoUs summed into the dynamic ``k`` of a box.
            iou_weight: Weight of the IoU cost.
        """
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def assign(
        self,
        pred_boxes: Float32[Tensor, "num_priors 4"],
        pred_objectness_logits: Float32[Tensor, "num_priors 1"],
        pred_class_logits: Float32[Tensor, "num_priors num_classes"],
        priors: Float32[Tensor, "num_priors 3"],
        gt_boxes: Float32[Tensor, "num_gt 4"],
        gt_labels: Int64[Tensor, " num_gt"],
    ) -> SimOTAAssignment:
        """Assign the priors of one image.

        Args:
            pred_boxes: Decoded predicted boxes ``(cx, cy, w, h)`` in pixels.
            pred_objectness_logits: Objectness logit of every prior.
            pred_class_logits: Class logits of every prior.
            priors: ``(grid_x, grid_y, stride)`` of every prior.
            gt_boxes: Ground truth boxes ``(cx, cy, w, h)`` in pixels.
            gt_labels: Class index of every ground truth box.

        Returns:
            The foreground priors, their ground truth boxes and their IoUs.
        """
        num_priors = pred_boxes.shape[0]
        num_gt = gt_boxes.shape[0]
        device = pred_boxes.device
        empty = SimOTAAssignment(
            foreground_mask=torch.zeros(num_priors, dtype=torch.bool, device=device),
            matched_gt_indices=torch.zeros(0, dtype=torch.long, device=device),
            matched_ious=torch.zeros(0, dtype=torch.float32, device=device),
        )
        if num_gt == 0:
            return empty

        strides = priors[:, 2]
        centers_x = (priors[:, 0] + 0.5) * strides
        centers_y = (priors[:, 1] + 0.5) * strides
        center_dist = strides * self.center_radius
        deltas = torch.stack(
            [
                centers_x[None] - (gt_boxes[:, 0:1] - center_dist[None]),
                (gt_boxes[:, 0:1] + center_dist[None]) - centers_x[None],
                centers_y[None] - (gt_boxes[:, 1:2] - center_dist[None]),
                (gt_boxes[:, 1:2] + center_dist[None]) - centers_y[None],
            ],
            dim=-1,
        )
        is_in_centers = deltas.min(dim=-1).values > 0.0
        candidate_mask = is_in_centers.any(dim=0)
        num_candidates = int(candidate_mask.sum())
        if num_candidates == 0:
            return empty
        geometry_relation = is_in_centers[:, candidate_mask]

        pair_ious = BaseBBoxes2D.from_cxcywh(gt_boxes).iou(
            BaseBBoxes2D.from_cxcywh(pred_boxes[candidate_mask]), eps=1e-16
        )
        iou_cost = -torch.log(pair_ious + 1e-8)

        num_classes = pred_class_logits.shape[1]
        gt_one_hot = F.one_hot(gt_labels, num_classes).to(torch.float32)
        scores = (
            pred_class_logits[candidate_mask].sigmoid()
            * pred_objectness_logits[candidate_mask].sigmoid()
        ).sqrt()
        class_cost = F.binary_cross_entropy(
            scores[None].expand(num_gt, -1, -1),
            gt_one_hot[:, None].expand(-1, num_candidates, -1),
            reduction="none",
        ).sum(dim=-1)

        cost = class_cost + self.iou_weight * iou_cost + 1e6 * (~geometry_relation)

        matching = torch.zeros_like(cost, dtype=torch.bool)
        topk_ious = pair_ious.topk(min(self.candidate_topk, num_candidates), dim=1).values
        dynamic_ks = topk_ious.sum(dim=1).int().clamp(min=1)
        for gt_index in range(num_gt):
            positions = cost[gt_index].topk(int(dynamic_ks[gt_index]), largest=False).indices
            matching[gt_index, positions] = True

        multiple = matching.sum(dim=0) > 1
        if multiple.any():
            cheapest = cost[:, multiple].argmin(dim=0)
            matching[:, multiple] = False
            matching[cheapest, multiple] = True

        foreground_candidates = matching.any(dim=0)
        foreground_mask = candidate_mask.clone()
        foreground_mask[candidate_mask] = foreground_candidates
        matched_gt_indices = matching[:, foreground_candidates].to(torch.uint8).argmax(dim=0)
        matched_ious = (matching * pair_ious).sum(dim=0)[foreground_candidates]
        return SimOTAAssignment(
            foreground_mask=foreground_mask,
            matched_gt_indices=matched_gt_indices,
            matched_ious=matched_ious,
        )
