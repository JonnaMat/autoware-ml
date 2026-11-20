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

"""Unit tests for the SimOTA assignment on hand-built predictions."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection2d.task_modules.simota import SimOTAAssigner


def _grid_priors(size: int, stride: int) -> torch.Tensor:
    """Priors of one feature level of ``size`` by ``size`` cells."""
    grid_y, grid_x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y, torch.full_like(grid_x, float(stride))], dim=-1).reshape(
        -1, 3
    )


class TestSimOTAAssigner(unittest.TestCase):
    """Unit tests for the SimOTA assigner."""

    def setUp(self) -> None:
        """Set up an 8x8 grid of stride 8 whose priors predict a 16 px box around their center."""
        self.priors = _grid_priors(size=8, stride=8)
        centers = (self.priors[:, :2] + 0.5) * 8
        self.pred_boxes = torch.cat([centers, torch.full((64, 2), 16.0)], dim=1)
        self.objectness = torch.zeros(64, 1)
        self.class_logits = torch.zeros(64, 2)
        self.assigner = SimOTAAssigner(center_radius=1.5, candidate_topk=10, iou_weight=3.0)

    def test_no_ground_truth_assigns_nothing(self) -> None:
        """Test that an image without ground truth has no foreground prior."""
        assignment = self.assigner.assign(
            self.pred_boxes,
            self.objectness,
            self.class_logits,
            self.priors,
            torch.zeros(0, 4),
            torch.zeros(0, dtype=torch.long),
        )

        self.assertFalse(assignment.foreground_mask.any())
        self.assertEqual(assignment.matched_gt_indices.numel(), 0)

    def test_best_overlapping_prior_is_assigned(self) -> None:
        """Test that a box takes the prior predicting it exactly, and only priors near its center.

        Setup: one 16 px box centered on the prior of cell (2, 3), whose prediction matches it
        with an IoU of 1. The dynamic k is the sum of the ten largest IoUs.
        Check: the matched prior is foreground with IoU 1, every foreground prior lies within
        1.5 strides of the box center, and their number is the dynamic k.
        """
        gt_boxes = torch.tensor([[2.5 * 8, 3.5 * 8, 16.0, 16.0]])
        gt_labels = torch.tensor([1])

        assignment = self.assigner.assign(
            self.pred_boxes, self.objectness, self.class_logits, self.priors, gt_boxes, gt_labels
        )

        exact = 3 * 8 + 2
        self.assertTrue(assignment.foreground_mask[exact])
        centers = (self.priors[:, :2] + 0.5) * 8
        offsets = (centers[assignment.foreground_mask] - gt_boxes[0, :2]).abs()
        self.assertTrue((offsets < 1.5 * 8).all())
        # IoUs around the center: 1 at the center, 1/3 for the four neighbors, 1/7 for the
        # four diagonals, which sum to 2.9 and floor to a dynamic k of 2.
        self.assertEqual(int(assignment.foreground_mask.sum()), 2)
        self.assertAlmostEqual(float(assignment.matched_ious.max()), 1.0, places=5)
        self.assertTrue((assignment.matched_gt_indices == 0).all())

    def test_prior_claimed_twice_goes_to_the_cheaper_box(self) -> None:
        """Test that a prior two boxes select is assigned to one of them only.

        Setup: two identical boxes of different classes on the same cell, and class logits of
        the priors that favor class 0.
        Check: both boxes select the same priors, and every one of them goes to the box of
        class 0, whose classification cost is lower, so no prior is assigned twice.
        """
        gt_boxes = torch.tensor([[2.5 * 8, 3.5 * 8, 16.0, 16.0], [2.5 * 8, 3.5 * 8, 16.0, 16.0]])
        gt_labels = torch.tensor([1, 0])
        class_logits = torch.tensor([[4.0, -4.0]]).repeat(64, 1)

        assignment = self.assigner.assign(
            self.pred_boxes, self.objectness, class_logits, self.priors, gt_boxes, gt_labels
        )

        exact = 3 * 8 + 2
        foreground_indices = assignment.foreground_mask.nonzero().flatten().tolist()
        matched = dict(zip(foreground_indices, assignment.matched_gt_indices.tolist(), strict=True))
        self.assertEqual(matched[exact], 1)
        self.assertEqual(len(foreground_indices), len(set(foreground_indices)))
        self.assertEqual(set(matched.values()), {1})

    def test_box_without_a_center_prior_is_left_unassigned(self) -> None:
        """Test that a box whose center lies far outside the grid gets no prior."""
        gt_boxes = torch.tensor([[500.0, 500.0, 16.0, 16.0]])

        assignment = self.assigner.assign(
            self.pred_boxes,
            self.objectness,
            self.class_logits,
            self.priors,
            gt_boxes,
            torch.tensor([0]),
        )

        self.assertFalse(assignment.foreground_mask.any())
