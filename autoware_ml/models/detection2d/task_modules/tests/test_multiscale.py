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

"""Unit tests for the batch-level multi-scale resize."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection2d.task_modules.multiscale import BatchMultiScaleResize
from autoware_ml.models.detection2d.tests.yolox_fixtures import build_batch_inputs


class TestBatchMultiScaleResize(unittest.TestCase):
    """Unit tests for the size schedule and the resize of a batch."""

    def setUp(self) -> None:
        """Set up a batch of two 64x64 images, one with two boxes and one without."""
        torch.manual_seed(0)
        self.batch = build_batch_inputs(
            images=torch.rand(2, 3, 64, 64) * 255.0,
            gt_boxes=[
                torch.tensor([[8.0, 8.0, 40.0, 40.0], [30.0, 10.0, 50.0, 28.0]]),
                torch.zeros(0, 4),
            ],
            gt_labels=[torch.tensor([0, 1]), torch.zeros(0, dtype=torch.long)],
            scale=0.25,
        )

    def test_sizes_are_the_multiples_of_the_divisor_within_the_range(self) -> None:
        """Test that the range is cut to the multiples of the divisor, both ends included."""
        self.assertEqual(
            BatchMultiScaleResize(size_range=[800, 1280]).sizes, tuple(range(800, 1281, 32))
        )
        self.assertEqual(
            BatchMultiScaleResize(size_range=[790, 900], size_divisor=32).sizes,
            (800, 832, 864, 896),
        )

    def test_size_follows_the_step_and_the_seed_only(self) -> None:
        """Test the sequence of sizes.

        Check: a size holds for an interval of steps, two instances of the same seed draw the
        same sequence whatever the global random state, another seed draws another one, and
        over many intervals every size of the range is drawn.
        """
        resize = BatchMultiScaleResize(size_range=[800, 1280], interval=10, seed=3)
        sizes = [resize.size_at(step) for step in range(0, 5000)]
        torch.manual_seed(123)
        same_seed = BatchMultiScaleResize(size_range=[800, 1280], interval=10, seed=3)
        other_seed = BatchMultiScaleResize(size_range=[800, 1280], interval=10, seed=4)

        self.assertEqual(sizes, [same_seed.size_at(step) for step in range(0, 5000)])
        self.assertNotEqual(sizes, [other_seed.size_at(step) for step in range(0, 5000)])
        for start in range(0, 5000, 10):
            self.assertEqual(len(set(sizes[start : start + 10])), 1)
        self.assertEqual(set(sizes), set(range(800, 1281, 32)))

    def test_batch_is_resized_with_its_boxes(self) -> None:
        """Test that the images, the boxes and the augmentation matrices scale together.

        Setup: a 64x64 batch resized to the one size of the range, 96.
        Check: the images are the bilinear resize of the input, every box coordinate is 1.5
        times as large, padding rows and labels are untouched, the matrices map raw pixels
        onto the resized image, and the input batch is not modified.
        """
        resize = BatchMultiScaleResize(size_range=[96, 96])
        images = self.batch.image_data.images.clone()
        boxes = self.batch.multi_task_gt_batch.detection2d_gt_batch.gt_bboxes_2d.clone()

        resized = resize(self.batch, step=0)

        expected_images = torch.nn.functional.interpolate(
            images[:, 0], size=(96, 96), mode="bilinear", align_corners=False
        )
        gt_batch = resized.multi_task_gt_batch.detection2d_gt_batch
        self.assertEqual(tuple(resized.image_data.images.shape), (2, 1, 3, 96, 96))
        torch.testing.assert_close(resized.image_data.images[:, 0], expected_images)
        self.assertIs(resized.multi_task_gt_batch.image_gt_batch, resized.image_data)
        torch.testing.assert_close(gt_batch.gt_bboxes_2d, boxes * 1.5)
        torch.testing.assert_close(
            gt_batch.gt_bboxes_2d[0, 0, 0, :4], torch.tensor([12.0, 12.0, 60.0, 60.0])
        )
        self.assertEqual(float(gt_batch.gt_bboxes_2d[1].abs().max()), 0.0)
        self.assertEqual(gt_batch.gt_valid_bboxes_2d.tolist(), [[2], [0]])
        self.assertEqual(gt_batch.gt_labels_2d[0, 0, :3].tolist(), [0, 1, -1])
        matrices = resized.image_data.image_augmentation_matrices
        self.assertAlmostEqual(float(matrices[0, 0, 0, 0]), 0.375)
        self.assertAlmostEqual(float(matrices[0, 0, 1, 1]), 0.375)
        self.assertEqual(float(matrices[0, 0, 2, 2]), 1.0)
        torch.testing.assert_close(self.batch.image_data.images, images)
        torch.testing.assert_close(
            self.batch.multi_task_gt_batch.detection2d_gt_batch.gt_bboxes_2d, boxes
        )

    def test_each_axis_of_a_rectangular_batch_has_its_own_factor(self) -> None:
        """Test that a 32x64 batch resized to 64x64 doubles the y coordinates only."""
        batch = build_batch_inputs(
            images=torch.rand(1, 3, 32, 64),
            gt_boxes=[torch.tensor([[8.0, 4.0, 40.0, 20.0]])],
            gt_labels=[torch.tensor([0])],
        )

        resized = BatchMultiScaleResize(size_range=[64, 64])(batch, step=0)

        torch.testing.assert_close(
            resized.multi_task_gt_batch.detection2d_gt_batch.gt_bboxes_2d[0, 0, 0, :4],
            torch.tensor([8.0, 8.0, 40.0, 40.0]),
        )

    def test_batch_at_the_drawn_size_is_returned_as_it_is(self) -> None:
        """Test that a batch that already has the size of the step is not copied."""
        self.assertIs(BatchMultiScaleResize(size_range=[64, 64])(self.batch, step=0), self.batch)

    def test_invalid_configuration_is_refused(self) -> None:
        """Test that a range without a multiple of the divisor and bad values are refused."""
        for arguments in (
            {"size_range": [801, 831]},
            {"size_range": [1280, 800]},
            {"size_range": [0, 64]},
            {"size_range": [800]},
            {"size_range": [800, 1280], "size_divisor": 0},
            {"size_range": [800, 1280], "interval": 0},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                BatchMultiScaleResize(**arguments)
