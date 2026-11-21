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

"""Batch-level multi-scale training of a 2D detector.

Follows the multi-scale training of the YOLOX reference implementation (``random_resize`` and
``preprocess`` of ``yolox/exp/yolox_base.py`` of https://github.com/Megvii-BaseDetection/YOLOX,
Apache-2.0): every few optimizer steps a new input size is drawn and the whole training batch
is resized to it, the boxes with it.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import torch.nn.functional as F

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


class BatchMultiScaleResize:
    """Resize a training batch to a square size that changes every few optimizer steps.

    The size of a step is a function of the step and the seed only, not of the state of a
    random number generator: every rank of a distributed run resizes to the same size, and a
    resumed run continues the sequence of sizes. The images are resized with bilinear
    interpolation, each axis by its own factor, and the 2D boxes and the image augmentation
    matrices are scaled by the same factors. Other image-space data of the batch is left as
    it is.
    """

    def __init__(
        self,
        size_range: Sequence[int],
        size_divisor: int = 32,
        interval: int = 10,
        enabled: bool = True,
        keep_in_no_aug_epochs: bool = True,
        seed: int = 0,
    ) -> None:
        """Initialize the resize.

        Args:
            size_range: Smallest and largest side of the resized batch in pixels, both
              included.
            size_divisor: Number every drawn size is a multiple of, the largest stride of the
              detector or a multiple of it.
            interval: Number of optimizer steps a drawn size is kept for.
            enabled: Whether training batches are resized at all.
            keep_in_no_aug_epochs: Whether the batches are still resized in the last epochs
              of a run, trained without strong augmentation. False trains those epochs at
              the size the data pipeline serves.
            seed: Seed of the sequence of sizes.

        Raises:
            ValueError: If the divisor or the interval is not positive, or the range is not
              two positive sizes in ascending order with a multiple of the divisor between
              them.
        """
        if size_divisor < 1:
            raise ValueError(f"size_divisor must be positive, got {size_divisor}.")
        if interval < 1:
            raise ValueError(f"interval must be positive, got {interval}.")
        if len(size_range) != 2 or not 0 < size_range[0] <= size_range[1]:
            raise ValueError(
                f"size_range must be two positive sizes in ascending order, got {size_range}."
            )
        first = -(-size_range[0] // size_divisor)
        last = size_range[1] // size_divisor
        if first > last:
            raise ValueError(
                f"size_range {list(size_range)} holds no multiple of size_divisor {size_divisor}."
            )
        self.sizes = tuple(index * size_divisor for index in range(first, last + 1))
        self.interval = interval
        self.enabled = enabled
        self.keep_in_no_aug_epochs = keep_in_no_aug_epochs
        self.seed = seed

    def size_at(self, step: int) -> int:
        """Give the size the batch of an optimizer step is resized to.

        Args:
            step: Number of optimizer steps taken before the batch.

        Returns:
            One of the multiples of the divisor within the range, each equally likely, the
            same for every step of an interval.
        """
        digest = hashlib.sha256(f"{self.seed}:{step // self.interval}".encode()).digest()
        return self.sizes[int.from_bytes(digest[:8], "big") % len(self.sizes)]

    def __call__(self, batch_inputs: ModelBatchInputs, step: int) -> ModelBatchInputs:
        """Resize the images and the 2D boxes of a batch to the size of a step.

        Args:
            batch_inputs: Batch after the runtime preprocessing.
            step: Number of optimizer steps taken before the batch.

        Returns:
            The batch at the size of the step, the batch itself when it has that size already.

        Raises:
            ValueError: If the batch carries no images.
        """
        image_data = batch_inputs.image_data
        if image_data is None:
            raise ValueError("Multi-scale training needs the images of the batch.")
        size = self.size_at(step)
        images = image_data.images
        height, width = images.shape[-2:]
        if (height, width) == (size, size):
            return batch_inputs
        scale_x, scale_y = size / width, size / height

        resized = F.interpolate(
            images.flatten(0, 1), size=(size, size), mode="bilinear", align_corners=False
        ).reshape(*images.shape[:3], size, size)
        # The matrices map raw pixels onto the served image: the resize scales their x and
        # y rows.
        matrices = image_data.image_augmentation_matrices.clone()
        matrices[..., 0, :] *= scale_x
        matrices[..., 1, :] *= scale_y
        image_data = image_data._replace(images=resized, image_augmentation_matrices=matrices)

        gt_batch = batch_inputs.multi_task_gt_batch
        detection2d = gt_batch.detection2d_gt_batch
        if detection2d is not None:
            # Every field of a box is an x or a y coordinate in pixels, alternating.
            boxes = detection2d.gt_bboxes_2d
            factors = boxes.new_tensor([scale_x, scale_y]).repeat(boxes.shape[-1] // 2)
            detection2d = detection2d._replace(gt_bboxes_2d=boxes * factors)
        return batch_inputs.replace(
            multi_task_gt_batch=gt_batch._replace(
                image_gt_batch=image_data, detection2d_gt_batch=detection2d
            ),
            image_data=image_data,
        )
