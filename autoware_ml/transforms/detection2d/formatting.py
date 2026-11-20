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

"""Final steps of the 2D detection pipelines: letterboxing and the box filter."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.detection2d.sample_io import (
    boxes_of,
    image_of,
    replace_image_and_boxes,
)


class Letterbox(BaseTransform):
    """Fit the image into the network input with its aspect ratio kept and pad the rest.

    The image is resized so it fits ``image_size``, placed at the top left corner and padded
    to the right and to the bottom, the letterbox the YOLOX runtimes apply. An image that
    already fits tightly is only padded.

    Required keys:
        - camera_image_data: image to letterbox.
        - detection2d_gt_sample: its boxes.
    """

    _required_keys = ["camera_image_data", "detection2d_gt_sample"]

    def __init__(self, image_size: Sequence[int], pad_value: float = 114.0) -> None:
        """Initialize the transform.

        Args:
            image_size: ``(height, width)`` of the network input.
            pad_value: Value of the padding.
        """
        super().__init__(probability=None)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.pad_value = pad_value

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Letterbox the image of the sample.

        Args:
            model_gt_sample: Loaded sample.

        Returns:
            The sample holding an image of exactly ``image_size`` and its boxes scaled along.
        """
        image = image_of(model_gt_sample)
        boxes, labels = boxes_of(model_gt_sample)
        height, width = image.shape[:2]
        ratio = min(self.image_size[0] / height, self.image_size[1] / width)
        new_width, new_height = int(width * ratio), int(height * ratio)
        scale_x, scale_y = 1.0, 1.0
        if (new_height, new_width) != (height, width):
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            scale_x, scale_y = new_width / width, new_height / height
            boxes *= np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        padded = np.full((*self.image_size, 3), self.pad_value, dtype=np.float32)
        padded[:new_height, :new_width] = image
        return replace_image_and_boxes(
            model_gt_sample, padded, boxes, labels, np.diag([scale_x, scale_y, 1.0])
        )


class FilterSmallBoxes2D(BaseTransform):
    """Drop the boxes an augmentation left too small to be a target.

    A box is kept when its shorter side is at least ``min_side`` pixels of the image the
    network reads. The threshold is on the side and not on the area, so a small but well
    formed box survives while a sliver cut off by the image border does not.

    Required keys:
        - camera_image_data: image the boxes are measured in.
        - detection2d_gt_sample: boxes to filter.
    """

    _required_keys = ["camera_image_data", "detection2d_gt_sample"]

    def __init__(self, min_side: float = 2.0) -> None:
        """Initialize the transform.

        Args:
            min_side: Shortest side in pixels a kept box may have.
        """
        super().__init__(probability=None)
        self.min_side = min_side

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Drop the boxes whose shorter side is below the threshold.

        Args:
            model_gt_sample: Loaded sample.

        Returns:
            The sample holding the kept boxes, its image unchanged.
        """
        boxes, labels = boxes_of(model_gt_sample)
        keep = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= self.min_side
        return replace_image_and_boxes(
            model_gt_sample, image_of(model_gt_sample), boxes[keep], labels[keep]
        )
