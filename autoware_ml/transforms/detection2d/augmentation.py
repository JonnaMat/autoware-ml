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

"""Training augmentations of YOLOX for single camera 2D detection.

Ported from the YOLOX reference implementation (``yolox/data/datasets/mosaicdetection.py`` and
``yolox/data/data_augment.py`` of https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0).
Every transform reads and writes the image as RGB float32 in ``[0, 255]`` and the boxes as
``(x_min, y_min, x_max, y_max)`` in pixels of that image.

The reference applies mosaic, the random affine and mixup as one step; here they are three
transforms a pipeline lists in that order. All three take the name of a strong augmentation
switch, which turns them off for the last epochs of a run.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import cv2
import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.base import BaseTransform, random_source
from autoware_ml.transforms.detection2d.multi_sample import (
    MultiSampleTransform,
    SwitchableTransform,
)
from autoware_ml.transforms.detection2d.sample_io import (
    boxes_of,
    clip_boxes,
    image_of,
    replace_image_and_boxes,
)


class Mosaic(MultiSampleTransform):
    """Tile the sample and three further samples around a random center.

    The canvas is twice the network input in both directions. The center is drawn within its
    middle half, and every image is placed with one corner on the center and cut at the
    canvas border. A random affine transform that maps the canvas onto the network input
    follows in the pipeline.

    Required keys:
        - camera_image_data: image no larger than ``image_size``.
        - detection2d_gt_sample: its boxes.
    """

    _required_keys = ["camera_image_data", "detection2d_gt_sample"]

    def __init__(
        self,
        image_size: Sequence[int],
        center_ratio_range: Sequence[float] = (0.5, 1.5),
        pad_value: float = 114.0,
        probability: float | None = None,
        switch_name: str | None = None,
    ) -> None:
        """Initialize the transform.

        Args:
            image_size: ``(height, width)`` of the network input, half the canvas.
            center_ratio_range: Range the mosaic center is drawn from, as a share of
              ``image_size``.
            pad_value: Value of the canvas no image covers.
            probability: Probability of applying the transform, None to always apply it.
            switch_name: Name of the strong augmentation switch that turns the transform off.
        """
        super().__init__(probability=probability, switch_name=switch_name)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.center_ratio_range = (float(center_ratio_range[0]), float(center_ratio_range[1]))
        self.pad_value = pad_value

    @staticmethod
    def _placement(
        position: int,
        center_x: int,
        center_y: int,
        width: int,
        height: int,
        canvas: tuple[int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        """Compute where an image lands on the canvas and which part of it is kept.

        Args:
            position: 0 top left, 1 top right, 2 bottom left, 3 bottom right of the center.
            center_x: X coordinate of the mosaic center.
            center_y: Y coordinate of the mosaic center.
            width: Width of the image.
            height: Height of the image.
            canvas: ``(height, width)`` of the canvas.

        Returns:
            The canvas region and the image region, each ``(x1, y1, x2, y2)``.
        """
        canvas_height, canvas_width = canvas
        if position == 0:
            x1, y1, x2, y2 = max(center_x - width, 0), max(center_y - height, 0), center_x, center_y
            return (x1, y1, x2, y2), (width - (x2 - x1), height - (y2 - y1), width, height)
        if position == 1:
            x1, y1 = center_x, max(center_y - height, 0)
            x2, y2 = min(center_x + width, canvas_width), center_y
            return (x1, y1, x2, y2), (0, height - (y2 - y1), min(width, x2 - x1), height)
        if position == 2:
            x1, y1 = max(center_x - width, 0), center_y
            x2, y2 = center_x, min(canvas_height, center_y + height)
            return (x1, y1, x2, y2), (width - (x2 - x1), 0, width, min(y2 - y1, height))
        x1, y1 = center_x, center_y
        x2, y2 = min(center_x + width, canvas_width), min(canvas_height, center_y + height)
        return (x1, y1, x2, y2), (0, 0, min(width, x2 - x1), min(y2 - y1, height))

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Build the mosaic of the sample and three samples drawn from the dataset.

        Args:
            model_gt_sample: Loaded sample.

        Returns:
            The sample holding the mosaic canvas and the boxes of the four images.
        """
        rng = random_source(None)
        height, width = self.image_size
        canvas_size = (2 * height, 2 * width)
        center_y = int(rng.uniform(*self.center_ratio_range) * height)
        center_x = int(rng.uniform(*self.center_ratio_range) * width)
        canvas = np.full((*canvas_size, 3), self.pad_value, dtype=np.float32)

        all_boxes = []
        all_labels = []
        primary_offset = (0, 0)
        samples = [model_gt_sample] + [self.draw_sample() for _ in range(3)]
        for position, sample in enumerate(samples):
            image = image_of(sample)
            boxes, labels = boxes_of(sample)
            (x1, y1, x2, y2), (sx1, sy1, sx2, sy2) = self._placement(
                position, center_x, center_y, image.shape[1], image.shape[0], canvas_size
            )
            canvas[y1:y2, x1:x2] = image[sy1:sy2, sx1:sx2]
            offset = (x1 - sx1, y1 - sy1)
            if position == 0:
                primary_offset = offset
            boxes[:, 0::2] += offset[0]
            boxes[:, 1::2] += offset[1]
            all_boxes.append(boxes)
            all_labels.append(labels)

        boxes, labels = clip_boxes(
            np.concatenate(all_boxes), np.concatenate(all_labels), canvas_size[1], canvas_size[0]
        )
        pixel_affine = np.eye(3)
        pixel_affine[:2, 2] = primary_offset
        return replace_image_and_boxes(model_gt_sample, canvas, boxes, labels, pixel_affine)


class RandomAffine(SwitchableTransform):
    """Rotate, scale, shear and shift the image about its top left corner onto the input size.

    After a mosaic the image is a canvas of twice the input size, of which the transform keeps
    a window of the input size: a scale of 0.5 shows the whole canvas, a larger one zooms in.

    Required keys:
        - camera_image_data: image to transform.
        - detection2d_gt_sample: its boxes.
    """

    _required_keys = ["camera_image_data", "detection2d_gt_sample"]

    def __init__(
        self,
        image_size: Sequence[int],
        max_rotation_degrees: float = 10.0,
        max_translation_ratio: float = 0.1,
        scale_range: Sequence[float] = (0.1, 2.0),
        max_shear_degrees: float = 2.0,
        pad_value: float = 114.0,
        probability: float | None = None,
        switch_name: str | None = None,
    ) -> None:
        """Initialize the transform.

        Args:
            image_size: ``(height, width)`` of the output image.
            max_rotation_degrees: Largest rotation, drawn from both directions.
            max_translation_ratio: Largest shift as a share of the output size.
            scale_range: Range the scale is drawn from.
            max_shear_degrees: Largest shear angle along each axis.
            pad_value: Value of the output pixels the image does not cover.
            probability: Probability of applying the transform, None to always apply it.
            switch_name: Name of the strong augmentation switch that turns the transform off.
        """
        super().__init__(probability=probability, switch_name=switch_name)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_rotation_degrees = max_rotation_degrees
        self.max_translation_ratio = max_translation_ratio
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.max_shear_degrees = max_shear_degrees
        self.pad_value = pad_value

    def sample_matrix(self) -> np.ndarray:
        """Draw the 2x3 affine matrix of one sample."""
        rng = random_source(None)
        height, width = self.image_size
        angle = rng.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        scale = rng.uniform(*self.scale_range)
        rotation = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
        shear_x = math.tan(
            math.radians(rng.uniform(-self.max_shear_degrees, self.max_shear_degrees))
        )
        shear_y = math.tan(
            math.radians(rng.uniform(-self.max_shear_degrees, self.max_shear_degrees))
        )
        matrix = np.zeros((2, 3), dtype=np.float64)
        matrix[0] = rotation[0] + shear_y * rotation[1]
        matrix[1] = rotation[1] + shear_x * rotation[0]
        matrix[0, 2] = rng.uniform(-self.max_translation_ratio, self.max_translation_ratio) * width
        matrix[1, 2] = rng.uniform(-self.max_translation_ratio, self.max_translation_ratio) * height
        return matrix

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Warp the image and move its boxes along.

        A box becomes the axis aligned extent of its four transformed corners, clipped to the
        output image.

        Args:
            model_gt_sample: Loaded sample.

        Returns:
            The sample holding the warped image of the output size and its boxes.
        """
        height, width = self.image_size
        matrix = self.sample_matrix()
        image = cv2.warpAffine(
            image_of(model_gt_sample),
            matrix,
            dsize=(width, height),
            borderValue=(self.pad_value, self.pad_value, self.pad_value),
        )
        boxes, labels = boxes_of(model_gt_sample)
        if len(boxes):
            corners = np.ones((4 * len(boxes), 3), dtype=np.float64)
            corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * len(boxes), 2)
            corners = (corners @ matrix.T).reshape(len(boxes), 8)
            xs, ys = corners[:, 0::2], corners[:, 1::2]
            boxes = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1).astype(
                np.float32
            )
        boxes, labels = clip_boxes(boxes, labels, width, height)
        return replace_image_and_boxes(
            model_gt_sample, image, boxes, labels, np.vstack([matrix, [0.0, 0.0, 1.0]])
        )


class MixUp(MultiSampleTransform):
    """Blend the image half and half with a randomly scaled and flipped further sample.

    The partner is padded to the network input, scaled, flipped with a probability of one
    half and cropped (or padded) to the size of the image at a random offset. Its boxes join
    those of the sample. A partner with at least one box is looked for a few times; when none
    is found the sample is left as it is.

    Required keys:
        - camera_image_data: image to blend into.
        - detection2d_gt_sample: its boxes.
    """

    _required_keys = ["camera_image_data", "detection2d_gt_sample"]
    _PARTNER_DRAWS = 10

    def __init__(
        self,
        image_size: Sequence[int],
        scale_range: Sequence[float] = (0.5, 1.5),
        pad_value: float = 114.0,
        probability: float | None = None,
        switch_name: str | None = None,
    ) -> None:
        """Initialize the transform.

        Args:
            image_size: ``(height, width)`` of the network input the partner is padded to.
            scale_range: Range the scale of the partner is drawn from.
            pad_value: Value of the padding of the partner.
            probability: Probability of applying the transform, None to always apply it.
            switch_name: Name of the strong augmentation switch that turns the transform off.
        """
        super().__init__(probability=probability, switch_name=switch_name)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.pad_value = pad_value

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Blend a further sample into the sample.

        Args:
            model_gt_sample: Loaded sample.

        Returns:
            The sample holding the blended image and the boxes of both images.
        """
        rng = random_source(None)
        partner = None
        for _ in range(self._PARTNER_DRAWS):
            candidate = self.draw_sample()
            if len(boxes_of(candidate)[0]):
                partner = candidate
                break
        if partner is None:
            return model_gt_sample

        scale = rng.uniform(*self.scale_range)
        flip = rng.rand() > 0.5
        partner_image = image_of(partner)
        partner_boxes, partner_labels = boxes_of(partner)

        padded = np.full((*self.image_size, 3), self.pad_value, dtype=np.float32)
        padded[: partner_image.shape[0], : partner_image.shape[1]] = partner_image
        padded = cv2.resize(
            padded,
            (int(padded.shape[1] * scale), int(padded.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        partner_boxes *= scale
        scaled_height, scaled_width = padded.shape[:2]
        if flip:
            padded = padded[:, ::-1]
            partner_boxes[:, 0::2] = scaled_width - partner_boxes[:, [2, 0]]

        image = image_of(model_gt_sample)
        target_height, target_width = image.shape[:2]
        mixed = np.zeros(
            (max(scaled_height, target_height), max(scaled_width, target_width), 3),
            dtype=np.float32,
        )
        mixed[:scaled_height, :scaled_width] = padded
        offset_y = (
            rng.randint(0, mixed.shape[0] - target_height) if mixed.shape[0] > target_height else 0
        )
        offset_x = (
            rng.randint(0, mixed.shape[1] - target_width) if mixed.shape[1] > target_width else 0
        )
        mixed = mixed[offset_y : offset_y + target_height, offset_x : offset_x + target_width]

        partner_boxes[:, 0::2] -= offset_x
        partner_boxes[:, 1::2] -= offset_y
        partner_boxes, partner_labels = clip_boxes(
            partner_boxes, partner_labels, target_width, target_height
        )
        boxes, labels = boxes_of(model_gt_sample)
        return replace_image_and_boxes(
            model_gt_sample,
            0.5 * image + 0.5 * mixed,
            np.concatenate([boxes, partner_boxes]),
            np.concatenate([labels, partner_labels]),
        )


class HSVAugmentation(BaseTransform):
    """Shift hue, saturation and value by random amounts, each with a probability of one half.

    Required keys:
        - camera_image_data: RGB image in ``[0, 255]``.
    """

    _required_keys = ["camera_image_data", "detection2d_gt_sample"]

    def __init__(
        self,
        hue_gain: float = 5.0,
        saturation_gain: float = 30.0,
        value_gain: float = 30.0,
        probability: float | None = None,
    ) -> None:
        """Initialize the transform.

        Args:
            hue_gain: Largest hue shift, in the 8 bit hue of OpenCV that spans ``[0, 180)``.
            saturation_gain: Largest saturation shift out of 255.
            value_gain: Largest value shift out of 255.
            probability: Probability of applying the transform, None to always apply it.
        """
        super().__init__(probability=probability)
        self.gains = np.array([hue_gain, saturation_gain, value_gain], dtype=np.float64)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Shift the colors of the image.

        Args:
            model_gt_sample: Loaded sample.

        Returns:
            The sample holding the color shifted image, its boxes unchanged.
        """
        rng = random_source(None)
        shifts = rng.uniform(-1, 1, 3) * self.gains
        shifts = (shifts * rng.randint(0, 2, 3)).astype(np.int16)
        image = np.clip(np.rint(image_of(model_gt_sample)), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + shifts[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + shifts[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + shifts[2], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        boxes, labels = boxes_of(model_gt_sample)
        return replace_image_and_boxes(model_gt_sample, image, boxes, labels)


class RandomHorizontalFlip(BaseTransform):
    """Mirror the image and its boxes left to right.

    Required keys:
        - camera_image_data: image to mirror.
        - detection2d_gt_sample: its boxes.
    """

    _required_keys = ["camera_image_data", "detection2d_gt_sample"]

    def __init__(self, probability: float = 0.5) -> None:
        """Initialize the transform.

        Args:
            probability: Probability of mirroring a sample.
        """
        super().__init__(probability=probability)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Mirror the sample.

        Args:
            model_gt_sample: Loaded sample.

        Returns:
            The mirrored sample.
        """
        image = image_of(model_gt_sample)
        width = image.shape[1]
        boxes, labels = boxes_of(model_gt_sample)
        boxes[:, 0::2] = width - boxes[:, [2, 0]]
        pixel_affine = np.array([[-1.0, 0.0, width], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return replace_image_and_boxes(model_gt_sample, image[:, ::-1], boxes, labels, pixel_affine)
