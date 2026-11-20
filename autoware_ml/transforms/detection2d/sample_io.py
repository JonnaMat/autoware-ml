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

"""Access to the image and the 2D boxes of a single camera detection sample.

The 2D detection transforms work on the image as a ``(height, width, 3)`` float32 NumPy array
in RGB order and on the boxes as ``(x_min, y_min, x_max, y_max)`` rows, the layout OpenCV
reads. The sample keeps the image as the one camera of a ``BaseImages`` and the boxes as a
``Detection2DGTSample``; the helpers here convert between the two without copying the pixels.
"""

from __future__ import annotations

from jaxtyping import Float32, Int32
import numpy as np
import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.types.geometry import Box2DFieldIndex


def image_of(sample: ModelGTSample) -> Float32[np.ndarray, "height width 3"]:
    """Read the image of the one camera of a sample.

    Args:
        sample: Sample with loaded ``camera_image_data``.

    Returns:
        The image in height, width, channel layout, sharing its memory with the sample when
        the sample stores it that way.

    Raises:
        ValueError: If the sample carries no image or more than one camera.
    """
    data = sample.camera_image_data
    if data is None or data.images.shape[0] != 1:
        raise ValueError("A 2D detection transform reads the image of exactly one camera.")
    return np.ascontiguousarray(data.images[0].permute(1, 2, 0).numpy())


def boxes_of(
    sample: ModelGTSample,
) -> tuple[Float32[np.ndarray, "num_boxes 4"], Int32[np.ndarray, " num_boxes"]]:
    """Read the 2D boxes and the class indices of the one camera of a sample.

    Args:
        sample: Sample with a ``detection2d_gt_sample``.

    Returns:
        Copies of the boxes ``(x_min, y_min, x_max, y_max)`` and of their class indices.

    Raises:
        ValueError: If the sample carries no 2D boxes or those of more than one camera.
    """
    gt_sample = sample.detection2d_gt_sample
    if gt_sample is None or len(gt_sample.gt_bboxes_2d) != 1:
        raise ValueError("A 2D detection transform reads the boxes of exactly one camera.")
    boxes = gt_sample.gt_bboxes_2d[0][:, :4].numpy().astype(np.float32, copy=True)
    labels = gt_sample.gt_labels_2d[0].numpy().astype(np.int32, copy=True)
    return boxes, labels


def detection2d_gt_sample(
    boxes: Float32[np.ndarray, "num_boxes 4"], labels: Int32[np.ndarray, " num_boxes"]
) -> Detection2DGTSample:
    """Build the 2D ground truth of a single camera sample.

    Args:
        boxes: Boxes ``(x_min, y_min, x_max, y_max)``.
        labels: Class index of every box.

    Returns:
        The ground truth laid out by Box2DFieldIndex, the object center being the box center.
    """
    boxes_tensor = torch.from_numpy(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
    boxes_with_centers = torch.zeros(
        (boxes_tensor.shape[0], len(Box2DFieldIndex)), dtype=torch.float32
    )
    boxes_with_centers[:, :4] = boxes_tensor
    boxes_with_centers[:, Box2DFieldIndex.CENTER_X] = (boxes_tensor[:, 0] + boxes_tensor[:, 2]) / 2
    boxes_with_centers[:, Box2DFieldIndex.CENTER_Y] = (boxes_tensor[:, 1] + boxes_tensor[:, 3]) / 2
    return Detection2DGTSample(
        gt_bboxes_2d=[boxes_with_centers],
        gt_labels_2d=[torch.from_numpy(np.asarray(labels, dtype=np.int32).reshape(-1))],
    )


def replace_image_and_boxes(
    sample: ModelGTSample,
    image: Float32[np.ndarray, "height width 3"],
    boxes: Float32[np.ndarray, "num_boxes 4"],
    labels: Int32[np.ndarray, " num_boxes"],
    pixel_affine: np.ndarray | None = None,
) -> ModelGTSample:
    """Return the sample with a new image and new boxes.

    Args:
        sample: Sample with loaded ``camera_image_data``.
        image: New image in height, width, channel layout.
        boxes: New boxes ``(x_min, y_min, x_max, y_max)`` in pixels of the new image.
        labels: Class index of every new box.
        pixel_affine: 3x3 affine mapping a pixel of the old image onto the new one, composed
          into the image augmentation of the sample. None for a transform that moves no pixel.

    Returns:
        The updated sample.
    """
    data = sample.camera_image_data
    update: dict[str, torch.Tensor] = {
        # Kept in height, width, channel memory order below the channel first view, so the
        # next transform reads it back without a copy.
        "images": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
    }
    if pixel_affine is not None:
        affine = torch.from_numpy(np.asarray(pixel_affine, dtype=np.float32)).unsqueeze(0)
        update["augmented_camera_intrinsics"] = affine @ data.augmented_camera_intrinsics
        update["image_augmentation_matrices"] = (
            BaseImages.homogeneous_image_augmentation_matrices(affine)
            @ data.image_augmentation_matrices
        )
    return sample._replace(
        camera_image_data=BaseImages.model_validate(data.model_copy(update=update)),
        detection2d_gt_sample=detection2d_gt_sample(boxes, labels),
    )


def clip_boxes(
    boxes: Float32[np.ndarray, "num_boxes 4"],
    labels: Int32[np.ndarray, " num_boxes"],
    width: float,
    height: float,
) -> tuple[Float32[np.ndarray, "num_kept 4"], Int32[np.ndarray, " num_kept"]]:
    """Clip boxes to an image and drop those left without an area.

    Args:
        boxes: Boxes ``(x_min, y_min, x_max, y_max)``.
        labels: Class index of every box.
        width: Width of the image.
        height: Height of the image.

    Returns:
        The clipped boxes that still have a width and a height, with their class indices.
    """
    boxes = boxes.copy()
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], labels[keep]
