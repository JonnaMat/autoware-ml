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

"""Samples for the 2D detection transform tests: images whose boxes are painted as markers.

Every box of a sample is a filled rectangle of its own color on a black image, so a test can
find where a transform moved the pixels of a box and compare it with where it moved the box.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.detection2d.sample_io import (
    boxes_of,
    detection2d_gt_sample,
    image_of,
    replace_image_and_boxes,
)

MARKER_COLORS = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]],
    dtype=np.float32,
)


def marker_image(height: int, width: int, boxes: np.ndarray) -> np.ndarray:
    """Paint every box as a filled rectangle of its own color on a black RGB image."""
    image = np.zeros((height, width, 3), dtype=np.float32)
    for index, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=int)):
        image[y1:y2, x1:x2] = MARKER_COLORS[index % len(MARKER_COLORS)]
    return image


def loaded_sample(height: int, width: int, boxes: np.ndarray, labels: np.ndarray) -> ModelGTSample:
    """Build a sample as the loading transform leaves it, holding a marker image."""
    intrinsics = torch.eye(3).unsqueeze(0)
    identity = torch.eye(4).unsqueeze(0)
    camera_image_data = BaseImages(
        images=torch.zeros((1, 3, 1, 1), dtype=torch.float32),
        timestamps=torch.zeros(1, dtype=torch.float64),
        camera_intrinsics=intrinsics,
        camera_names=["CAM"],
        lidar2images=identity,
        lidar2cams=identity,
        distortion_models=[""],
        distortion_coefficients=[torch.zeros(0)],
        augmented_camera_intrinsics=intrinsics.clone(),
        image_augmentation_matrices=identity.clone(),
    )
    sample = ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=camera_image_data,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
    )
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return replace_image_and_boxes(
        sample, marker_image(height, width, boxes), boxes, np.asarray(labels, dtype=np.int32)
    )


def file_sample(
    path: Path, height: int, width: int, boxes: np.ndarray, labels: np.ndarray
) -> ModelGTSample:
    """Write a marker image to a lossless file and build the sample naming it."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cv2.imwrite(str(path), marker_image(height, width, boxes)[:, :, ::-1].astype(np.uint8))
    image_sample = ImageSample(
        image_path=str(path),
        camera_name="CAM",
        timestamp=0.0,
        camera_intrinsic=torch.eye(3),
        lidar2cam=torch.eye(4),
        lidar2image=torch.eye(4),
        distortion_model="",
        distortion_coefficients=torch.zeros(0),
    )
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=[image_sample],
        point_cloud_data=None,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
        detection2d_gt_sample=detection2d_gt_sample(boxes, np.asarray(labels, dtype=np.int32)),
    )


def marker_extent(
    image: np.ndarray, color: np.ndarray, tolerance: float = 90.0
) -> np.ndarray | None:
    """Find the extent of the pixels of a marker color.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` with exclusive maxima, None when no pixel is close to
        the color.
    """
    ys, xs = np.nonzero(np.abs(image - color).sum(axis=2) < tolerance)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def assert_boxes_cover_their_markers(sample: ModelGTSample, tolerance: float) -> None:
    """Check that every box of a sample lies on the pixels of its marker color.

    Args:
        sample: Transformed sample whose boxes kept the order of the markers.
        tolerance: Largest distance in pixels between a box edge and the marker extent.
    """
    image = image_of(sample)
    boxes, _ = boxes_of(sample)
    for index, box in enumerate(boxes):
        extent = marker_extent(image, MARKER_COLORS[index % len(MARKER_COLORS)])
        if extent is None:
            raise AssertionError(f"The marker of box {index} is not in the image.")
        np.testing.assert_allclose(box, extent, atol=tolerance, err_msg=f"box {index}")
