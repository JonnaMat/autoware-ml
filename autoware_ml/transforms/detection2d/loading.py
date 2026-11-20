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

"""Image loading of the 2D detection pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.detection2d.sample_io import boxes_of, replace_image_and_boxes

_REDUCED_DECODING_FLAGS = {
    1: cv2.IMREAD_COLOR,
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}


class LoadImageAndResize(BaseTransform):
    """Load the image of a single camera sample and resize it to fit the network input.

    The image is read as 8 bit, resized with its aspect ratio kept so it fits ``image_size``
    and served as RGB float32 in ``[0, 255]``, the range the YOLOX weights read. Resizing at
    load time keeps every later transform on a small image, which is what makes mosaic
    affordable on high resolution cameras. The boxes of the sample are scaled along.

    Required keys:
        - image_samples: the one camera of the sample.
        - detection2d_gt_sample: its 2D boxes in pixels of the raw image.

    Generated keys:
        - camera_image_data: BaseImages holding the resized image and its calibration.
    """

    _required_keys = ["image_samples", "detection2d_gt_sample"]

    def __init__(self, image_size: Sequence[int] | None, decoding_reduction: int = 1) -> None:
        """Initialize the transform.

        Args:
            image_size: ``(height, width)`` the image is resized to fit, None to keep the raw
              size.
            decoding_reduction: Factor the decoder shrinks the image by while decoding, one of
              1, 2, 4 and 8. A JPEG decodes several times faster at a reduced size, which
              pays off in training when the image is shrunk by at least that factor anyway.
              The raw size is then known up to the rounding of the reduction, so the boxes can
              be off by a fraction of a pixel; evaluation pipelines keep 1.

        Raises:
            ValueError: If the reduction is none of the supported factors.
        """
        super().__init__(probability=None)
        if decoding_reduction not in _REDUCED_DECODING_FLAGS:
            raise ValueError(
                f"decoding_reduction must be one of {sorted(_REDUCED_DECODING_FLAGS)}, got "
                f"{decoding_reduction}."
            )
        self.image_size = tuple(int(size) for size in image_size) if image_size else None
        self.decoding_reduction = decoding_reduction

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Load and resize the image of the sample.

        Args:
            model_gt_sample: Sample naming its image file.

        Returns:
            The sample with ``camera_image_data`` and its boxes in pixels of the resized image.

        Raises:
            ValueError: If the sample holds another number of cameras than one.
            FileNotFoundError: If the image file cannot be read.
        """
        image_samples = model_gt_sample.image_samples
        if len(image_samples) != 1:
            raise ValueError(f"Expected one camera per sample, got {len(image_samples)}.")
        image_sample = image_samples[0]
        image = cv2.imread(
            image_sample.image_path, _REDUCED_DECODING_FLAGS[self.decoding_reduction]
        )
        if image is None:
            raise FileNotFoundError(f"Cannot read the image {image_sample.image_path}.")
        raw_height = image.shape[0] * self.decoding_reduction
        raw_width = image.shape[1] * self.decoding_reduction

        if self.image_size is not None:
            ratio = min(self.image_size[0] / raw_height, self.image_size[1] / raw_width)
            new_width, new_height = int(raw_width * ratio), int(raw_height * ratio)
            if (new_height, new_width) != image.shape[:2]:
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        scale_x = image.shape[1] / raw_width
        scale_y = image.shape[0] / raw_height
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        camera_intrinsics = image_sample.camera_intrinsic.unsqueeze(0)
        camera_image_data = BaseImages(
            # Replaced below, together with the composed augmentation.
            images=torch.zeros((1, 3, 1, 1), dtype=torch.float32),
            depth_maps=None,
            timestamps=torch.tensor([image_sample.timestamp], dtype=torch.float64),
            camera_intrinsics=camera_intrinsics,
            camera_names=[image_sample.camera_name],
            lidar2images=image_sample.lidar2image.unsqueeze(0),
            lidar2cams=image_sample.lidar2cam.unsqueeze(0),
            distortion_models=[image_sample.distortion_model],
            distortion_coefficients=[image_sample.distortion_coefficients],
            augmented_camera_intrinsics=camera_intrinsics.clone(),
            image_augmentation_matrices=BaseImages.identity_image_augmentation_matrices(
                camera_intrinsics
            ),
        )
        boxes, labels = boxes_of(model_gt_sample)
        boxes *= np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        return replace_image_and_boxes(
            model_gt_sample._replace(camera_image_data=camera_image_data),
            image,
            boxes,
            labels,
            pixel_affine=np.diag([scale_x, scale_y, 1.0]),
        )
