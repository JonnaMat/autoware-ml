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

"""Builders of small YOLOX detectors and of their model inputs for the tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.common.backbones.csp_darknet import CSPDarknet
from autoware_ml.models.common.necks.yolox_pafpn import YOLOXPAFPN
from autoware_ml.models.detection2d.heads.yolox import YOLOXHead
from autoware_ml.models.detection2d.yolox import YOLOXDetector
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.transforms.detection2d.sample_io import detection2d_gt_sample


def build_detector(
    num_classes: int = 2, widen_factor: float = 0.125, **kwargs: Any
) -> YOLOXDetector:
    """Build a narrow YOLOX detector that runs fast on a CPU.

    Args:
        num_classes: Number of classes of the head.
        widen_factor: Width multiplier of the backbone.
        **kwargs: Further arguments of the detector.

    Returns:
        The detector.
    """
    backbone = CSPDarknet(deepen_factor=0.33, widen_factor=widen_factor)
    return YOLOXDetector(
        backbone=backbone,
        neck=YOLOXPAFPN(in_channels=backbone.out_channels),
        head=YOLOXHead(
            num_classes=num_classes, in_channels=backbone.out_channels, feat_channels=16
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        **kwargs,
    )


def build_batch_inputs(
    images: torch.Tensor,
    gt_boxes: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    scale: float = 1.0,
) -> ModelBatchInputs:
    """Build the model inputs of a batch of single camera images with 2D boxes.

    Args:
        images: Images ``(batch_size, 3, height, width)`` in RGB.
        gt_boxes: Boxes ``(x_min, y_min, x_max, y_max)`` of every image, in pixels of the images.
        gt_labels: Class index of every box of every image.
        scale: Scale the test pipeline resized the raw images by, recorded as the image
          augmentation of every image.

    Returns:
        The model inputs.
    """
    batch_size = images.shape[0]
    identity = torch.eye(4).repeat(batch_size, 1, 1, 1)
    augmentation = identity.clone()
    augmentation[:, :, 0, 0] = scale
    augmentation[:, :, 1, 1] = scale
    image_batch = ImageGTBatch(
        images=images[:, None].to(torch.float32),
        depth_maps=None,
        camera_intrinsics=torch.eye(3).repeat(batch_size, 1, 1, 1),
        image_augmentation_matrices=augmentation,
        lidar2images=identity,
        lidar2cams=identity,
        calibration_statuses=None,
    )
    samples = [
        detection2d_gt_sample(boxes.numpy(), labels.numpy())
        for boxes, labels in zip(gt_boxes, gt_labels, strict=True)
    ]
    batch = ModelGTBatch(
        point_cloud_gt_batch=None,
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=None,
        image_gt_batch=image_batch,
        padding_mask=torch.zeros(batch_size, dtype=torch.bool),
        detection2d_gt_batch=Detection2DGTBatch.collate_gt_samples(samples, max_num_2d_gt_bboxes=8),
    )
    return ModelBatchInputs.from_gt_batch(batch)
