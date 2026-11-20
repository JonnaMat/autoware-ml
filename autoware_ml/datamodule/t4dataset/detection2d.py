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

"""Dataset of single camera images with 2D box annotations."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import logging

import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.box2d_schemas import Box2DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.image_frames import ImageFrameDatasetSchema
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.datamodule.base_dataset import BaseDataset
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.detection2d.multi_sample import MultiSampleTransform
from autoware_ml.types.geometry import Box2DFieldIndex

logger = logging.getLogger(__name__)


class _RandomSampleSource:
    """Sample source of the mixing transforms: a random sample of a dataset, loaded.

    A class instead of a closure, so a dataset holding it pickles into dataloader workers that
    are spawned.
    """

    def __init__(self, dataset: T4Detection2DDataset, loading: TransformsCompose) -> None:
        """Keep the dataset to draw from and the transforms that load a sample.

        Args:
          dataset: Dataset bound to its records.
          loading: Transforms of the pipeline before its first mixing transform.
        """
        self.dataset = dataset
        self.loading = loading

    def __call__(self) -> ModelGTSample:
        """Load another sample of the dataset, drawn at random."""
        index = int(np.random.randint(len(self.dataset)))
        return self.loading(self.dataset.get_data_sample(index))


class T4Detection2DDataset(BaseDataset):
    """
    Dataset serving the camera images of a 2D database, one sample per record.

    A record holds one camera image and the 2D boxes annotated in it. The sample names the
    image file, which the loading transform of the pipeline reads, and carries the boxes in
    pixels of the raw image with their class indices. A box whose label the taxonomy does not
    train is dropped.

    The records are decoded once, when the dataset is bound, so a sample is built by index
    without touching the record table: the mixing transforms of a training pipeline (mosaic,
    mixup) draw further samples at random for every sample they augment. The dataset hands
    them the samples through :class:`MultiSampleTransform`, loaded by the transforms of the
    pipeline that come before the first mixing transform.
    """

    _SERVED_COLUMNS = (
        DatasetTableSchema.SCENARIO_ID.name,
        DatasetTableSchema.TIMESTAMP_SECONDS.name,
        DatasetTableSchema.IMAGE_FRAMES.name,
        DatasetTableSchema.BOXES_2D.name,
    )

    def __init__(
        self,
        database_root_path: str,
        *,
        taxonomy: LabelTaxonomy,
        max_num_2d_gt_bboxes: int,
        transforms: TransformsCompose | None,
        dataset_records_dataframe: pl.DataFrame | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
          database_root_path: Root directory of the database the dataset reads.
          taxonomy: Taxonomy resolving the fine label name of a box to its class index.
          max_num_2d_gt_bboxes: Number of 2D boxes every sample is padded to in a batch, extra
            boxes are dropped.
          transforms: Transforms applied to every sample, starting with the loading transform.
          dataset_records_dataframe: Records of the corpus. None until bound to records.
        """
        super().__init__(
            database_root_path=database_root_path,
            max_num_2d_gt_bboxes=max_num_2d_gt_bboxes,
            dataset_records_dataframe=None,
            transforms=transforms,
            det3d_supervised=False,
            seg3d_supervised=False,
        )
        self.taxonomy = taxonomy
        self._image_samples: list[ImageSample] = []
        self._boxes: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        if dataset_records_dataframe is not None:
            self._serve_records(dataset_records_dataframe)

    def _serve_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Decode the records into the image and the boxes of every sample.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Raises:
          ValueError: If a record carries no 2D boxes column or not exactly one image frame,
            so the table was not built by a single camera 2D database.
        """
        records = dataset_records_dataframe.select(self._SERVED_COLUMNS)
        self.dataset_records_dataframe = records
        self._image_samples = []
        self._boxes = []
        self._labels = []
        num_dropped = 0
        identity = torch.eye(4, dtype=torch.float32)
        for row in records.iter_rows(named=True):
            image_frames = row[DatasetTableSchema.IMAGE_FRAMES.name]
            boxes_2d = row[DatasetTableSchema.BOXES_2D.name]
            if boxes_2d is None or not image_frames or len(image_frames[0]) != 1:
                raise ValueError(
                    "A record carries no 2D boxes or not exactly one camera image, the record "
                    "table was not built by a single camera 2D database."
                )
            image_frame = image_frames[0][0]
            self._image_samples.append(
                ImageSample(
                    image_path=str(
                        self.database_root_path
                        / image_frame[ImageFrameDatasetSchema.image_relative_path.name]
                    ),
                    camera_name=image_frame[ImageFrameDatasetSchema.image_sensor_channel_name.name],
                    timestamp=image_frame[ImageFrameDatasetSchema.image_timestamp_seconds.name],
                    camera_intrinsic=torch.tensor(
                        image_frame[ImageFrameDatasetSchema.cam2img.name], dtype=torch.float32
                    ),
                    lidar2cam=identity,
                    lidar2image=identity,
                    distortion_model=image_frame[
                        ImageFrameDatasetSchema.image_distortion_model.name
                    ],
                    distortion_coefficients=torch.tensor(
                        image_frame[ImageFrameDatasetSchema.image_distortion_coefficients.name],
                        dtype=torch.float32,
                    ),
                )
            )
            boxes = []
            labels = []
            for box in boxes_2d:
                label = self.taxonomy.class_index(box[Box2DDatasetSchema.BOX2D_LABEL_NAME.name])
                if label == self.taxonomy.ignore_index:
                    num_dropped += 1
                    continue
                boxes.append(box[Box2DDatasetSchema.BOX2D_PARAMS.name])
                labels.append(label)
            self._boxes.append(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
            self._labels.append(np.asarray(labels, dtype=np.int32))
        logger.info(
            f"{type(self).__name__}: {len(self._image_samples)} images, "
            f"{sum(len(labels) for labels in self._labels)} boxes of the classes "
            f"{list(self.taxonomy.class_names)}, {num_dropped} boxes of other labels dropped."
        )
        self._bind_sample_source()

    def _bind_sample_source(self) -> None:
        """
        Hand the mixing transforms of the pipeline the samples of this dataset.

        The pipeline and its mixing transforms are copied first: the declared dataset is bound
        once per split, and every bound copy serves the samples of its own records.
        """
        if self.transforms is None:
            return
        pipeline = list(self.transforms.pipeline)
        mixing = [
            index
            for index, transform in enumerate(pipeline)
            if isinstance(transform, MultiSampleTransform)
        ]
        if not mixing:
            return
        sample_source = _RandomSampleSource(self, TransformsCompose(pipeline[: mixing[0]]))
        for index in mixing:
            pipeline[index] = copy.copy(pipeline[index])
            pipeline[index].bind_sample_source(sample_source)
        self.transforms = TransformsCompose(pipeline)

    def box_sizes(self) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        """
        Boxes and class indices of every sample, in pixels of the raw images.

        Returns:
          Sequence[tuple[np.ndarray, np.ndarray]]: ``(boxes, labels)`` per sample, the boxes
            as ``(x_min, y_min, x_max, y_max)``.
        """
        return list(zip(self._boxes, self._labels, strict=True))

    def get_data_sample(self, index: int) -> ModelGTSample:
        """
        Build the sample of one image before the transform pipeline.

        Args:
          index: Index of the record.

        Returns:
          ModelGTSample: The sample naming its image file and carrying its 2D boxes in pixels
            of the raw image.
        """
        boxes = torch.from_numpy(self._boxes[index])
        boxes_with_centers = torch.zeros(
            (boxes.shape[0], len(Box2DFieldIndex)), dtype=torch.float32
        )
        boxes_with_centers[:, :4] = boxes
        boxes_with_centers[:, 4] = (boxes[:, 0] + boxes[:, 2]) * 0.5
        boxes_with_centers[:, 5] = (boxes[:, 1] + boxes[:, 3]) * 0.5
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=[self._image_samples[index]],
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
            detection2d_gt_sample=Detection2DGTSample(
                gt_bboxes_2d=[boxes_with_centers],
                gt_labels_2d=[torch.from_numpy(self._labels[index])],
            ),
        )
