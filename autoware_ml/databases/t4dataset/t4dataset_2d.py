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

"""T4 database of camera images with 2D box annotations and the generation of its records.

A T4 scene annotated in 2D carries its boxes in ``annotation/object_ann.json``, one entry per
box of one camera image, and needs no lidar frame. The record table of this database holds one
record per camera image: the image as the single image frame of the record and its boxes in
the ``boxes_2d`` column. The annotation tables are read as plain JSON.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl
from pyquaternion import Quaternion
from tqdm import tqdm

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.database_interface import DatabaseInterface
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.box2d_schemas import Box2DDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.schemas.image_frames import (
    ImageFrameDataModel,
    distortion_model_from_coefficients,
)
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.taxonomy import LabelVocabulary
from autoware_ml.utils.dataset import convert_quaternion_to_matrix

logger = logging.getLogger(__name__)

_IMAGE_FILE_FORMATS = frozenset({"jpg", "jpeg", "png"})


@dataclass(frozen=True)
class T4Records2DGeneratorWorkerParams:
    """
    Parameters of one scenario processed by a worker.

    Attributes:
      database_root_path: Root path of the T4 database.
      scenario_data: Scenario data.
      vocabulary: Vocabulary of the raw box labels.
      camera_channels: Camera channels whose images are served, every camera when None.
      keyframes_only: Whether only the key frames of a camera are served.
    """

    database_root_path: str
    scenario_data: ScenarioData
    vocabulary: LabelVocabulary
    camera_channels: tuple[str, ...] | None
    keyframes_only: bool


def _load_table(annotation_dir: Path, name: str) -> list[dict[str, Any]]:
    """Read one annotation table of a scene."""
    with open(annotation_dir / f"{name}.json", "r") as table_file:
        return json.load(table_file)


def _pose_matrix(record: Mapping[str, Any]) -> np.ndarray:
    """Build the 4x4 transform of a record holding a translation and a ``wxyz`` rotation."""
    return convert_quaternion_to_matrix(
        rotation_quaternion=Quaternion(record["rotation"]),
        translation=np.asarray(record["translation"], dtype=np.float64),
    )


def generate_scene_records(worker_params: T4Records2DGeneratorWorkerParams) -> list[DatasetRecord]:
    """
    Build the records of one scene, one per served camera image.

    Images without a box are served too, with an empty box list: they are the negatives of
    the scene.

    Args:
      worker_params: Scenario and settings of the generation.

    Returns:
      list[DatasetRecord]: The records of the scene in time order, every
        ``sample_steps``-th image of each camera.

    Raises:
      ValueError: If a box carries a label the vocabulary does not list.
    """
    scenario = worker_params.scenario_data
    scene_relative_path = (
        Path(scenario.dataset_params.dataset_name)
        / scenario.scenario_id
        / scenario.scenario_version
    )
    annotation_dir = Path(worker_params.database_root_path) / scene_relative_path / "annotation"

    sensors = {record["token"]: record for record in _load_table(annotation_dir, "sensor")}
    calibrated_sensors = {
        record["token"]: record for record in _load_table(annotation_dir, "calibrated_sensor")
    }
    ego_poses = {record["token"]: record for record in _load_table(annotation_dir, "ego_pose")}
    categories = {
        record["token"]: record["name"] for record in _load_table(annotation_dir, "category")
    }
    attributes = {
        record["token"]: record["name"] for record in _load_table(annotation_dir, "attribute")
    }
    scenes = _load_table(annotation_dir, "scene")
    scenario_name = scenes[0]["name"] if scenes else scenario.scenario_id

    unlisted = worker_params.vocabulary.unlisted(categories.values())
    object_annotations = _load_table(annotation_dir, "object_ann")
    used_unlisted = sorted(
        {categories[annotation["category_token"]] for annotation in object_annotations}
        & set(unlisted)
    )
    if used_unlisted:
        raise ValueError(
            f"Scene {scene_relative_path} annotates boxes with the labels {used_unlisted}, "
            "which the vocabulary of the database does not list. List them with their fine "
            "label name, or with null when they are outside every taxonomy."
        )

    boxes_by_image: defaultdict[str, list[Box2DDataModel]] = defaultdict(list)
    for annotation in object_annotations:
        raw_label = categories[annotation["category_token"]]
        boxes_by_image[annotation["sample_data_token"]].append(
            Box2DDataModel(
                box2d_params=np.asarray(annotation["bbox"], dtype=np.float64),
                box2d_instance_id=annotation["instance_token"],
                box2d_dataset_label_name=raw_label,
                box2d_label_name=worker_params.vocabulary.fine_name(raw_label),
                box2d_attributes={
                    attributes[token] for token in annotation.get("attribute_tokens", [])
                },
                box2d_image_frame_id=annotation["sample_data_token"],
            )
        )

    images_by_channel: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_data in _load_table(annotation_dir, "sample_data"):
        if sample_data.get("fileformat") not in _IMAGE_FILE_FORMATS:
            continue
        if worker_params.keyframes_only and not sample_data.get("is_key_frame", True):
            continue
        calibrated_sensor = calibrated_sensors[sample_data["calibrated_sensor_token"]]
        channel = sensors[calibrated_sensor["sensor_token"]]["channel"]
        if (
            worker_params.camera_channels is not None
            and channel not in worker_params.camera_channels
        ):
            continue
        images_by_channel[channel].append(sample_data)

    records: list[DatasetRecord] = []
    for channel in sorted(images_by_channel):
        channel_images = sorted(images_by_channel[channel], key=lambda record: record["timestamp"])
        for sample_data in channel_images[:: scenario.dataset_params.sample_steps]:
            calibrated_sensor = calibrated_sensors[sample_data["calibrated_sensor_token"]]
            distortion = [float(value) for value in calibrated_sensor["camera_distortion"]]
            timestamp_seconds = float(sample_data["timestamp"]) * 1e-6
            image_frame = ImageFrameDataModel(
                image_frame_id=sample_data["token"],
                image_keyframe=bool(sample_data.get("is_key_frame", True)),
                image_sensor_id=sample_data["calibrated_sensor_token"],
                image_sensor_channel_name=channel,
                image_timestamp_seconds=timestamp_seconds,
                image_relative_path=(scene_relative_path / sample_data["filename"]).as_posix(),
                image_height=sample_data.get("height"),
                image_width=sample_data.get("width"),
                cam2img=np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64),
                image_distortion_coefficients=distortion,
                image_distortion_model=distortion_model_from_coefficients(distortion),
                image_sensor_to_ego_pose_matrix=_pose_matrix(calibrated_sensor),
                image_frame_ego_pose_to_global_matrix=_pose_matrix(
                    ego_poses[sample_data["ego_pose_token"]]
                ),
                lidar2cam=None,
                lidar2img=None,
            )
            records.append(
                DatasetRecord(
                    scenario_id=scenario.scenario_id,
                    sample_id=sample_data["token"],
                    sample_index=len(records),
                    timestamp_seconds=timestamp_seconds,
                    location=scenario.location,
                    vehicle_type=scenario.vehicle_type,
                    scenario_name=scenario_name,
                    lidar_frames=[],
                    lidar_sources=None,
                    image_frames=[[image_frame]],
                    category_mapping=None,
                    boxes_3d=None,
                    boxes_2d=boxes_by_image.get(sample_data["token"], []),
                )
            )
    return records


class T4Dataset2D(BaseDatabase):
    """T4 database of camera images annotated with 2D boxes."""

    def __init__(
        self,
        version: str,
        root_path: str,
        scenarios: Mapping[str, T4Scenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        vocabulary: LabelVocabulary,
        camera_channels: Sequence[str] | None = None,
        keyframes_only: bool = True,
    ) -> None:
        """
        Initialize the database. Please refer to the BaseDatabase class for more details.

        Args:
          version: Version of the dataset.
          root_path: Root path where the scenes are stored, as
            ``<root_path>/<dataset_name>/<scene>/<version>``.
          scenarios: Scenario configurations by scenario group name.
          cache_path: Path to cache the dataset records.
          cache_file_prefix_name: Prefix name of the cache file.
          num_workers: Number of workers to use for processing the dataset.
          vocabulary: Vocabulary of the raw box labels of the annotations.
          camera_channels: Camera channels whose images are served, every camera when None.
          keyframes_only: Whether only the key frames of a camera are served.
        """
        super().__init__(
            version=version,
            root_path=root_path,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            num_workers=num_workers,
            vocabulary=vocabulary,
        )
        self._scenarios = scenarios
        self._camera_channels = tuple(camera_channels) if camera_channels is not None else None
        self._keyframes_only = keyframes_only

    def __str__(self) -> str:
        """
        String representation of the database.

        Returns:
          str: String representation of the database.
        """
        return (
            f"{self.__class__.__name__}(version={self._version}, "
            f"root_path={str(self._root_path)}, "
            f"cache path={str(self._cache_path)}, "
            f"cache file prefix name={self._cache_file_prefix_name}, "
            f"camera_channels={self._camera_channels}, "
            f"keyframes_only={self._keyframes_only}, "
            f"vocabulary={self._vocabulary}, "
            f"{self.scenarios_string_repr}"
            f")"
        )

    def __eq__(self, other: DatabaseInterface) -> bool:
        """
        Compare two databases by their string representation.

        Returns:
          bool: True if the databases are equal, False otherwise.
        """
        if not isinstance(other, T4Dataset2D):
            return False
        return str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the database by its string representation."""
        return hash(str(self))

    @property
    def hash_fields(self) -> Mapping[str, str]:
        """
        Get the explicit content fields that identify the cached records of the database.

        Extends the base fields with the served cameras and the vocabulary: the records store
        the fine label name of every box, which the vocabulary decides.

        Returns:
          Mapping[str, str]: Field name to its content representation, in a fixed order.
        """
        return {
            **super().hash_fields,
            "camera_channels": str(self._camera_channels),
            "keyframes_only": str(self._keyframes_only),
            "vocabulary": str(self._vocabulary),
        }

    def process_scenario_records(self) -> None:
        """Generate the record table of the database unless its cache exists."""
        start_time = time.perf_counter()
        df_cache_path = self.cache_file_path
        if df_cache_path.exists():
            logger.info(f"Cache file {df_cache_path} already exists, skip generating the caches")
            return

        unique_scenario_data = self.get_unique_scenario_data()
        logger.info(
            f"Processing a total of {len(unique_scenario_data)} unique scenarios in "
            f"{self.__class__.__name__}"
        )
        worker_params = [
            T4Records2DGeneratorWorkerParams(
                database_root_path=str(self._root_path),
                scenario_data=scenario_data,
                vocabulary=self._vocabulary,
                camera_channels=self._camera_channels,
                keyframes_only=self._keyframes_only,
            )
            for scenario_data in unique_scenario_data.values()
        ]
        records: list[DatasetRecord] = []
        if self._num_workers > 1:
            with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
                results = executor.map(generate_scene_records, worker_params)
                for result in tqdm(results, total=len(worker_params)):
                    records.extend(result)
        else:
            for worker_param in tqdm(worker_params, total=len(worker_params)):
                records.extend(generate_scene_records(worker_param))
        logger.info(f"Processed {len(records)} camera image records")

        df = pl.DataFrame(
            [record.to_dictionary() for record in records], schema=self.get_polars_schema()
        )
        df.write_parquet(df_cache_path)
        logger.info(
            f"Saved the database cache to {df_cache_path} in "
            f"{time.perf_counter() - start_time:.1f} seconds for the database: {self.version}"
        )
