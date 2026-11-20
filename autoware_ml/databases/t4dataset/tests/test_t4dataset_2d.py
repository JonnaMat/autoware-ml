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

"""Unit tests for the T4 database of camera images with 2D boxes: record generation from the
annotation tables, the label vocabulary, the cache and the scenario list formats."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.databases.t4dataset.t4dataset_2d import T4Dataset2D
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.t4dataset.tests.t4_2d_fixtures import write_scenario_list, write_scene
from autoware_ml.databases.taxonomy import LabelVocabulary
from autoware_ml.types.dataset import SplitType

DATASET_NAME = "db_test_2d_v1"
VOCABULARY = {
    "green": "traffic_light",
    "crosswalk_red": "pedestrian_traffic_light",
    "billboard": None,
}


class T4Dataset2DTestCase(unittest.TestCase):
    """Shared fixture: a database root with a training and a test scene."""

    def setUp(self) -> None:
        """Write two scenes, their scenario list and build the database."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_scene(self.root / "data", DATASET_NAME, "scene_train", num_frames=3)
        write_scene(self.root / "data", DATASET_NAME, "scene_test", num_frames=2)
        write_scenario_list(
            self.root / "scenarios",
            DATASET_NAME,
            {"train": ["scene_train"], "test": ["scene_test"]},
        )

    def build_database(
        self, vocabulary: dict[str, str | None] | None = None, **kwargs: object
    ) -> T4Dataset2D:
        """Build the database over the two scenes."""
        scenarios = T4Scenarios(
            scenario_root_path=self.root / "scenarios",
            dataset_params=[DatasetParams(dataset_name=DATASET_NAME, sample_steps=1)],
        )
        return T4Dataset2D(
            version="test-2d",
            root_path=str(self.root / "data"),
            scenarios={"db_test": scenarios},
            cache_path=str(self.root / "cache"),
            cache_file_prefix_name="database_2d",
            num_workers=1,
            vocabulary=LabelVocabulary(class_renaming=vocabulary or VOCABULARY),
            **kwargs,
        )


class TestT4Dataset2D(T4Dataset2DTestCase):
    """Unit tests for the record generation."""

    def test_scenario_list_accepts_whitespace_separated_versions(self) -> None:
        """Test that ``<scene>   <version>`` entries parse like ``<scene>/<version>`` ones."""
        database = self.build_database()

        scenarios = database.scenarios["db_test"].scenario_data
        self.assertEqual([s.scenario_id for s in scenarios[SplitType.TRAIN]], ["scene_train"])
        self.assertEqual([s.scenario_version for s in scenarios[SplitType.TEST]], ["0"])
        self.assertEqual(scenarios[SplitType.VAL], [])

    def test_records_hold_one_key_frame_image_each(self) -> None:
        """Test the records of the two scenes.

        Check: one record per key frame (3 + 2), none for the frame that is no key frame,
        each with one image frame whose path is relative to the database root, no lidar
        frame, and the boxes of the image with their dataset and fine label names.
        """
        database = self.build_database()
        database.process_scenario_records()

        records = database.load_scenario_records()

        self.assertEqual(len(records), 5)
        by_id = {record.sample_id: record for record in records}
        record = by_id["scene_train_frame_2"]
        self.assertIsInstance(record, DatasetRecord)
        self.assertEqual(record.scenario_id, "scene_train")
        self.assertEqual(record.scenario_name, "recording_of_scene_train")
        self.assertEqual(record.timestamp_seconds, 3.0)
        self.assertEqual(len(record.lidar_frames), 0)
        self.assertIsNone(record.boxes_3d)
        (image_frame,) = record.image_frames[0]
        self.assertEqual(
            image_frame.image_relative_path,
            f"{DATASET_NAME}/scene_train/0/data/CAM_TRAFFIC_LIGHT_NEAR/2.jpg",
        )
        self.assertEqual((image_frame.image_height, image_frame.image_width), (48, 64))
        self.assertEqual(image_frame.image_sensor_channel_name, "CAM_TRAFFIC_LIGHT_NEAR")
        np.testing.assert_allclose(
            image_frame.image_sensor_to_ego_pose_matrix[:3, 3], [1.0, 0.0, 2.0]
        )
        np.testing.assert_allclose(
            image_frame.image_frame_ego_pose_to_global_matrix[:3, 3], [2.0, 0.0, 0.0]
        )
        self.assertEqual(
            [(box.box2d_dataset_label_name, box.box2d_label_name) for box in record.boxes_2d],
            [("green", "traffic_light"), ("crosswalk_red", "pedestrian_traffic_light")],
        )
        np.testing.assert_allclose(record.boxes_2d[1].box2d_params, [14.0, 6.0, 22.0, 22.0])
        self.assertEqual(record.boxes_2d[0].box2d_attributes, {"occlusion_state.none"})
        self.assertEqual(record.boxes_2d[0].box2d_image_frame_id, "scene_train_frame_2")
        # An image without a box is a negative of the scene, not a missing annotation.
        self.assertEqual(by_id["scene_train_frame_0"].boxes_2d, [])

    def test_label_outside_every_taxonomy_keeps_no_fine_name(self) -> None:
        """Test that a label the vocabulary maps to null is stored without a fine label name."""
        write_scene(self.root / "data", DATASET_NAME, "scene_extra", num_frames=4)
        write_scenario_list(self.root / "scenarios", DATASET_NAME, {"train": ["scene_extra"]})
        database = self.build_database()
        database.process_scenario_records()

        records = {record.sample_id: record for record in database.load_scenario_records()}

        labels = [box.box2d_label_name for box in records["scene_extra_frame_3"].boxes_2d]
        self.assertEqual(labels, ["traffic_light", "pedestrian_traffic_light", None])

    def test_unlisted_label_fails_the_generation(self) -> None:
        """Test that a box label the vocabulary does not list stops the generation by name."""
        database = self.build_database(vocabulary={"green": "traffic_light"})

        with self.assertRaisesRegex(ValueError, "crosswalk_red"):
            database.process_scenario_records()

    def test_sample_steps_and_camera_channels_select_the_images(self) -> None:
        """Test that other camera channels serve nothing and the key frame filter can be lifted."""
        other_camera = self.build_database(camera_channels=["CAM_OTHER"])
        all_frames = self.build_database(keyframes_only=False)
        other_camera.process_scenario_records()
        all_frames.process_scenario_records()

        self.assertEqual(len(other_camera.load_scenario_records()), 0)
        self.assertEqual(len(all_frames.load_scenario_records()), 7)

    def test_cache_is_keyed_by_the_content_settings(self) -> None:
        """Test that the vocabulary and the served cameras change the cache, the root path not."""
        reference = self.build_database()

        self.assertEqual(reference.database_hash, self.build_database().database_hash)
        self.assertNotEqual(
            reference.database_hash,
            self.build_database(camera_channels=["CAM_OTHER"]).database_hash,
        )
        self.assertNotEqual(
            reference.database_hash,
            self.build_database(
                vocabulary={**VOCABULARY, "billboard": "traffic_light"}
            ).database_hash,
        )
        self.assertIn(DatasetTableSchema.BOXES_2D.name, reference.get_polars_schema())
