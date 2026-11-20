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

"""Unit tests for the single camera 2D detection dataset and its datamodule wiring: samples,
class resolution, the sample source of the mixing transforms, collation and the validation
split served by the test scenarios."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.databases.t4dataset.t4dataset_2d import T4Dataset2D
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.t4dataset.tests.t4_2d_fixtures import write_scenario_list, write_scene
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary
from autoware_ml.datamodule.data_module import DataLoaderConfig, DataModule
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.datamodule.splitters.dataset_record_splitter import ScenarioSplitter
from autoware_ml.datamodule.t4dataset.detection2d import T4Detection2DDataset
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.detection2d.augmentation import MixUp, Mosaic, RandomAffine
from autoware_ml.transforms.detection2d.formatting import Letterbox
from autoware_ml.transforms.detection2d.loading import LoadImageAndResize
from autoware_ml.types.dataset import SplitType

DATASET_NAME = "db_test_2d_v1"
CLASS_RENAMING = {
    "green": "traffic_light",
    "crosswalk_red": "pedestrian_traffic_light",
    "billboard": None,
}


class TestT4Detection2DDataset(unittest.TestCase):
    """Unit tests for the 2D detection dataset."""

    def setUp(self) -> None:
        """Write a training and a test scene and build their database and taxonomy."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.data_root = root / "data"
        write_scene(self.data_root, DATASET_NAME, "scene_train", num_frames=4)
        write_scene(self.data_root, DATASET_NAME, "scene_test", num_frames=2)
        write_scenario_list(
            root / "scenarios", DATASET_NAME, {"train": ["scene_train"], "test": ["scene_test"]}
        )
        vocabulary = LabelVocabulary(class_renaming=CLASS_RENAMING)
        self.database = T4Dataset2D(
            version="test-2d",
            root_path=str(self.data_root),
            scenarios={
                "db_test": T4Scenarios(
                    scenario_root_path=root / "scenarios",
                    dataset_params=[DatasetParams(dataset_name=DATASET_NAME, sample_steps=1)],
                )
            },
            cache_path=str(root / "cache"),
            cache_file_prefix_name="database_2d",
            num_workers=1,
            vocabulary=vocabulary,
        )
        self.taxonomy = LabelTaxonomy(
            vocabulary=vocabulary,
            class_names=["traffic_light", "pedestrian_traffic_light"],
            class_mapping={
                "traffic_light": "traffic_light",
                "pedestrian_traffic_light": "pedestrian_traffic_light",
            },
            ignore_index=-1,
        )

    def build_dataset(self, pipeline: list | None) -> T4Detection2DDataset:
        """Build an unbound dataset with a pipeline."""
        return T4Detection2DDataset(
            database_root_path=str(self.data_root),
            taxonomy=self.taxonomy,
            max_num_2d_gt_bboxes=16,
            transforms=TransformsCompose(pipeline) if pipeline is not None else None,
        )

    def build_datamodule(self, train_pipeline: list, split_fallbacks: dict | None) -> DataModule:
        """Build the datamodule over the database with a letterbox test pipeline."""
        test_pipeline = [LoadImageAndResize(image_size=[64, 64]), Letterbox(image_size=[64, 64])]
        dataloader = DataLoaderConfig(batch_size=2, num_workers=0)
        return DataModule(
            splitter=ScenarioSplitter(split_fallbacks=split_fallbacks),
            sources={
                "main": DatasetSource(
                    database=self.database,
                    train_dataset=self.build_dataset(train_pipeline),
                    validation_dataset=self.build_dataset(test_pipeline),
                    test_dataset=self.build_dataset(test_pipeline),
                )
            },
            train_dataloader=dataloader,
            validation_dataloader=dataloader,
            test_dataloader=dataloader,
            predict_dataloader=None,
            train_frame_sampling=None,
        )

    def test_sample_carries_the_image_path_and_the_trained_boxes(self) -> None:
        """Test the raw sample of a record.

        Check: the image path is joined onto the database root, the boxes are in raw pixels
        with the box center as object center, and the box of the label outside the taxonomy
        is dropped.
        """
        self.database.process_scenario_records()
        records = self.database.load_polars_scenario_dataframe().filter(scenario_id="scene_train")
        dataset = self.build_dataset(None).bind(records)

        sample = dataset.get_data_sample(3)

        self.assertEqual(len(dataset), 4)
        self.assertEqual(
            sample.image_samples[0].image_path,
            str(self.data_root / DATASET_NAME / "scene_train/0/data/CAM_TRAFFIC_LIGHT_NEAR/3.jpg"),
        )
        boxes = sample.detection2d_gt_sample.gt_bboxes_2d[0]
        torch.testing.assert_close(
            boxes,
            torch.tensor([[4.0, 6.0, 12.0, 23.0, 8.0, 14.5], [14.0, 6.0, 22.0, 23.0, 18.0, 14.5]]),
        )
        self.assertEqual(sample.detection2d_gt_sample.gt_labels_2d[0].tolist(), [0, 1])
        self.assertEqual(len(dataset.get_data_sample(0).detection2d_gt_sample.gt_labels_2d[0]), 0)

    def test_records_of_another_database_are_refused(self) -> None:
        """Test that records without a 2D boxes column are refused when the dataset is bound."""
        self.database.process_scenario_records()
        records = self.database.load_polars_scenario_dataframe().with_columns(boxes_2d=None)

        with self.assertRaises(ValueError):
            self.build_dataset(None).bind(records)

    def test_validation_falls_back_to_the_test_scenarios(self) -> None:
        """Test that a database without validation scenarios validates on its test scenarios.

        Check: with the fallback the validation split serves the records of the test
        scenario, and without it setting up the validation split fails as an empty split.
        """
        with_fallback = self.build_datamodule(
            [LoadImageAndResize(image_size=[64, 64])], {"val": "test"}
        )
        with_fallback.prepare_data()
        with_fallback.setup("fit")
        without_fallback = self.build_datamodule([LoadImageAndResize(image_size=[64, 64])], None)

        self.assertEqual(len(with_fallback.datasets[SplitType.TRAIN]), 4)
        self.assertEqual(len(with_fallback.datasets[SplitType.VAL]), 2)
        with self.assertRaises(ValueError):
            without_fallback.setup("fit")

    def test_training_pipeline_mixes_samples_and_collates(self) -> None:
        """Test a training pipeline with mosaic and mixup end to end through the dataloader.

        Check: the mixing transforms draw their samples from the bound dataset, a batch holds
        images of the input size and box tensors padded to the configured maximum, and every
        valid box lies inside the image.
        """
        np.random.seed(0)
        image_size = [64, 64]
        pipeline = [
            LoadImageAndResize(image_size=image_size),
            Mosaic(image_size=image_size),
            RandomAffine(image_size=image_size, scale_range=[0.5, 1.0]),
            MixUp(image_size=image_size),
            Letterbox(image_size=image_size),
        ]
        datamodule = self.build_datamodule(pipeline, {"val": "test"})
        datamodule.prepare_data()
        datamodule.setup("fit")

        batch = next(iter(datamodule.train_dataloader()))
        validation_batch = next(iter(datamodule.val_dataloader()))

        self.assertEqual(tuple(batch.image_gt_batch.images.shape), (2, 1, 3, 64, 64))
        self.assertEqual(tuple(batch.detection2d_gt_batch.gt_bboxes_2d.shape), (2, 1, 16, 6))
        for cameras in batch.detection2d_gt_batch.valid_bboxes_2d():
            boxes = cameras[0][:, :4]
            self.assertTrue(((boxes >= 0) & (boxes <= 64)).all())
        # The validation images are 64x48 scaled by 1 and padded at the bottom.
        self.assertEqual(tuple(validation_batch.image_gt_batch.images.shape), (2, 1, 3, 64, 64))
        self.assertTrue((validation_batch.image_gt_batch.images[:, :, :, 48:] == 114.0).all())
        self.assertEqual(
            validation_batch.detection2d_gt_batch.gt_valid_bboxes_2d.flatten().tolist(), [0, 1]
        )

    def test_mixing_transforms_are_bound_per_dataset(self) -> None:
        """Test that binding leaves the declared pipeline alone and binds a copy of its own."""
        self.database.process_scenario_records()
        records = self.database.load_polars_scenario_dataframe()
        mosaic = Mosaic(image_size=[64, 64])
        declared = self.build_dataset([LoadImageAndResize(image_size=[64, 64]), mosaic])

        bound = declared.bind(records)

        self.assertIs(declared.transforms.pipeline[1], mosaic)
        self.assertIsNot(bound.transforms.pipeline[1], mosaic)
        with self.assertRaises(RuntimeError):
            mosaic.draw_sample()
        self.assertIsNotNone(bound.transforms.pipeline[1].draw_sample().camera_image_data)
