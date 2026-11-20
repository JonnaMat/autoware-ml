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

"""Record table schema and data model of 2D bounding box annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

from jaxtyping import Float32, Float64
from pydantic import BaseModel, ConfigDict
import numpy as np
import polars as pl

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DataModelInterface,
    DatasetTableColumn,
)


@dataclass(frozen=True)
class Box2DDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to 2D bounding boxes.
    """

    BOX2D_PARAMS = DatasetTableColumn("box2d_params", pl.Array(pl.Float32, shape=(4,)))
    BOX2D_INSTANCE_ID = DatasetTableColumn("box2d_instance_id", pl.String)
    BOX2D_DATASET_LABEL_NAME = DatasetTableColumn("box2d_dataset_label_name", pl.String)
    BOX2D_LABEL_NAME = DatasetTableColumn("box2d_label_name", pl.String)
    BOX2D_ATTRIBUTES = DatasetTableColumn("box2d_attributes", pl.List(pl.String))
    BOX2D_IMAGE_FRAME_ID = DatasetTableColumn("box2d_image_frame_id", pl.String)


class Box2DDataModel(BaseModel, DataModelInterface):
    """
    Data model to represent annotation data for a 2D bounding box of a camera image.

    Attributes:
      box2d_params: Box ``(x_min, y_min, x_max, y_max)`` in pixels of the raw image.
      box2d_instance_id: Instance ID of the annotated object.
      box2d_dataset_label_name: Label name the dataset gives the box.
      box2d_label_name: Fine label name of the box, None for a dataset label outside every
        taxonomy. The tasks resolve it to a class index when a sample loads.
      box2d_attributes: Attributes of the box.
      box2d_image_frame_id: ID of the image frame of the record the box is annotated in.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    box2d_params: Float64[np.ndarray, " 4"]
    box2d_instance_id: str
    box2d_dataset_label_name: str
    box2d_label_name: str | None
    box2d_attributes: set[str]
    box2d_image_frame_id: str

    @property
    def box2d_params_fp32(self) -> Float32[np.ndarray, " 4"]:
        """
        Convert the box2d_params to float32.

        Returns:
          Float32[np.ndarray, " 4"]: Box parameters in float32.
        """
        return self.box2d_params.astype(np.float32)

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the 2D bounding box data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the 2D bounding box data model.
        """
        return {
            Box2DDatasetSchema.BOX2D_PARAMS.name: self.box2d_params_fp32,
            Box2DDatasetSchema.BOX2D_INSTANCE_ID.name: self.box2d_instance_id,
            Box2DDatasetSchema.BOX2D_DATASET_LABEL_NAME.name: self.box2d_dataset_label_name,
            Box2DDatasetSchema.BOX2D_LABEL_NAME.name: self.box2d_label_name,
            Box2DDatasetSchema.BOX2D_ATTRIBUTES.name: sorted(self.box2d_attributes),
            Box2DDatasetSchema.BOX2D_IMAGE_FRAME_ID.name: self.box2d_image_frame_id,
        }

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> Box2DDataModel:
        """
        Load the 2D bounding box data model from a dictionary, which is deserialized from a
        Polars dataframe.

        Args:
          data_model: Dictionary representation of the 2D bounding box data model.

        Returns:
          Box2DDataModel: The data model of the box.
        """
        return cls(
            box2d_params=np.asarray(
                data_model[Box2DDatasetSchema.BOX2D_PARAMS.name], dtype=np.float64
            ),
            box2d_instance_id=data_model[Box2DDatasetSchema.BOX2D_INSTANCE_ID.name],
            box2d_dataset_label_name=data_model[Box2DDatasetSchema.BOX2D_DATASET_LABEL_NAME.name],
            box2d_label_name=data_model[Box2DDatasetSchema.BOX2D_LABEL_NAME.name],
            box2d_attributes=set(data_model[Box2DDatasetSchema.BOX2D_ATTRIBUTES.name]),
            box2d_image_frame_id=data_model[Box2DDatasetSchema.BOX2D_IMAGE_FRAME_ID.name],
        )
