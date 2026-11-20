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

"""
Modules to save decoded predictions from a detection2d model.
"""

from __future__ import annotations

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict, model_validator
import torch


class Detection2DSamplePredictions(BaseModel):
    """
    Dataclass to save the decoded 2D detections of one image.

    Attributes:
      bboxes_2d: Boxes ``(x_min, y_min, x_max, y_max)`` in pixels.
      scores_2d: Confidence of every box.
      labels_2d: Class index of every box.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    bboxes_2d: Float32[torch.Tensor, "num_boxes 4"]
    scores_2d: Float32[torch.Tensor, " num_boxes"]
    labels_2d: Int64[torch.Tensor, " num_boxes"]

    @model_validator(mode="after")
    def _check_num_boxes(self) -> Detection2DSamplePredictions:
        """Check that boxes, scores and labels describe the same number of boxes."""
        num_boxes = self.bboxes_2d.shape[0]
        if self.scores_2d.shape[0] != num_boxes or self.labels_2d.shape[0] != num_boxes:
            raise ValueError(
                "bboxes_2d, scores_2d and labels_2d must have the same number of boxes, got "
                f"{num_boxes}, {self.scores_2d.shape[0]} and {self.labels_2d.shape[0]}."
            )
        return self
