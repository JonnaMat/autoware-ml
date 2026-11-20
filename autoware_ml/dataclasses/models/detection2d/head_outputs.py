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
Modules to save raw outputs from a detection2d head.
"""

from __future__ import annotations

from jaxtyping import Float32
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class FocalHead2DOutputs(BaseModel):
    """
    Dataclass to save the dense outputs of an auxiliary 2D head over the image tokens.

    Every camera image of every sample gets one prediction per feature map token, the images
    ordered sample by sample.

    Attributes:
      cls_scores: Class logits of every token.
      bbox_preds: Box ``(cx, cy, w, h)`` of every token, normalized by the image size.
      centers_2d: Projected 3D center of every token, normalized by the image size.
      heatmap: Center heatmap logit of every token, one per class.
      feature_height: Rows of the token grid of every image; the tokens are flattened row by
        row, so the heatmap targets are drawn on this grid.
      feature_width: Columns of the token grid of every image.
      image_height: Height in pixels of the images the predictions are normalized by.
      image_width: Width in pixels of the images the predictions are normalized by.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    cls_scores: Float32[torch.Tensor, "num_images num_tokens num_classes"]
    bbox_preds: Float32[torch.Tensor, "num_images num_tokens 4"]
    centers_2d: Float32[torch.Tensor, "num_images num_tokens 2"]
    heatmap: Float32[torch.Tensor, "num_images num_tokens num_classes"]
    feature_height: int
    feature_width: int
    image_height: int
    image_width: int

    @model_validator(mode="after")
    def _check_images_and_tokens(self) -> FocalHead2DOutputs:
        """Check that every prediction covers the same images and the tokens of the grid."""
        images_and_tokens = self.cls_scores.shape[:2]
        for name, tensor in (
            ("bbox_preds", self.bbox_preds),
            ("centers_2d", self.centers_2d),
            ("heatmap", self.heatmap),
        ):
            if tensor.shape[:2] != images_and_tokens:
                raise ValueError(
                    f"{name} must cover {tuple(images_and_tokens)} images and tokens, got "
                    f"{tuple(tensor.shape)}."
                )
        if self.heatmap.shape[2] != self.cls_scores.shape[2]:
            raise ValueError(
                f"heatmap must have one channel per class of cls_scores, "
                f"{self.cls_scores.shape[2]}, got {self.heatmap.shape[2]}."
            )
        if self.feature_height * self.feature_width != images_and_tokens[1]:
            raise ValueError(
                f"The predictions cover {images_and_tokens[1]} tokens per image but the token "
                f"grid is {self.feature_height} x {self.feature_width}."
            )
        return self


class YOLOXHeadOutputs(BaseModel):
    """
    Dataclass to save the raw dense outputs of a YOLOX head.

    Every image gets one prediction per prior, the priors of the feature levels concatenated
    from the finest level to the coarsest and flattened row by row.

    Attributes:
      predictions: Raw outputs of every prior: the box regression ``(dx, dy, log_w, log_h)`` in
        units of the stride of the prior, the objectness logit, then one logit per class.
      priors: ``(grid_x, grid_y, stride)`` of every prior, the grid position being the column
        and the row of the prior on its feature level.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    predictions: Float32[torch.Tensor, "batch_size num_priors num_outputs"]
    priors: Float32[torch.Tensor, "num_priors 3"]

    @model_validator(mode="after")
    def _check_priors(self) -> YOLOXHeadOutputs:
        """Check that the predictions cover the priors and carry at least one class."""
        if self.predictions.shape[1] != self.priors.shape[0]:
            raise ValueError(
                f"predictions cover {self.predictions.shape[1]} priors, priors hold "
                f"{self.priors.shape[0]}."
            )
        if self.predictions.shape[2] < 6:
            raise ValueError(
                "predictions need 4 box values, 1 objectness and at least 1 class logit per "
                f"prior, got {self.predictions.shape[2]} values."
            )
        return self
