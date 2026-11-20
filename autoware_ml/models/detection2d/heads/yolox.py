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

"""Decoupled anchor-free detection head of YOLOX.

Ported from the YOLOX reference implementation (``yolox/models/yolo_head.py`` and
``yolox/models/losses.py`` of https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0).
The submodule names follow the reference, so its checkpoints load after a key rename.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

from jaxtyping import Float, Float32, Int64
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, nms

from autoware_ml.dataclasses.models.detection2d.head_outputs import YOLOXHeadOutputs
from autoware_ml.dataclasses.models.detection2d.predictions import (
    Detection2DSamplePredictions,
)
from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D
from autoware_ml.models.common.layers.csp import ConvBNAct
from autoware_ml.models.detection2d.task_modules.simota import SimOTAAssigner

# Largest log size the box decoding exponentiates: a box of exp(10) strides is far larger than
# any image, and an unbounded exponent of an untrained head overflows float32.
_MAX_LOG_SIZE = 10.0


def decode_boxes(
    predictions: Float32[Tensor, "*batch num_priors num_outputs"],
    priors: Float32[Tensor, "num_priors 3"],
) -> Float32[Tensor, "*batch num_priors 4"]:
    """Decode the raw box regression of every prior into pixels.

    Args:
        predictions: Raw outputs whose first four values are ``(dx, dy, log_w, log_h)`` in
          units of the stride of the prior.
        priors: ``(grid_x, grid_y, stride)`` of every prior.

    Returns:
        Boxes ``(cx, cy, w, h)`` in pixels of the network input.
    """
    strides = priors[:, 2:3]
    centers = (predictions[..., :2] + priors[:, :2]) * strides
    sizes = torch.exp(predictions[..., 2:4].clamp(max=_MAX_LOG_SIZE)) * strides
    return torch.cat([centers, sizes], dim=-1)


class YOLOXHead(nn.Module):
    """Decoupled head: a classification and a regression branch per feature level.

    Every level predicts, per position, a box ``(dx, dy, log_w, log_h)`` relative to the
    position and the stride, an objectness logit and one logit per class. Training assigns the
    positions with SimOTA and sums an IoU loss, an objectness and a class binary cross
    entropy, and, once :attr:`use_l1` is set for the last epochs without strong
    augmentation, an L1 loss on the raw box regression.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Sequence[int] = (128, 256, 512),
        feat_channels: int = 128,
        strides: Sequence[int] = (8, 16, 32),
        activation: str = "silu",
        prior_probability: float = 0.01,
        iou_loss_weight: float = 5.0,
        use_l1: bool = False,
        assigner: SimOTAAssigner | None = None,
    ) -> None:
        """Initialize the head.

        Args:
            num_classes: Number of classes.
            in_channels: Channels of the input feature map of every level.
            feat_channels: Channels of the two branches.
            strides: Stride of every level relative to the network input.
            activation: Activation name of every block.
            prior_probability: Initial objectness and class probability the output biases are
              set for.
            iou_loss_weight: Weight of the IoU loss in the total loss.
            use_l1: Whether the L1 loss on the raw box regression is on from the start.
            assigner: Label assigner, SimOTA with the reference settings when None.

        Raises:
            ValueError: If the levels and the strides differ in count.
        """
        super().__init__()
        if len(in_channels) != len(strides):
            raise ValueError(
                f"Got {len(in_channels)} input levels for {len(strides)} strides, one stride "
                "is needed per level."
            )
        self.num_classes = num_classes
        self.strides = tuple(int(stride) for stride in strides)
        self.iou_loss_weight = iou_loss_weight
        self.use_l1 = use_l1
        self.assigner = assigner if assigner is not None else SimOTAAssigner()

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for channels in in_channels:
            self.stems.append(ConvBNAct(int(channels), feat_channels, 1, 1, activation=activation))
            self.cls_convs.append(
                nn.Sequential(
                    ConvBNAct(feat_channels, feat_channels, 3, 1, activation=activation),
                    ConvBNAct(feat_channels, feat_channels, 3, 1, activation=activation),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvBNAct(feat_channels, feat_channels, 3, 1, activation=activation),
                    ConvBNAct(feat_channels, feat_channels, 3, 1, activation=activation),
                )
            )
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, 1))
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, 1))
        self.initialize_biases(prior_probability)

    def initialize_biases(self, prior_probability: float) -> None:
        """Set the objectness and class biases so every prior starts at a low probability.

        Args:
            prior_probability: Initial probability of the sigmoid outputs.
        """
        bias = -math.log((1.0 - prior_probability) / prior_probability)
        for conv in [*self.cls_preds, *self.obj_preds]:
            nn.init.constant_(conv.bias, bias)

    def forward(
        self, features: Sequence[Float[Tensor, "batch_size channels height width"]]
    ) -> list[Float[Tensor, "batch_size num_outputs height width"]]:
        """Predict the raw outputs of every level.

        Args:
            features: Feature map of every level, finest first.

        Returns:
            One map per level with the box regression, the objectness logit and the class
            logits along the channels.

        Raises:
            ValueError: If the number of feature maps differs from the number of levels.
        """
        if len(features) != len(self.strides):
            raise ValueError(f"Expected {len(self.strides)} feature maps, got {len(features)}.")
        outputs = []
        for level, feature in enumerate(features):
            x = self.stems[level](feature)
            cls_output = self.cls_preds[level](self.cls_convs[level](x))
            reg_feature = self.reg_convs[level](x)
            reg_output = self.reg_preds[level](reg_feature)
            obj_output = self.obj_preds[level](reg_feature)
            outputs.append(torch.cat([reg_output, obj_output, cls_output], dim=1))
        return outputs

    def build_priors(
        self, level_sizes: Sequence[tuple[int, int]], device: torch.device
    ) -> Float32[Tensor, "num_priors 3"]:
        """Build the priors of feature levels of the given sizes.

        Args:
            level_sizes: ``(height, width)`` of every level, finest first.
            device: Device of the priors.

        Returns:
            ``(grid_x, grid_y, stride)`` of every prior, level after level, row by row.
        """
        priors = []
        for (height, width), stride in zip(level_sizes, self.strides, strict=True):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=device),
                torch.arange(width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            priors.append(
                torch.stack(
                    [grid_x, grid_y, torch.full_like(grid_x, float(stride))], dim=-1
                ).reshape(-1, 3)
            )
        return torch.cat(priors, dim=0)

    def flatten_outputs(
        self,
        level_outputs: Sequence[Float[Tensor, "batch_size num_outputs height width"]],
    ) -> YOLOXHeadOutputs:
        """Flatten the maps of every level into one prediction per prior.

        The predictions are cast to float32: the losses, the assignment and the box decoding
        are not safe in half precision.

        Args:
            level_outputs: Raw output map of every level, as :meth:`forward` returns them.

        Returns:
            The raw predictions of every prior and the priors.
        """
        predictions = torch.cat(
            [output.flatten(start_dim=2) for output in level_outputs], dim=2
        ).permute(0, 2, 1)
        priors = self.build_priors(
            [(output.shape[2], output.shape[3]) for output in level_outputs],
            predictions.device,
        )
        return YOLOXHeadOutputs(predictions=predictions.to(torch.float32), priors=priors)

    def export_outputs(
        self,
        level_outputs: Sequence[Float[Tensor, "batch_size num_outputs height width"]],
    ) -> Float[Tensor, "batch_size num_priors num_outputs"]:
        """Build the output tensor of the deployed graph.

        The box regression stays raw and the objectness and class logits pass a sigmoid, the
        layout the YOLOX reference exports without its in-graph decoding: the runtime adds the
        grid, multiplies by the stride and exponentiates the size.

        Args:
            level_outputs: Raw output map of every level, as :meth:`forward` returns them.

        Returns:
            One row per prior: ``(dx, dy, log_w, log_h, objectness, class scores...)``.
        """
        flattened = [
            torch.cat([output[:, :4], output[:, 4:].sigmoid()], dim=1).flatten(start_dim=2)
            for output in level_outputs
        ]
        return torch.cat(flattened, dim=2).permute(0, 2, 1)

    def loss(
        self,
        outputs: YOLOXHeadOutputs,
        gt_boxes: Sequence[Float32[Tensor, "num_gt 4"]],
        gt_labels: Sequence[Int64[Tensor, " num_gt"]],
    ) -> dict[str, Tensor]:
        """Compute the losses of a batch.

        Every loss is summed over the batch and divided by the number of foreground priors of
        the batch (at least 1).

        Args:
            outputs: Raw predictions of the batch.
            gt_boxes: Ground truth boxes ``(x_min, y_min, x_max, y_max)`` of every image, in
              pixels of the network input.
            gt_labels: Class index of every ground truth box of every image.

        Returns:
            ``loss`` and its parts ``loss_iou`` (unweighted), ``loss_obj``, ``loss_cls`` and
            ``loss_l1``, plus ``num_fg``, the mean number of foreground priors per ground
            truth box.
        """
        # The assignment and the losses are not safe in half precision, whatever the trainer
        # runs the network in.
        with torch.autocast(device_type=outputs.predictions.device.type, enabled=False):
            return self._loss(outputs, gt_boxes, gt_labels)

    def _loss(
        self,
        outputs: YOLOXHeadOutputs,
        gt_boxes: Sequence[Float32[Tensor, "num_gt 4"]],
        gt_labels: Sequence[Int64[Tensor, " num_gt"]],
    ) -> dict[str, Tensor]:
        """Compute the losses of a batch in full precision, see :meth:`loss`."""
        predictions = outputs.predictions
        priors = outputs.priors
        pred_boxes = decode_boxes(predictions, priors)
        objectness_logits = predictions[..., 4:5]
        class_logits = predictions[..., 5:]

        foreground_masks = []
        class_targets = []
        box_targets = []
        l1_targets = []
        num_gt_total = 0
        for index, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels, strict=True)):
            boxes_cxcywh = BaseBBoxes2D(boxes.to(torch.float32)).cxcywh
            num_gt_total += boxes.shape[0]
            assignment = self.assigner.assign(
                pred_boxes[index].detach(),
                objectness_logits[index].detach(),
                class_logits[index].detach(),
                priors,
                boxes_cxcywh,
                labels,
            )
            matched = assignment.matched_gt_indices
            foreground_masks.append(assignment.foreground_mask)
            class_targets.append(
                F.one_hot(labels[matched], self.num_classes).to(torch.float32)
                * assignment.matched_ious[:, None]
            )
            box_targets.append(boxes_cxcywh[matched])
            if self.use_l1:
                foreground_priors = priors[assignment.foreground_mask]
                strides = foreground_priors[:, 2:3]
                l1_targets.append(
                    torch.cat(
                        [
                            boxes_cxcywh[matched, :2] / strides - foreground_priors[:, :2],
                            torch.log(boxes_cxcywh[matched, 2:] / strides + 1e-8),
                        ],
                        dim=1,
                    )
                )

        foreground_mask = torch.stack(foreground_masks)
        class_target = torch.cat(class_targets)
        box_target = torch.cat(box_targets)
        num_foreground = max(int(foreground_mask.sum()), 1)

        ious = BaseBBoxes2D.from_cxcywh(pred_boxes[foreground_mask]).iou(
            BaseBBoxes2D.from_cxcywh(box_target), is_aligned=True, eps=1e-16
        )
        loss_iou = (1.0 - ious**2).sum() / num_foreground
        loss_obj = (
            F.binary_cross_entropy_with_logits(
                objectness_logits,
                foreground_mask[..., None].to(torch.float32),
                reduction="sum",
            )
            / num_foreground
        )
        loss_cls = (
            F.binary_cross_entropy_with_logits(
                class_logits[foreground_mask], class_target, reduction="sum"
            )
            / num_foreground
        )
        if self.use_l1:
            loss_l1 = (
                F.l1_loss(
                    predictions[..., :4][foreground_mask],
                    torch.cat(l1_targets),
                    reduction="sum",
                )
                / num_foreground
            )
        else:
            loss_l1 = predictions.new_zeros(())
        loss = self.iou_loss_weight * loss_iou + loss_obj + loss_cls + loss_l1
        return {
            "loss": loss,
            "loss_iou": loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_l1": loss_l1,
            "num_fg": predictions.new_tensor(int(foreground_mask.sum()) / max(num_gt_total, 1)),
        }

    @torch.no_grad()
    def predict(
        self,
        outputs: YOLOXHeadOutputs,
        score_threshold: float,
        nms_iou_threshold: float,
        class_agnostic_nms: bool = False,
        max_detections: int | None = None,
    ) -> list[Detection2DSamplePredictions]:
        """Decode the raw predictions into the detections of every image.

        Every prior proposes its best class with the score ``objectness * class score``. The
        proposals above the score threshold pass non-maximum suppression.

        Args:
            outputs: Raw predictions of the batch.
            score_threshold: Lowest score of a detection.
            nms_iou_threshold: IoU above which the lower scored of two boxes is suppressed.
            class_agnostic_nms: Whether boxes of different classes suppress each other.
            max_detections: Number of highest scored detections kept per image, all when None.

        Returns:
            The detections of every image, boxes in pixels of the network input.
        """
        return postprocess_detections(
            decode_boxes(outputs.predictions, outputs.priors),
            outputs.predictions[..., 4].sigmoid(),
            outputs.predictions[..., 5:].sigmoid(),
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            class_agnostic_nms=class_agnostic_nms,
            max_detections=max_detections,
        )


@torch.no_grad()
def postprocess_detections(
    boxes_cxcywh: Float32[Tensor, "batch_size num_priors 4"],
    objectness: Float32[Tensor, "batch_size num_priors"],
    class_scores: Float32[Tensor, "batch_size num_priors num_classes"],
    score_threshold: float,
    nms_iou_threshold: float,
    class_agnostic_nms: bool = False,
    max_detections: int | None = None,
) -> list[Detection2DSamplePredictions]:
    """Turn decoded dense predictions into the detections of every image.

    Args:
        boxes_cxcywh: Decoded boxes ``(cx, cy, w, h)`` of every prior, in pixels.
        objectness: Objectness probability of every prior.
        class_scores: Class probabilities of every prior.
        score_threshold: Lowest score of a detection.
        nms_iou_threshold: IoU above which the lower scored of two boxes is suppressed.
        class_agnostic_nms: Whether boxes of different classes suppress each other.
        max_detections: Number of highest scored detections kept per image, all when None.

    Returns:
        The detections of every image, sorted by decreasing score.
    """
    detections = []
    for image_boxes, image_objectness, image_class_scores in zip(
        boxes_cxcywh, objectness, class_scores, strict=True
    ):
        best_scores, labels = image_class_scores.max(dim=1)
        scores = image_objectness * best_scores
        keep = scores >= score_threshold
        boxes = BaseBBoxes2D.from_cxcywh(image_boxes[keep].to(torch.float32)).bbox_params
        scores = scores[keep].to(torch.float32)
        labels = labels[keep]
        if class_agnostic_nms:
            kept = nms(boxes, scores, nms_iou_threshold)
        else:
            kept = batched_nms(boxes, scores, labels, nms_iou_threshold)
        if max_detections is not None:
            kept = kept[:max_detections]
        detections.append(
            Detection2DSamplePredictions(
                bboxes_2d=boxes[kept], scores_2d=scores[kept], labels_2d=labels[kept]
            )
        )
    return detections
