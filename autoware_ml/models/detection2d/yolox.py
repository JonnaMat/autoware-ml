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

"""YOLOX 2D object detector.

The detector composes a backbone, a neck and the YOLOX head. The backbone returns the feature
maps at strides 8, 16 and 32 and the neck fuses them, so another backbone or neck is a config
change. A backbone that also returns the map at stride 4 gives the neck and the head a fourth
level. The training recipe follows the YOLOX reference implementation
(https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0): an exponential moving average of
the weights is kept and evaluated, and the L1 box loss is turned on for the last epochs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jaxtyping import Float, Float32
import torch
from torch import Tensor
import torch.nn as nn

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTBatch
from autoware_ml.dataclasses.models.detection2d.head_outputs import YOLOXHeadOutputs
from autoware_ml.dataclasses.models.detection2d.predictions import (
    Detection2DSamplePredictions,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.eval_output import (
    Detection2DEvalFrame,
    Detection2DEvalOutput,
    EvalOutput,
)
from autoware_ml.models.common.ema import ModelEMA
from autoware_ml.models.detection2d.heads.yolox import YOLOXHead
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.utils.deploy import ExportSpec


class _DetectorNetwork(nn.Module):
    """Backbone, neck and head of the detector as one module, the unit the average copies."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: YOLOXHead,
        input_mean: Sequence[float] | None = None,
        input_std: Sequence[float] | None = None,
    ) -> None:
        """Hold the three stages and the normalization of the input.

        Args:
            backbone: Image backbone.
            neck: Neck fusing the feature maps of the backbone.
            head: YOLOX head.
            input_mean: Per-channel mean subtracted from the image, None for none.
            input_std: Per-channel divisor of the image, None for none.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.normalize = input_mean is not None or input_std is not None
        # Constants of the config, not weights: they stay out of the state dict.
        self.register_buffer(
            "input_mean",
            torch.tensor(input_mean or [0.0, 0.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(input_std or [1.0, 1.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self, images: Float[Tensor, "batch_size 3 height width"]
    ) -> list[Float[Tensor, "batch_size num_outputs level_height level_width"]]:
        """Predict the raw output map of every level."""
        if self.normalize:
            images = (images - self.input_mean) / self.input_std
        return self.head(self.neck(self.backbone(images)))


def _invert_axis_aligned_affine(
    boxes: Float32[Tensor, "num_boxes 4"], matrix: Float32[Tensor, "4 4"]
) -> Float32[Tensor, "num_boxes 4"]:
    """Map boxes of the network input back onto the raw image.

    Args:
        boxes: Boxes ``(x_min, y_min, x_max, y_max)`` in pixels of the network input.
        matrix: Image augmentation matrix of the image, mapping raw pixels onto the input.

    Returns:
        The boxes in pixels of the raw image.

    Raises:
        ValueError: If the matrix rotates, shears or mirrors, which an axis aligned box does
          not survive.
    """
    scale_x, scale_y = matrix[0, 0], matrix[1, 1]
    if matrix[0, 1].abs() > 1e-6 or matrix[1, 0].abs() > 1e-6 or scale_x <= 0 or scale_y <= 0:
        raise ValueError(
            "Detections are mapped back through a scale and a translation only, the "
            f"evaluation pipeline applied the image augmentation {matrix[:2].tolist()}."
        )
    offset = matrix[:2, 3].repeat(2)
    scale = torch.stack([scale_x, scale_y]).repeat(2)
    return (boxes - offset) / scale


class YOLOXDetector(ModuleBaseModel):
    """Single-camera 2D detector with a YOLOX head.

    The images reach the model as RGB in ``[0, 255]`` and are reordered to
    ``input_channel_order`` before the backbone, BGR for the weights of the reference
    implementation. In eval mode the averaged weights predict when the average is on.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: YOLOXHead,
        log_dict_configs: LogDictConfigs,
        input_channel_order: str = "bgr",
        input_mean: Sequence[float] | None = None,
        input_std: Sequence[float] | None = None,
        use_ema: bool = True,
        ema_decay: float = 0.9998,
        ema_warmup_updates: int = 2000,
        score_threshold: float = 0.01,
        nms_iou_threshold: float = 0.65,
        class_agnostic_nms: bool = False,
        max_detections: int | None = 300,
        export_background_class: bool = False,
        export_input_channel_order: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the detector.

        Args:
            backbone: Image backbone returning the feature maps at strides 8, 16 and 32,
              preceded by the one at stride 4 for a detector of four levels.
            neck: Neck fusing the feature maps.
            head: YOLOX head.
            log_dict_configs: Configuration for logging metrics in Lightning.
            input_channel_order: Channel order the backbone reads, ``bgr`` or ``rgb``.
            input_mean: Per-channel mean subtracted from the image inside the network, in the
              channel order the backbone reads and the ``[0, 255]`` range. None for a
              backbone reading raw pixel values, as the YOLOX weights do.
            input_std: Per-channel divisor applied after the mean, None for none.
            use_ema: Whether to keep and evaluate an exponential moving average of the weights.
            ema_decay: Decay of the average.
            ema_warmup_updates: Time constant of the ramp of the decay, in updates.
            score_threshold: Lowest score of a detection.
            nms_iou_threshold: IoU above which the lower scored of two boxes is suppressed.
            class_agnostic_nms: Whether boxes of different classes suppress each other.
            max_detections: Number of highest scored detections kept per image, all when None.
            export_background_class: Whether the deployed graph outputs a leading background
              class score that is always zero, for runtimes whose label list starts with a
              background entry.
            export_input_channel_order: Channel order the deployed graph reads, ``bgr`` or
              ``rgb``. When it differs from ``input_channel_order`` the graph reorders the
              channels itself, so every backbone is deployed behind the same input. None
              deploys the order the backbone reads.
            **kwargs: Arguments of :class:`ModuleBaseModel`.

        Raises:
            ValueError: If a channel order is neither ``bgr`` nor ``rgb``.
        """
        super().__init__(log_dict_configs=log_dict_configs, **kwargs)
        self.network = _DetectorNetwork(backbone, neck, head, input_mean, input_std)
        if input_channel_order not in ("bgr", "rgb"):
            raise ValueError(
                f"input_channel_order must be bgr or rgb, got {input_channel_order!r}."
            )
        self.input_channel_order = input_channel_order
        if export_input_channel_order not in (None, "bgr", "rgb"):
            raise ValueError(
                "export_input_channel_order must be bgr, rgb or None, got "
                f"{export_input_channel_order!r}."
            )
        self.export_input_channel_order = export_input_channel_order or input_channel_order
        self.ema = (
            ModelEMA(self.network, decay=ema_decay, warmup_updates=ema_warmup_updates)
            if use_ema
            else None
        )
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.class_agnostic_nms = class_agnostic_nms
        self.max_detections = max_detections
        self.export_background_class = export_background_class

    @property
    def head(self) -> YOLOXHead:
        """Head of the trained network."""
        return self.network.head

    def inference_network(self) -> nn.Module:
        """Network that predicts outside training: the averaged one when the average is on."""
        if self.ema is not None and not self.training:
            return self.ema.module
        return self.network

    def on_fit_start(self) -> None:
        """Start the average from the weights the run starts with, pretrained ones included."""
        if self.ema is not None and int(self.ema.num_updates.item()) == 0:
            self.ema.copy_from(self.network)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update the average after every training step."""
        if self.ema is not None:
            self.ema.update(self.network)

    def set_strong_augmentation(self, enabled: bool) -> None:
        """Follow the strong augmentation of the training data: without it the L1 loss is on.

        Args:
            enabled: Whether the training samples are still strongly augmented.
        """
        self.head.use_l1 = not enabled

    def prepare_images(
        self, batch_inputs: ModelBatchInputs
    ) -> Float[Tensor, "batch_size 3 height width"]:
        """Read the image of the one camera of every sample, in the order the backbone reads.

        Raises:
            ValueError: If the batch carries no images or more than one camera per sample.
        """
        if batch_inputs.image_data is None:
            raise ValueError("The detector needs the images of the batch.")
        images = batch_inputs.image_data.images
        if images.shape[1] != 1:
            raise ValueError(
                f"The detector reads one camera per sample, the batch carries {images.shape[1]}."
            )
        images = images[:, 0]
        if self.input_channel_order == "bgr":
            images = images.flip(1)
        return images

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Predict the raw outputs of every prior of every image."""
        level_outputs = self.inference_network()(self.prepare_images(multi_task_batch_inputs))
        return ModelOutputs(detection2d_head_outputs=self.head.flatten_outputs(level_outputs))

    @staticmethod
    def _ground_truth(batch_inputs: ModelBatchInputs) -> Detection2DGTBatch:
        """Read the 2D boxes of the batch."""
        gt_batch = batch_inputs.multi_task_gt_batch.detection2d_gt_batch
        if gt_batch is None:
            raise ValueError("The detector needs the 2D boxes of the batch.")
        return gt_batch

    def compute_metrics(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        multi_task_outputs: ModelOutputs,
    ) -> Mapping[str, Tensor]:
        """Compute the losses of the batch."""
        gt_batch = self._ground_truth(multi_task_batch_inputs)
        gt_boxes = [cameras[0][:, :4] for cameras in gt_batch.valid_bboxes_2d()]
        gt_labels = [cameras[0] for cameras in gt_batch.valid_labels_2d()]
        return self.head.loss(multi_task_outputs.detection2d(), gt_boxes, gt_labels)

    def _detect(self, outputs: YOLOXHeadOutputs) -> list[Detection2DSamplePredictions]:
        """Decode the raw outputs into detections in pixels of the network input."""
        return self.head.predict(
            outputs,
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            class_agnostic_nms=self.class_agnostic_nms,
            max_detections=self.max_detections,
        )

    def _detect_in_raw_pixels(
        self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> list[Detection2DSamplePredictions]:
        """Decode the raw outputs into detections in pixels of the raw images."""
        if batch_inputs.image_data is None:
            raise ValueError("The detector needs the images of the batch.")
        matrices = batch_inputs.image_data.image_augmentation_matrices[:, 0]
        return [
            Detection2DSamplePredictions(
                bboxes_2d=_invert_axis_aligned_affine(detections.bboxes_2d, matrix),
                scores_2d=detections.scores_2d,
                labels_2d=detections.labels_2d,
            )
            for detections, matrix in zip(
                self._detect(outputs.detection2d()), matrices, strict=True
            )
        ]

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode the raw outputs into the detections of every image, in raw image pixels."""
        return ModelPredictions(
            detection2d_predictions=self._detect_in_raw_pixels(multi_task_batch_inputs, outputs)
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> EvalOutput:
        """Pair the detections of every image with its ground truth, in raw image pixels."""
        gt_batch = self._ground_truth(batch)
        if batch.image_data is None:
            raise ValueError("The detector needs the images of the batch.")
        matrices = batch.image_data.image_augmentation_matrices[:, 0]
        padding = batch.multi_task_gt_batch.padding_mask.tolist()
        frames = [
            Detection2DEvalFrame(
                predictions=detections,
                gt_boxes=_invert_axis_aligned_affine(gt_boxes[0][:, :4], matrix),
                gt_labels=gt_labels[0],
                is_padding=is_padding,
            )
            for detections, gt_boxes, gt_labels, matrix, is_padding in zip(
                self._detect_in_raw_pixels(batch, outputs),
                gt_batch.valid_bboxes_2d(),
                gt_batch.valid_labels_2d(),
                matrices,
                padding,
                strict=True,
            )
        ]
        return EvalOutput(detection2d=Detection2DEvalOutput(frames=frames))

    def build_optimizer_groups(self) -> Mapping[str, Sequence[nn.Parameter]]:
        """Split the trained parameters by whether weight decay applies.

        Biases and normalization weights, the parameters of fewer than two dimensions, are
        not decayed. The averaged copy is never optimized.

        Returns:
            The groups ``decay`` and ``no_decay``.
        """
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for parameter in self.network.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        return {"decay": decay, "no_decay": no_decay}

    def build_export_spec(self, multi_task_batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build the export specification of the detector.

        The graph takes the image in ``export_input_channel_order``, the order of the backbone
        unless configured, and returns one row per prior: the raw box regression, the
        objectness and the class scores.

        Args:
            multi_task_batch_inputs: Example preprocessed batch used for export.

        Returns:
            Export specification of the end to end graph.
        """
        flip_channels = self.export_input_channel_order != self.input_channel_order
        images = self.prepare_images(multi_task_batch_inputs)
        return ExportSpec(
            module=YOLOXExportModule(
                self.inference_network(), self.export_background_class, flip_channels
            ),
            args=(images.flip(1) if flip_channels else images,),
            # The tensor names the YOLOX runtimes bind.
            input_names=["images"],
            output_names=["output"],
        )


class YOLOXExportModule(nn.Module):
    """Deployed graph of the detector: image in, one row per prior out."""

    def __init__(
        self, network: nn.Module, background_class: bool = False, flip_channels: bool = False
    ) -> None:
        """Initialize the export wrapper.

        Args:
            network: Backbone, neck and head predicting the raw output map of every level.
            background_class: Whether to output a leading class score that is always zero.
            flip_channels: Whether the graph reverses the channels of its input, for a
              backbone reading the other channel order than the runtime delivers.
        """
        super().__init__()
        self.network = network
        self.background_class = background_class
        self.flip_channels = flip_channels
        # The exporter restores the mode the module had before the export. A new module
        # starts in training mode, which would switch the wrapped network to training.
        self.train(network.training)

    def forward(
        self, images: Float[Tensor, "batch_size 3 height width"]
    ) -> Float[Tensor, "batch_size num_priors num_outputs"]:
        """Predict ``(dx, dy, log_w, log_h, objectness, class scores...)`` of every prior."""
        if self.flip_channels:
            images = images.flip(1)
        output = self.network.head.export_outputs(self.network(images))
        if not self.background_class:
            return output
        return torch.cat(
            [output[..., :5], torch.zeros_like(output[..., :1]), output[..., 5:]],
            dim=-1,
        )
