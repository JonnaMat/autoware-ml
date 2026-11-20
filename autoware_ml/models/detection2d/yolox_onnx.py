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

"""YOLOX detector whose network is an ONNX file, for scoring deployed graphs.

The wrapper runs an exported graph with ONNX Runtime and hands its output to the decoding and
the metrics of :class:`YOLOXDetector`, so a deployed model is scored on the same data, with
the same preprocessing and by the same metric code as a checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection2d.head_outputs import YOLOXHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.models.detection2d.heads.yolox import YOLOXHead
from autoware_ml.models.detection2d.yolox import YOLOXDetector
from autoware_ml.models.module_base_model import LogDictConfigs

# Probabilities are turned back into logits, clipped so that 0 and 1 stay finite.
_PROBABILITY_EPS = 1e-7


class ONNXYOLOXDetector(YOLOXDetector):
    """Evaluate an ONNX file with the I/O contract of the exported YOLOX detector.

    The graph takes one image ``(1, 3, height, width)`` in the channel order and value range
    of the detector and returns one row per prior: the raw box regression, the objectness
    probability and the class probabilities, the priors ordered level by level. The graph may
    score more classes than the evaluation knows, a background entry for example:
    ``class_indices`` names the output class of every evaluated class.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        num_classes: int,
        log_dict_configs: LogDictConfigs,
        class_indices: Sequence[int] | None = None,
        strides: Sequence[int] = (8, 16, 32),
        providers: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the wrapper and load the graph.

        Args:
            onnx_path: Path of the ONNX file.
            num_classes: Number of evaluated classes.
            log_dict_configs: Configuration for logging metrics in Lightning.
            class_indices: Class output of the graph of every evaluated class, in class order.
              The first ``num_classes`` outputs when None.
            strides: Stride of every feature level of the graph, finest first.
            providers: ONNX Runtime execution providers in order of preference, CUDA then CPU
              when None. Providers the installation lacks are left out.
            **kwargs: Arguments of :class:`YOLOXDetector`, the postprocessing among them.

        Raises:
            ValueError: If ``class_indices`` does not name one output per evaluated class, or
              the input of the graph is not a static single image.
        """
        head = YOLOXHead(
            num_classes=num_classes,
            in_channels=[1] * len(strides),
            feat_channels=1,
            strides=strides,
        )
        kwargs["use_ema"] = False
        super().__init__(
            backbone=nn.Identity(),
            neck=nn.Identity(),
            head=head,
            log_dict_configs=log_dict_configs,
            **kwargs,
        )
        self.class_indices = (
            list(range(num_classes)) if class_indices is None else [int(i) for i in class_indices]
        )
        if len(self.class_indices) != num_classes:
            raise ValueError(
                f"class_indices names {len(self.class_indices)} outputs for {num_classes} classes."
            )
        available = ort.get_available_providers()
        requested = providers or ("CUDAExecutionProvider", "CPUExecutionProvider")
        self.session = ort.InferenceSession(
            str(onnx_path), providers=[name for name in requested if name in available]
        )
        graph_input = self.session.get_inputs()[0]
        self.input_name = graph_input.name
        shape = graph_input.shape
        if len(shape) != 4 or shape[0] != 1 or not all(isinstance(size, int) for size in shape):
            raise ValueError(
                f"The graph input must be a static (1, 3, height, width) image, got {shape}."
            )
        self.input_size = (int(shape[2]), int(shape[3]))

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the graph on every image of the batch and rebuild the raw head outputs.

        Raises:
            ValueError: If the images are not of the input size of the graph, or the graph
              returns another number of priors than its strides give.
        """
        images = self.prepare_images(multi_task_batch_inputs)
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError(
                f"The graph reads images of {self.input_size}, the pipeline serves "
                f"{tuple(images.shape[-2:])}."
            )
        outputs = np.concatenate(
            [
                self.session.run(None, {self.input_name: image[None].cpu().numpy()})[0]
                for image in images.to(torch.float32)
            ]
        )
        output = torch.from_numpy(outputs).to(device=images.device, dtype=torch.float32)
        priors = self.head.build_priors(
            [
                (self.input_size[0] // stride, self.input_size[1] // stride)
                for stride in self.head.strides
            ],
            images.device,
        )
        if output.shape[1] != priors.shape[0]:
            raise ValueError(
                f"The graph returns {output.shape[1]} priors, its strides {self.head.strides} "
                f"give {priors.shape[0]} for an input of {self.input_size}."
            )
        class_columns = [5 + index for index in self.class_indices]
        probabilities = output[..., [4, *class_columns]].clamp(
            _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS
        )
        predictions = torch.cat([output[..., :4], torch.logit(probabilities)], dim=-1)
        return ModelOutputs(
            detection2d_head_outputs=YOLOXHeadOutputs(predictions=predictions, priors=priors)
        )
