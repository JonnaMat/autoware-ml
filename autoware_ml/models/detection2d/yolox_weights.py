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

"""Conversion of YOLOX weights from other sources into the state dict of ``YOLOXDetector``.

Two sources are supported: a checkpoint of the YOLOX reference implementation
(https://github.com/Megvii-BaseDetection/YOLOX) and an ONNX file exported from it, whose batch
normalizations are folded into the convolutions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import Tensor

from autoware_ml.models.common.layers.csp import BN_EPS

NETWORK_PREFIX = "network."
EMA_PREFIX = "ema.module."
EMA_NUM_UPDATES_KEY = "ema.num_updates"
NETWORK_PARTS = ("backbone", "neck", "head")


def rename_reference_key(key: str) -> str:
    """Rename a state dict key of the reference implementation to the detector network.

    The reference nests the CSPDarknet inside its neck (``backbone.backbone``) and calls the
    neck ``backbone``; here they are the ``backbone`` and the ``neck`` of the network.

    Args:
        key: Key of the reference state dict.

    Returns:
        The key below the detector network, without the ``network.`` prefix.

    Raises:
        ValueError: If the key belongs to none of the three stages.
    """
    if key.startswith("backbone.backbone."):
        return "backbone." + key.removeprefix("backbone.backbone.")
    if key.startswith("backbone."):
        return "neck." + key.removeprefix("backbone.")
    if key.startswith("head."):
        return key
    raise ValueError(f"Unexpected key {key!r} in a YOLOX state dict.")


def shift_head_levels(key: str, offset: int) -> str:
    """Move a key of the head to the level ``offset`` positions coarser in the level order.

    The head numbers its levels from the finest. A detector with a level at stride 4 in front
    reads the weights of a three-level head, whose first level is at stride 8, one level up.

    Args:
        key: Key below the detector network.
        offset: Number of levels the detector has in front of those of the weights.

    Returns:
        The key with the level index of a head module raised by ``offset``, any other key
        as it is.
    """
    parts = key.split(".")
    if offset == 0 or len(parts) < 3 or parts[0] != "head" or not parts[2].isdigit():
        return key
    parts[2] = str(int(parts[2]) + offset)
    return ".".join(parts)


def to_detector_state_dict(
    network_state_dict: Mapping[str, Tensor], with_ema: bool = True
) -> dict[str, Tensor]:
    """Place a network state dict into the state dict layout of the detector.

    Args:
        network_state_dict: State of the backbone, neck and head, keyed below the network.
        with_ema: Whether to also fill the averaged copy of the network with the same weights,
          which an evaluation or an export of the converted weights reads.

    Returns:
        State dict of the detector.
    """
    state_dict = {NETWORK_PREFIX + key: value for key, value in network_state_dict.items()}
    if with_ema:
        state_dict.update(
            {EMA_PREFIX + key: value.clone() for key, value in network_state_dict.items()}
        )
        state_dict[EMA_NUM_UPDATES_KEY] = torch.zeros((), dtype=torch.long)
    return state_dict


def select_parts(state_dict: Mapping[str, Tensor], parts: Sequence[str]) -> dict[str, Tensor]:
    """Keep the tensors of some parts of the network in a state dict of the detector.

    Weights reduced to the backbone initialize a detector whose neck and head start from
    their own initialization; weights reduced to the neck and the head initialize those under
    another backbone, whatever the names of its tensors.

    Args:
        state_dict: State dict of the detector.
        parts: Parts to keep, out of ``backbone``, ``neck`` and ``head``.

    Returns:
        The tensors of the parts, of the weights and of the averaged copy, and the update
        count of the average when the state dict holds one.

    Raises:
        ValueError: If no part or an unknown part is named.
    """
    unknown = sorted(set(parts) - set(NETWORK_PARTS))
    if unknown or not parts:
        raise ValueError(f"The parts must be some of {list(NETWORK_PARTS)}, got {list(parts)}.")
    prefixes = tuple(
        prefix + part + "." for prefix in (NETWORK_PREFIX, EMA_PREFIX) for part in parts
    )
    return {
        key: value
        for key, value in state_dict.items()
        if key.startswith(prefixes) or key == EMA_NUM_UPDATES_KEY
    }


def convert_reference_checkpoint(
    checkpoint: Mapping[str, object], with_ema: bool = True, head_level_offset: int = 0
) -> dict[str, Tensor]:
    """Convert a checkpoint of the reference implementation.

    Args:
        checkpoint: Loaded checkpoint, holding the weights under ``model`` or being the state
          dict itself.
        with_ema: Whether to also fill the averaged copy of the network.
        head_level_offset: Number of finer levels the detector has in front of the levels of
          the checkpoint, 1 for a detector with a level at stride 4, see
          :func:`shift_head_levels`.

    Returns:
        State dict of the detector.
    """
    reference_state = checkpoint.get("model", checkpoint)
    network_state = {
        shift_head_levels(rename_reference_key(key), head_level_offset): value
        for key, value in reference_state.items()
    }
    return to_detector_state_dict(network_state, with_ema=with_ema)


def _module_path(node_name: str) -> str:
    """Turn the name the exporter gave a convolution node into the path of its module.

    The exporter names a node after the scopes it sits in, repeating the name of a container
    before the name of its child: ``/backbone/backbone/dark2/dark2.0/conv/Conv`` is the
    module ``backbone.backbone.dark2.0.conv``.

    Args:
        node_name: Name of the ONNX node.

    Returns:
        Dotted path of the convolution module.
    """
    scopes = node_name.strip("/").split("/")[:-1]
    kept = [
        scope
        for index, scope in enumerate(scopes)
        if index + 1 == len(scopes) or not scopes[index + 1].startswith(scope + ".")
    ]
    return ".".join(kept)


def convert_folded_onnx(onnx_path: str | Path, with_ema: bool = True) -> dict[str, Tensor]:
    """Convert an ONNX file exported from the reference implementation.

    The exporter folds every batch normalization into the convolution before it, which leaves
    a convolution with a bias. The weight goes to the convolution and the bias to a batch
    normalization set to the identity (unit scale, zero mean, a variance of ``1 - eps``), so
    the detector computes what the graph computes. The converted normalization statistics
    are not those of the training run, so the weights suit evaluation and export, and a
    fine-tuning run re-estimates the statistics from this starting point.

    Args:
        onnx_path: Path of the ONNX file.
        with_ema: Whether to also fill the averaged copy of the network.

    Returns:
        State dict of the detector.

    Raises:
        ValueError: If a weight or a bias of a convolution of the graph is no initializer.
    """
    model = onnx.load(str(onnx_path))
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    # The exporter stores equal constants once and hands the others out through Identity nodes.
    aliases = {
        node.output[0]: node.input[0] for node in model.graph.node if node.op_type == "Identity"
    }

    def initializer(name: str, node_name: str) -> Tensor:
        """Read the initializer behind a convolution input, following Identity nodes."""
        while name in aliases:
            name = aliases[name]
        if name not in initializers:
            raise ValueError(f"Convolution {node_name!r} reads {name!r}, which is no initializer.")
        return torch.from_numpy(np.array(initializers[name]))

    network_state: dict[str, Tensor] = {}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        weight = initializer(node.input[1], node.name)
        bias = (
            initializer(node.input[2], node.name)
            if len(node.input) > 2
            else torch.zeros(weight.shape[0])
        )
        path = rename_reference_key(_module_path(node.name))
        network_state[f"{path}.weight"] = weight
        if path.endswith(".conv"):
            # A convolution of a block: the bias is the folded batch normalization.
            norm_path = path.removesuffix(".conv") + ".bn"
            network_state[f"{norm_path}.weight"] = torch.ones_like(bias)
            network_state[f"{norm_path}.bias"] = bias
            network_state[f"{norm_path}.running_mean"] = torch.zeros_like(bias)
            network_state[f"{norm_path}.running_var"] = torch.full_like(bias, 1.0 - BN_EPS)
            network_state[f"{norm_path}.num_batches_tracked"] = torch.zeros((), dtype=torch.long)
        else:
            network_state[f"{path}.bias"] = bias
    return to_detector_state_dict(network_state, with_ema=with_ema)
