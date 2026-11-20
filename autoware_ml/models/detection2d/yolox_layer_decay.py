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

"""YOLOX detector whose pretrained backbone trains at its own learning rates.

A transformer backbone is usually fine-tuned at a lower learning rate than the new layers on
top of it, lower still for its early blocks (the layer-wise decay of BEiT and ViTDet). The
detector here differs from :class:`YOLOXDetector` in its optimizer groups only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn
from torch.optim import Optimizer

from autoware_ml.models.detection2d.yolox import YOLOXDetector
from autoware_ml.utils.optimizer import materialize_partial_kwargs


class YOLOXLayerDecayDetector(YOLOXDetector):
    """YOLOX detector with a learning rate per layer of the backbone.

    A backbone may tell the layer of its pretrained parameters with ``lr_layer_ids()``, which
    returns the layer of every such parameter by name and the highest layer, and name
    parameters that are never decayed with ``no_weight_decay()``. The learning rate of layer
    ``i`` of ``n`` is the rate of the optimizer times
    ``backbone_lr_factor * layer_decay ** (n - i)``. Parameters of the backbone without a
    layer, the neck and the head train at the rate of the optimizer. A backbone without
    ``lr_layer_ids()`` is one layer: all of it trains at ``backbone_lr_factor`` times the rate.

    The groups ``decay`` and ``no_decay`` keep their meaning, and every backbone group takes
    the overrides of the one it would otherwise belong to.
    """

    def __init__(
        self,
        *args: Any,
        backbone_lr_factor: float = 1.0,
        layer_decay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the detector.

        Args:
            *args: Arguments of :class:`YOLOXDetector`.
            backbone_lr_factor: Learning rate of the last layer of the backbone, as a share of
              the rate of the optimizer.
            layer_decay: Factor between the learning rates of two consecutive layers of the
              backbone, 1.0 for one rate.
            **kwargs: Arguments of :class:`YOLOXDetector`.

        Raises:
            ValueError: If a factor is not in (0, 1].
        """
        super().__init__(*args, **kwargs)
        for name, value in (
            ("backbone_lr_factor", backbone_lr_factor),
            ("layer_decay", layer_decay),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")
        self.backbone_lr_factor = float(backbone_lr_factor)
        self.layer_decay = float(layer_decay)
        self._configured_group_overrides = dict(self.optimizer_group_overrides or {})

    def _backbone_layers(self) -> tuple[dict[str, int], int, set[str]]:
        """Read the layer of every backbone parameter and the names that are never decayed."""
        backbone = self.network.backbone
        if hasattr(backbone, "lr_layer_ids"):
            layer_ids, num_layers = backbone.lr_layer_ids()
        else:
            layer_ids = {name: 0 for name, _ in backbone.named_parameters()}
            num_layers = 0
        no_decay_names = (
            set(backbone.no_weight_decay()) if hasattr(backbone, "no_weight_decay") else set()
        )
        return layer_ids, num_layers, no_decay_names

    def layer_lr_scale(self, layer: int, num_layers: int) -> float:
        """Share of the learning rate of the optimizer a backbone layer trains at.

        Args:
            layer: Index of the layer, from 0.
            num_layers: Index of the last layer.

        Returns:
            The multiplier of the learning rate.
        """
        return self.backbone_lr_factor * self.layer_decay ** (num_layers - layer)

    def build_optimizer_groups(self) -> Mapping[str, Sequence[nn.Parameter]]:
        """Split the trained parameters by backbone layer and by whether weight decay applies.

        Returns:
            The groups ``decay`` and ``no_decay`` of the parameters trained at the rate of the
            optimizer, and ``backbone_layer_<i>_decay`` and ``backbone_layer_<i>_no_decay``
            for every backbone layer that holds such parameters.
        """
        layer_ids, _, no_decay_names = self._backbone_layers()
        groups: dict[str, list[nn.Parameter]] = {"decay": [], "no_decay": []}
        for name, parameter in self.network.named_parameters():
            if not parameter.requires_grad:
                continue
            backbone_name = name.removeprefix("backbone.") if name.startswith("backbone.") else None
            decayed = parameter.ndim >= 2 and backbone_name not in no_decay_names
            group = "decay" if decayed else "no_decay"
            if backbone_name in layer_ids:
                group = f"backbone_layer_{layer_ids[backbone_name]}_{group}"
            groups.setdefault(group, []).append(parameter)
        return groups

    def build_group_overrides(self) -> dict[str, dict[str, Any]]:
        """Derive the optimizer overrides of every group from the configured ones.

        Returns:
            The configured overrides, and for every backbone group those of ``decay`` or
            ``no_decay`` with the learning rate of its layer.

        Raises:
            ValueError: If the optimizer is not configured with a learning rate.
        """
        keywords = getattr(self.optimizer_partial, "keywords", None) or {}
        if "lr" not in keywords:
            raise ValueError("The layer-wise learning rates need an optimizer configured with lr.")
        base_lr = float(keywords["lr"])
        configured = materialize_partial_kwargs(self._configured_group_overrides)
        _, num_layers, _ = self._backbone_layers()
        overrides = {name: dict(values) for name, values in configured.items()}
        for group in self.build_optimizer_groups():
            if not group.startswith("backbone_layer_"):
                continue
            _, _, layer, kind = group.split("_", 3)
            overrides[group] = {
                **configured.get(kind, {}),
                "lr": base_lr * self.layer_lr_scale(int(layer), num_layers),
                **configured.get(group, {}),
            }
        return overrides

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        """Configure the optimizer with the learning rate of every backbone layer."""
        self.optimizer_group_overrides = self.build_group_overrides()
        return super().configure_optimizers()
