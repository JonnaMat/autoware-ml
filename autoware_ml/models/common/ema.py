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

"""Exponential moving average of the weights of a module."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA(nn.Module):
    """Keep an exponential moving average of every floating point state of a module.

    The average is a copy of the module that never trains: its parameters need no gradient
    and it stays in eval mode. The decay ramps up from 0 with the number of updates,
    ``decay * (1 - exp(-updates / warmup_updates))``, so the average follows the first steps
    closely and smooths the later ones, as the YOLOX reference implementation does
    (``yolox/utils/ema.py`` of https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0).
    The average and the number of updates are part of the state dict, so a checkpoint carries
    them.
    """

    num_updates: torch.Tensor

    def __init__(
        self, module: nn.Module, decay: float = 0.9998, warmup_updates: int = 2000
    ) -> None:
        """Initialize the average with a copy of the module.

        Args:
            module: Module whose weights are averaged.
            decay: Decay of the average once the ramp is over.
            warmup_updates: Time constant of the ramp, in updates.

        Raises:
            ValueError: If the decay is outside [0, 1) or the time constant is not positive.
        """
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}.")
        if warmup_updates <= 0:
            raise ValueError(f"warmup_updates must be positive, got {warmup_updates}.")
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.module = copy.deepcopy(module)
        self.module.requires_grad_(False)
        self.module.eval()
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode: bool = True) -> ModelEMA:
        """Keep the averaged module in eval mode whatever the mode of its owner.

        Args:
            mode: Requested mode, ignored for the averaged module.

        Returns:
            The module itself.
        """
        super().train(mode)
        self.module.eval()
        return self

    def current_decay(self) -> float:
        """Decay the next update uses."""
        updates = int(self.num_updates.item()) + 1
        return self.decay * (1.0 - math.exp(-updates / self.warmup_updates))

    @torch.no_grad()
    def copy_from(self, module: nn.Module) -> None:
        """Overwrite the average with the state of a module.

        Args:
            module: Module of the same structure as the averaged one.
        """
        self.module.load_state_dict(module.state_dict())

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """Move the average one step towards the state of a module.

        Args:
            module: Module of the same structure as the averaged one.
        """
        decay = self.current_decay()
        self.num_updates += 1
        source_state = module.state_dict()
        for name, value in self.module.state_dict().items():
            source = source_state[name].detach()
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source.to(value.dtype), alpha=1.0 - decay)
            else:
                value.copy_(source)
