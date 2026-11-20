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

"""Warmup, cosine and constant tail learning-rate scheduler of YOLOX."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class YOLOXWarmCosineLR(LRScheduler):
    """Quadratic warmup, cosine decay, then a constant minimum for the last epochs.

    Stepped once per optimization step, as the ``yoloxwarmcos`` schedule of the YOLOX
    reference implementation (``yolox/utils/lr_scheduler.py`` of
    https://github.com/Megvii-BaseDetection/YOLOX, Apache-2.0):

    * during the first ``warmup_epochs`` the rate rises from ``warmup_start_factor * lr`` to
      ``lr`` with the square of the progress;
    * it then follows half a cosine down to ``min_lr_factor * lr``, reached
      ``no_aug_epochs`` before the end;
    * the last ``no_aug_epochs``, trained without strong augmentation, keep that minimum.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_epochs: int,
        warmup_epochs: float = 5.0,
        no_aug_epochs: float = 15.0,
        warmup_start_factor: float = 0.0,
        min_lr_factor: float = 0.05,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            total_steps: Number of optimization steps of the run.
            max_epochs: Number of epochs of the run.
            warmup_epochs: Length of the warmup in epochs.
            no_aug_epochs: Length of the constant tail in epochs.
            warmup_start_factor: Share of the base rate the warmup starts from.
            min_lr_factor: Share of the base rate the cosine ends at.
            last_epoch: Index of the last step, -1 for a fresh run.

        Raises:
            ValueError: If the epochs are not positive, the run has fewer steps than epochs,
              or the warmup and the tail leave no epoch for the cosine.
        """
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}.")
        if total_steps < max_epochs:
            raise ValueError(
                f"total_steps ({total_steps}) must cover at least one step per epoch "
                f"({max_epochs} epochs)."
            )
        if warmup_epochs < 0 or no_aug_epochs < 0 or warmup_epochs + no_aug_epochs >= max_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) plus no_aug_epochs ({no_aug_epochs}) must be "
                f"non-negative and leave room for the cosine within {max_epochs} epochs."
            )
        steps_per_epoch = total_steps / max_epochs
        self.total_steps = total_steps
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.no_aug_steps = no_aug_epochs * steps_per_epoch
        self.warmup_start_factor = warmup_start_factor
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def factor_at(self, step: int) -> float:
        """Share of the base rate at an optimization step.

        Args:
            step: Index of the optimization step, from 0.

        Returns:
            The multiplier of the base learning rate.
        """
        if step < self.warmup_steps:
            progress = step / self.warmup_steps
            return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress**2
        cosine_end = self.total_steps - self.no_aug_steps
        if step >= cosine_end:
            return self.min_lr_factor
        progress = (step - self.warmup_steps) / (cosine_end - self.warmup_steps)
        return self.min_lr_factor + 0.5 * (1.0 - self.min_lr_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    def get_lr(self) -> list[float]:
        """Compute the learning rate for the current optimization step."""
        factor = self.factor_at(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]
