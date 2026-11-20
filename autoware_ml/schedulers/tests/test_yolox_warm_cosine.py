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

"""Unit tests for the YOLOX warmup, cosine and constant tail learning-rate scheduler."""

import math

import pytest
import torch

from autoware_ml.utils.schedulers.yolox_warm_cosine import YOLOXWarmCosineLR

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`"
)


class TestYOLOXWarmCosineLR:
    """Tests for the YOLOXWarmCosineLR scheduler."""

    def build(self, optimizer: torch.optim.Optimizer) -> YOLOXWarmCosineLR:
        """Build a schedule of 10 epochs of 10 steps: 2 of warmup, 3 without augmentation."""
        return YOLOXWarmCosineLR(
            optimizer, total_steps=100, max_epochs=10, warmup_epochs=2, no_aug_epochs=3
        )

    def test_warmup_is_quadratic_from_zero(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test that the rate starts at zero and rises with the square of the warmup progress."""
        scheduler = self.build(optimizer)

        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(20):
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])

        assert rates[0] == 0.0
        assert rates[10] == pytest.approx(base_lr * 0.25)
        assert rates[20] == pytest.approx(base_lr)

    def test_cosine_decays_to_the_minimum_before_the_tail(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test the cosine between the warmup and the tail and the constant rate of the tail.

        Check: halfway through the cosine (step 45 of the steps 20 to 70) the rate is the
        mean of the base and the minimum rate, and every step of the last three epochs runs
        at the minimum of 5 % of the base rate.
        """
        scheduler = self.build(optimizer)

        assert scheduler.factor_at(45) == pytest.approx(0.05 + 0.5 * 0.95)
        assert scheduler.factor_at(69) > 0.05
        for step in (70, 85, 99, 100):
            assert scheduler.factor_at(step) == pytest.approx(0.05)
        expected = 0.05 + 0.5 * 0.95 * (1 + math.cos(math.pi * 10 / 50))
        for _ in range(30):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * expected)

    def test_schedules_without_room_for_the_cosine_are_refused(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """Test that a warmup and tail filling the whole run, and too few steps, are refused."""
        with pytest.raises(ValueError):
            YOLOXWarmCosineLR(
                optimizer, total_steps=100, max_epochs=10, warmup_epochs=5, no_aug_epochs=5
            )
        with pytest.raises(ValueError):
            YOLOXWarmCosineLR(optimizer, total_steps=5, max_epochs=10)
