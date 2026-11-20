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

"""Unit tests for the exponential moving average of module weights."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from autoware_ml.models.common.ema import ModelEMA


class TestModelEMA(unittest.TestCase):
    """Unit tests for ModelEMA."""

    def setUp(self) -> None:
        """Set up a small module with a normalization layer and its average."""
        torch.manual_seed(0)
        self.module = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
        self.ema = ModelEMA(self.module, decay=0.9, warmup_updates=10)

    def test_average_starts_as_a_frozen_copy(self) -> None:
        """Test that the average starts equal to the module, frozen and in eval mode."""
        for averaged, source in zip(
            self.ema.module.parameters(), self.module.parameters(), strict=True
        ):
            torch.testing.assert_close(averaged, source)
            self.assertFalse(averaged.requires_grad)
            self.assertIsNot(averaged, source)
        self.assertFalse(self.ema.train().module.training)

    def test_update_blends_by_the_ramped_decay(self) -> None:
        """Test one update against the formula ``decay * (1 - exp(-updates / warmup))``.

        Check: floating point states move by ``1 - d`` towards the source, and the integer
        counter of the normalization layer is copied.
        """
        before = self.ema.module[0].weight.clone()
        with torch.no_grad():
            self.module[0].weight.add_(1.0)
            self.module[1].num_batches_tracked.fill_(7)

        self.ema.update(self.module)

        decay = 0.9 * (1.0 - math.exp(-1 / 10))
        torch.testing.assert_close(
            self.ema.module[0].weight, decay * before + (1 - decay) * self.module[0].weight
        )
        self.assertEqual(int(self.ema.module[1].num_batches_tracked), 7)
        self.assertEqual(int(self.ema.num_updates), 1)

    def test_decay_approaches_its_limit(self) -> None:
        """Test that the decay ramps up towards the configured decay with the updates."""
        first = self.ema.current_decay()
        self.ema.num_updates.fill_(1000)

        self.assertLess(first, 0.1)
        self.assertAlmostEqual(self.ema.current_decay(), 0.9, places=6)

    def test_copy_from_overwrites_the_average(self) -> None:
        """Test that ``copy_from`` sets the average to the state of the module."""
        with torch.no_grad():
            self.module[0].weight.add_(3.0)

        self.ema.copy_from(self.module)

        torch.testing.assert_close(self.ema.module[0].weight, self.module[0].weight)

    def test_invalid_settings_are_refused(self) -> None:
        """Test that a decay outside [0, 1) and a non-positive ramp are refused."""
        with self.assertRaises(ValueError):
            ModelEMA(self.module, decay=1.0)
        with self.assertRaises(ValueError):
            ModelEMA(self.module, warmup_updates=0)
