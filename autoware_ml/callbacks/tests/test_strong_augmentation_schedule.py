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

"""Unit tests for the StrongAugmentationSchedule callback."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from lightning.pytorch import LightningModule, Trainer

from autoware_ml.callbacks.strong_augmentation_schedule import StrongAugmentationSchedule
from autoware_ml.transforms.detection2d.multi_sample import strong_augmentation_switch

SWITCH_NAME = "test_schedule_switch"


class TestStrongAugmentationSchedule(unittest.TestCase):
    """Unit tests for the schedule that ends the strong augmentation."""

    def setUp(self) -> None:
        """Build a callback ending the strong augmentation 3 epochs before the end of 10."""
        self.switch = strong_augmentation_switch(SWITCH_NAME)
        self.switch.set(True)
        self.addCleanup(self.switch.set, True)
        self.callback = StrongAugmentationSchedule(num_last_epochs=3, switch_name=SWITCH_NAME)
        self.module = MagicMock(spec=LightningModule)
        self.module.set_strong_augmentation = MagicMock()

    def start_epoch(self, epoch: int, max_epochs: int = 10) -> None:
        """Run the epoch start hook for an epoch of a run."""
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = epoch
        trainer.max_epochs = max_epochs
        self.callback.on_train_epoch_start(trainer, self.module)

    def test_switch_turns_off_for_the_last_epochs(self) -> None:
        """Test that the switch and the model follow the epoch: on up to epoch 6, off from 7."""
        self.start_epoch(6)
        self.assertTrue(self.switch.enabled)
        self.module.set_strong_augmentation.assert_called_with(True)

        self.start_epoch(7)
        self.assertFalse(self.switch.enabled)
        self.module.set_strong_augmentation.assert_called_with(False)

    def test_resumed_run_lands_in_the_right_phase(self) -> None:
        """Test that a run starting inside the last epochs turns the augmentation off at once."""
        self.start_epoch(9)

        self.assertFalse(self.switch.enabled)

    def test_zero_last_epochs_keep_the_augmentation_on(self) -> None:
        """Test that no last epochs keep the strong augmentation on until the end."""
        callback = StrongAugmentationSchedule(num_last_epochs=0, switch_name=SWITCH_NAME)

        self.assertTrue(callback.strong_augmentation_at(epoch=9, max_epochs=10))

    def test_model_without_the_hook_is_left_alone(self) -> None:
        """Test that a model without ``set_strong_augmentation`` only gets the switch flipped."""
        self.module = MagicMock(spec=LightningModule)

        self.start_epoch(8)

        self.assertFalse(self.switch.enabled)

    def test_negative_epoch_count_is_refused(self) -> None:
        """Test that a negative number of last epochs is refused."""
        with self.assertRaises(ValueError):
            StrongAugmentationSchedule(num_last_epochs=-1)
