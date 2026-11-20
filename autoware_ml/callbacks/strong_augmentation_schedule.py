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

"""Schedule that ends the strong augmentation of a training pipeline before the last epochs."""

from __future__ import annotations

import logging

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from autoware_ml.transforms.detection2d.multi_sample import strong_augmentation_switch

logger = logging.getLogger(__name__)


class StrongAugmentationSchedule(Callback):
    """Turn the strong augmentations off for the last epochs of a run.

    YOLOX trains its last epochs on plainly resized images, which closes the gap between the
    mosaic training samples and the images the detector is evaluated on. From the first of
    the last ``num_last_epochs`` epochs on, the callback flips the strong augmentation switch
    of the training pipeline and tells the model through its ``set_strong_augmentation``
    hook, when it has one: the YOLOX detector then adds its L1 box loss. The epoch is checked
    at every epoch start, so a resumed run lands in the right phase.
    """

    def __init__(self, num_last_epochs: int, switch_name: str = "strong_augmentation") -> None:
        """Initialize the callback.

        Args:
            num_last_epochs: Number of epochs at the end of the run without strong
              augmentation, 0 to keep it on throughout.
            switch_name: Name of the strong augmentation switch the transforms of the training
              pipeline read.

        Raises:
            ValueError: If the number of epochs is negative.
        """
        super().__init__()
        if num_last_epochs < 0:
            raise ValueError(f"num_last_epochs must not be negative, got {num_last_epochs}.")
        self.num_last_epochs = num_last_epochs
        self.switch = strong_augmentation_switch(switch_name)

    def strong_augmentation_at(self, epoch: int, max_epochs: int) -> bool:
        """Whether an epoch trains with strong augmentation.

        Args:
            epoch: Index of the epoch, from 0.
            max_epochs: Number of epochs of the run.

        Returns:
            False for the last ``num_last_epochs`` epochs.
        """
        return epoch < max_epochs - self.num_last_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set the switch and the model for the epoch that starts."""
        if trainer.max_epochs is None or trainer.max_epochs < 0:
            return
        enabled = self.strong_augmentation_at(trainer.current_epoch, trainer.max_epochs)
        if enabled != self.switch.enabled:
            logger.info(
                "Epoch %d: strong augmentation turned %s.",
                trainer.current_epoch,
                "on" if enabled else "off",
            )
        self.switch.set(enabled)
        set_strong_augmentation = getattr(pl_module, "set_strong_augmentation", None)
        if set_strong_augmentation is not None:
            set_strong_augmentation(enabled)
