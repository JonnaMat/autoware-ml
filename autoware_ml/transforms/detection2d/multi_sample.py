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

"""Shared machinery of the 2D detection augmentations: sample mixing and the strong switch."""

from __future__ import annotations

from collections.abc import Callable
import multiprocessing

import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.base import BaseTransform


class StrongAugmentationSwitch:
    """Process shared flag that turns the strong augmentations of a pipeline on and off.

    YOLOX trains its last epochs without mosaic, random affine and mixup. The transforms run
    in the dataloader workers, which may live for the whole run, so the flag lives in shared
    memory: the training process flips it and every worker reads the new value with its next
    sample. A switch is found by name with :func:`strong_augmentation_switch`, so the
    transforms of a pipeline and the callback that schedules them agree on one flag without
    being handed the same object.
    """

    def __init__(self) -> None:
        """Create the switch, strong augmentation on."""
        self._enabled = multiprocessing.Value("b", True)

    @property
    def enabled(self) -> bool:
        """Whether the strong augmentations run."""
        return bool(self._enabled.value)

    def set(self, enabled: bool) -> None:
        """Turn the strong augmentations on or off, in every process sharing the switch.

        Args:
            enabled: Whether the strong augmentations run from now on.
        """
        self._enabled.value = enabled


_SWITCHES: dict[str, StrongAugmentationSwitch] = {}


def strong_augmentation_switch(name: str) -> StrongAugmentationSwitch:
    """Get the switch of a name, creating it on first use.

    Call it in the training process before the dataloader workers start, as instantiating the
    pipeline and the callbacks from the config does: a worker then inherits the switches its
    transforms hold.

    Args:
        name: Name of the switch.

    Returns:
        The switch every caller of the process gets for this name.
    """
    if name not in _SWITCHES:
        _SWITCHES[name] = StrongAugmentationSwitch()
    return _SWITCHES[name]


class SwitchableTransform(BaseTransform):
    """Transform that a strong augmentation switch can turn off."""

    def __init__(self, probability: float | None = None, switch_name: str | None = None) -> None:
        """Initialize the transform.

        Args:
            probability: Probability of applying the transform, None to always apply it.
            switch_name: Name of the strong augmentation switch that turns the transform off,
              None for a transform that always runs.
        """
        super().__init__(probability=probability)
        self._switch = strong_augmentation_switch(switch_name) if switch_name is not None else None

    def _should_apply(self, random_state: np.random.RandomState | None = None) -> bool:
        """Skip the transform while its switch is off, else follow the probability.

        Args:
            random_state: Random state to draw from, None for the global one of NumPy.

        Returns:
            True if the transform should be applied.
        """
        if self._switch is not None and not self._switch.enabled:
            return False
        return super()._should_apply(random_state)


class MultiSampleTransform(SwitchableTransform):
    """Transform that mixes further samples of the dataset into the one it augments.

    The dataset binds a sample source when it is bound to its records: a callable returning
    another sample of the dataset, drawn at random and loaded by the transforms of the
    pipeline that come before the first mixing transform.
    """

    def __init__(self, probability: float | None = None, switch_name: str | None = None) -> None:
        """Initialize the transform without a sample source.

        Args:
            probability: Probability of applying the transform, None to always apply it.
            switch_name: Name of the strong augmentation switch that turns the transform off.
        """
        super().__init__(probability=probability, switch_name=switch_name)
        self._sample_source: Callable[[], ModelGTSample] | None = None

    def bind_sample_source(self, sample_source: Callable[[], ModelGTSample]) -> None:
        """Set the callable the transform draws further samples from.

        Args:
            sample_source: Returns another loaded sample of the dataset.
        """
        self._sample_source = sample_source

    def draw_sample(self) -> ModelGTSample:
        """Draw another loaded sample of the dataset.

        Raises:
            RuntimeError: If no dataset bound a sample source.
        """
        if self._sample_source is None:
            raise RuntimeError(
                f"{type(self).__name__} mixes samples of the dataset, but no dataset bound a "
                "sample source to it. Use it in the pipeline of a dataset that supports it."
            )
        return self._sample_source()
