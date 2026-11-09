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

"""The check every stage-graph model's tests run: the stages reproduce ``forward()``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple
import unittest

import torch

from autoware_ml.deployment.stages import RunState, Stage, model_output_names, run_stages_in_torch


class StageGraphRun(NamedTuple):
    """What :func:`assert_stage_graph_reproduces_forward` ran, for checks beyond the outputs."""

    stages: tuple[Stage, ...]
    forward_outputs: Any
    state: RunState


def assert_stage_graph_reproduces_forward(
    model: Any,
    batch: Any,
    device: torch.device,
    forward_tensors: Callable[[Any], Mapping[str, torch.Tensor]],
    *,
    rtol: float,
    atol: float,
) -> StageGraphRun:
    """Run ``forward()`` and the stage graph on one batch and compare every model output.

    ``forward()`` and ``build_stages()`` compose the same modules twice, once for training
    and once for deployment, and nothing but this check keeps the two compositions equal.
    Both runs share the weights, so only kernel-selection noise remains; a wrong dataflow is
    off by O(1).

    Args:
        model: Model in eval mode declaring ``build_stages()``.
        batch: Model inputs both runs start from.
        device: Device the stages run on.
        forward_tensors: Maps the outputs ``forward()`` returns to the model output names of
            the stage graph (:func:`~autoware_ml.deployment.stages.model_output_names`).
        rtol: Relative tolerance of :func:`torch.testing.assert_close`.
        atol: Absolute tolerance of :func:`torch.testing.assert_close`.

    Returns:
        The stages, what ``forward()`` returned and the filled run state.

    Raises:
        AssertionError: When the model declares no stages, ``forward_tensors`` does not name
            exactly the model outputs, or a model output differs from ``forward()``'s.
    """
    case = unittest.TestCase()
    stages = model.build_stages()
    case.assertIsNotNone(stages, f"{type(model).__name__} declares no stage graph.")
    names = model_output_names(stages)
    with torch.no_grad():
        forward_outputs = model(batch)
    reference = forward_tensors(forward_outputs)
    case.assertEqual(
        set(reference),
        set(names),
        "forward_tensors must name exactly the model outputs of the stage graph.",
    )
    state = run_stages_in_torch(stages, batch, device)
    for name in names:
        torch.testing.assert_close(
            state[name],
            reference[name],
            rtol=rtol,
            atol=atol,
            msg=lambda detail, name=name: f"Model output {name!r} differs from forward(): {detail}",
        )
    return StageGraphRun(tuple(stages), forward_outputs, state)
