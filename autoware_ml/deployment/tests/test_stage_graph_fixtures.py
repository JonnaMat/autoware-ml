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

"""Tests of the forward-parity check shared by the stage-graph models' tests."""

from collections.abc import Sequence
import unittest

import torch
from torch import nn

from autoware_ml.deployment.stages import GraphStage, RunState, Stage, TorchStage
from autoware_ml.deployment.tests.stage_graph_fixtures import assert_stage_graph_reproduces_forward


class _TwoStageModel(nn.Module):
    """``forward()`` = scale then linear; the stage graph declares the same two steps."""

    def __init__(self, graph_scale: float = 2.0) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.graph_scale = graph_scale

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.linear(batch * 2.0)

    def build_stages(self) -> Sequence[Stage]:
        def scale(state: RunState) -> dict[str, torch.Tensor]:
            return {"scaled": state.batch * self.graph_scale}

        return (
            TorchStage("scale", run=scale, inputs=(), outputs=("scaled",)),
            GraphStage("linear", module=self.linear, inputs=("scaled",), outputs=("logits",)),
        )


class TestAssertStageGraphReproducesForward(unittest.TestCase):
    """Unit tests of ``assert_stage_graph_reproduces_forward``."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.batch = torch.randn(4, 3)

    def test_returns_the_run_state_when_the_stages_reproduce_forward(self) -> None:
        """Test that equal compositions pass and the run, with its state, comes back."""
        model = _TwoStageModel().eval()
        run = assert_stage_graph_reproduces_forward(
            model,
            self.batch,
            self.device,
            lambda outputs: {"logits": outputs},
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual([stage.name for stage in run.stages], ["scale", "linear"])
        self.assertEqual(tuple(run.forward_outputs.shape), (4, 2))
        self.assertEqual(tuple(run.state["scaled"].shape), (4, 3))

    def test_fails_when_the_glue_differs_from_forward(self) -> None:
        """Test that a stage graph whose glue drifts from forward() fails the check."""
        model = _TwoStageModel(graph_scale=3.0).eval()
        with self.assertRaisesRegex(AssertionError, "'logits' differs from forward"):
            assert_stage_graph_reproduces_forward(
                model,
                self.batch,
                self.device,
                lambda outputs: {"logits": outputs},
                rtol=1e-6,
                atol=1e-6,
            )

    def test_fails_when_the_reference_does_not_name_the_model_outputs(self) -> None:
        """Test that a reference mapping with other names than the model outputs is refused."""
        model = _TwoStageModel().eval()
        with self.assertRaisesRegex(AssertionError, "exactly the model outputs"):
            assert_stage_graph_reproduces_forward(
                model,
                self.batch,
                self.device,
                lambda outputs: {"scores": outputs},
                rtol=1e-6,
                atol=1e-6,
            )

    def test_fails_for_a_model_without_stages(self) -> None:
        """Test that a model returning no stages fails instead of passing vacuously."""

        class _NoStages(nn.Module):
            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                return batch

            def build_stages(self) -> None:
                return None

        with self.assertRaisesRegex(AssertionError, "declares no stage graph"):
            assert_stage_graph_reproduces_forward(
                _NoStages().eval(),
                self.batch,
                self.device,
                lambda outputs: {"x": outputs},
                rtol=1e-6,
                atol=1e-6,
            )
