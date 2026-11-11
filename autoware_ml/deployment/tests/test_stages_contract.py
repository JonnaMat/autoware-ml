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

"""Unit tests for the stage declaration in ``stages.py``."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from autoware_ml.deployment.stages import (
    GraphStage,
    RunState,
    TorchStage,
    model_output_names,
    run_stages_in_torch,
)


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _SplitHead(nn.Module):
    def forward(self, x):
        return x + 1, x - 1


class _Subtract(nn.Module):
    def forward(self, a, b):
        return a - b


def _toy_stages():
    return (
        TorchStage(
            "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
        ),
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        TorchStage(
            "glue",
            run=lambda state: {"z": state["y"].to(state.device) + 0.5},
            inputs=("y",),
            outputs=("z",),
        ),
        GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "minus")),
    )


class TestStageDeclaration(unittest.TestCase):
    """Unit tests for the stage declaration and its static dataflow check."""

    def test_a_chain_outputs_its_last_graph(self) -> None:
        """Test that a chain's model outputs are its last graph's outputs."""
        self.assertEqual(model_output_names(_toy_stages()), ("plus", "minus"))

    def test_two_heads_over_one_encoder_both_output(self) -> None:
        """Test that every head over one encoder outputs, with the encoder's unread output."""
        stages = (
            TorchStage(
                "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
            ),
            GraphStage("encoder", module=_SplitHead(), inputs=("x",), outputs=("y", "aux")),
            GraphStage("seg", module=_SplitHead(), inputs=("y",), outputs=("plus", "minus")),
            GraphStage("det", module=_Double(), inputs=("y",), outputs=("twice",)),
        )
        # ``y`` is consumed by the heads; ``aux`` is not, so it stays a model output.
        self.assertEqual(model_output_names(stages), ("aux", "plus", "minus", "twice"))

    def test_a_glue_output_the_model_hands_over_is_not_a_model_output(self) -> None:
        """Test that a graph output read by later glue is not a model output."""
        post = TorchStage(
            "post", run=lambda state: {"aux": state["plus"]}, inputs=("plus",), outputs=("aux",)
        )
        self.assertEqual(model_output_names((*_toy_stages(), post)), ("minus",))

    def test_graph_stage_names_each_onnx_tensor_once(self) -> None:
        """Test that a graph stage listing a tensor name twice is rejected."""
        with self.assertRaisesRegex(ValueError, r"\['x'\] more than once in its inputs"):
            GraphStage("sub", module=_Subtract(), inputs=("x", "x"), outputs=("y",))
        with self.assertRaisesRegex(ValueError, r"\['y'\] more than once in its outputs"):
            GraphStage("head", module=_SplitHead(), inputs=("x",), outputs=("y", "y"))

    def test_state_names_what_it_holds_on_a_missing_tensor(self) -> None:
        """Test that a missing tensor's error lists the tensors the run state holds."""
        state = RunState(batch=None, device=torch.device("cpu"))
        state.tensors["mid"] = torch.ones(1)
        with self.assertRaisesRegex(KeyError, r"available: \['mid'\]"):
            state["typo"]
        with self.assertRaisesRegex(KeyError, r"available: \['mid'\]"):
            state.narrowed(("typo",))


class TestRunStagesInTorch(unittest.TestCase):
    """Unit tests for running a stage graph in PyTorch and its runtime checks."""

    def test_runs_all_stages_in_order(self) -> None:
        """Test that the stages run in order and fill the run state."""
        state = run_stages_in_torch(
            _toy_stages(), {"x": torch.tensor([1, 2, 3])}, torch.device("cpu")
        )
        # prep: x=[1,2,3]; encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        self.assertTrue(torch.equal(state["plus"], torch.tensor([3.5, 5.5, 7.5])))
        self.assertTrue(torch.equal(state["minus"], torch.tensor([1.5, 3.5, 5.5])))
        self.assertEqual(set(state.tensors), {"x", "y", "z", "plus", "minus"})

    def test_multi_input_graph_stage_positional_order(self) -> None:
        """Test that a graph stage's arguments follow its declared ``inputs`` order."""
        # Arguments follow ``inputs``, not the parameter names or the glue's output order.
        stages = (
            TorchStage(
                "prep",
                run=lambda state: {"b": torch.tensor([1.0]), "a": torch.tensor([10.0])},
                inputs=(),
                outputs=("b", "a"),
            ),
            GraphStage("ab", module=_Subtract(), inputs=("a", "b"), outputs=("a_minus_b",)),
            GraphStage("ba", module=_Subtract(), inputs=("b", "a"), outputs=("b_minus_a",)),
        )
        state = run_stages_in_torch(stages, None, torch.device("cpu"))
        self.assertTrue(torch.equal(state["a_minus_b"], torch.tensor([9.0])))
        self.assertTrue(torch.equal(state["b_minus_a"], torch.tensor([-9.0])))

    def test_glue_returning_other_names_than_declared_is_rejected(self) -> None:
        """Test that glue returning names other than its declared outputs is rejected."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"zz": state["y"]}, inputs=("y",), outputs=("z",)
        )
        with self.assertRaisesRegex(ValueError, r"'glue' returned \['zz'\] but declares"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_runs_on_its_declared_inputs_only(self) -> None:
        """Test that glue reading a name it did not declare fails."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"z": state["x"]}, inputs=("y",), outputs=("z",)
        )
        with self.assertRaisesRegex(KeyError, "did not declare as an input"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_stage_output_arity_is_checked(self) -> None:
        """Test that a graph stage returning fewer tensors than it declares is rejected."""
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_Double(), inputs=("z",), outputs=("plus", "minus"))
        with self.assertRaisesRegex(ValueError, "zip"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_inputs_are_the_tensors_the_graph_ran_on(self) -> None:
        """Test that the recorded graph inputs are the run-state tensors themselves."""
        state = run_stages_in_torch(_toy_stages(), {"x": torch.tensor([1.0])}, torch.device("cpu"))
        (head_input,) = state.graph_inputs["head"]
        self.assertIs(head_input, state["z"])
