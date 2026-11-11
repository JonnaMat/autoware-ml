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

"""Stage graph: one declaration of how a model splits into deployable graphs.

Inference is an ordered list of stages over named tensors (:class:`RunState`): a
:class:`GraphStage` is one exported ONNX / TensorRT graph, a :class:`TorchStage` is glue
that runs in PyTorch on every backend. The export specs derive from one run of the list.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from autoware_ml.utils.deploy import ExportSpec


@dataclass
class RunState:
    """Named tensors of one stage-graph run.

    Attributes:
        batch: The preprocessed batch the run started from; only glue reads it.
        device: Device the graph stages run on.
        tensors: Tensors produced so far, by name.
        graph_inputs: The positional inputs each graph stage ran on, by stage name.
    """

    batch: Any
    device: torch.device
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    graph_inputs: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return a produced tensor; a miss means a stage read a name it did not declare."""
        if name not in self.tensors:
            raise KeyError(
                f"Run state has no tensor {name!r}; available: {sorted(self.tensors)}. "
                "A stage reads a name it did not declare as an input."
            )
        return self.tensors[name]

    def narrowed(self, names: Sequence[str]) -> RunState:
        """The view a glue stage runs on: the same batch and device, only ``names``."""
        return RunState(self.batch, self.device, {name: self[name] for name in names})


@dataclass(frozen=True)
class TorchStage:
    """Glue that is not exported and runs in PyTorch on every backend.

    Attributes:
        name: Unique stage name.
        run: ``fn(state) -> {name: tensor}`` returning exactly ``outputs``.
        inputs: Run-state names the stage reads; it runs on a view holding only these.
        outputs: Run-state names the stage writes.
    """

    name: str
    run: Callable[[RunState], Mapping[str, torch.Tensor]]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class GraphStage:
    """One exported ONNX / TensorRT graph.

    Attributes:
        name: Unique stage name, also the ``deploy.onnx.modules`` key and the artifact stem.
        module: The traced module, export-ready as declared; its positional arguments are the
            ``inputs`` tensors.
        inputs: Run-state names fed to the module, in order; they are the ONNX input names.
        outputs: ONNX output names, in the module's return order.
        onnx_dynamic_axes: Axes dynamic by construction, ``{tensor: {dim: name}}``.
    """

    name: str
    module: nn.Module
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a tensor name listed twice: the exporter would rename it (``x`` -> ``x.1``)."""
        for kind, names in (("inputs", self.inputs), ("outputs", self.outputs)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"GraphStage {self.name!r} lists {repeated} more than once in its {kind}; "
                    "each is one ONNX tensor name."
                )


Stage = TorchStage | GraphStage


def model_output_names(stages: Sequence[Stage]) -> tuple[str, ...]:
    """Return the graph outputs no later stage reads, in stage and declaration order.

    These are the model outputs: the last graph's outputs for a chain, every head's outputs
    for several heads over one encoder.
    """
    names: list[str] = []
    for index, stage in enumerate(stages):
        if isinstance(stage, GraphStage):
            consumed = {name for later in stages[index + 1 :] for name in later.inputs}
            names.extend(name for name in stage.outputs if name not in consumed)
    return tuple(names)


def run_stages_in_torch(stages: Sequence[Stage], batch: Any, device: torch.device) -> RunState:
    """Run every stage in PyTorch and return the filled run state.

    Raises:
        ValueError: When a glue stage returns other names than it declares, or a graph stage
            another number of tensors.
    """
    state = RunState(batch=batch, device=device)
    with torch.no_grad():
        for stage in stages:
            if isinstance(stage, TorchStage):
                produced = stage.run(state.narrowed(stage.inputs))
                if set(produced) != set(stage.outputs):
                    raise ValueError(
                        f"TorchStage {stage.name!r} returned {sorted(produced)} but declares "
                        f"outputs {list(stage.outputs)}."
                    )
                state.tensors.update(produced)
                continue
            args = tuple(state[name] for name in stage.inputs)
            state.graph_inputs[stage.name] = args
            outputs = stage.module(*args)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            state.tensors.update(zip(stage.outputs, outputs, strict=True))
    return state


def build_export_specs_from_stages(
    stages: Sequence[Stage], batch: Any, device: torch.device
) -> dict[str, ExportSpec]:
    """Run the stage graph once and return ``{stage name: ExportSpec}`` per graph stage.

    The graph modules are switched to eval first: ``torch.onnx.export`` restores a wrapper's
    mode recursively, so a wrapper left in training mode would switch the submodules it shares
    with the model to training mode.

    Args:
        stages: The model's ``build_stages()``.
        batch: One preprocessed example batch.
        device: Device of the tracing run.
    """
    stages = tuple(stages)
    graphs = [stage for stage in stages if isinstance(stage, GraphStage)]
    for stage in graphs:
        stage.module.eval()
    state = run_stages_in_torch(stages, batch, device)
    return {
        stage.name: ExportSpec(
            module=stage.module,
            args=state.graph_inputs[stage.name],
            input_names=list(stage.inputs),
            output_names=list(stage.outputs),
            dynamic_axes={name: dict(axes) for name, axes in stage.onnx_dynamic_axes.items()}
            or None,
        )
        for stage in graphs
    }
