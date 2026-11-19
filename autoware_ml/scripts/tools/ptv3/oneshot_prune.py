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

"""One-shot structured pruning of the PTv3 encoder, without finetuning.

Every encoder block keeps fewer attention heads at the trained head width, fewer MLP hidden
units, or both. The layer after the pruned units (``attn.proj`` for heads, ``mlp.0.fc2`` for
hidden units) is then refit on the kept units so its output stays close to the unpruned one.
The statistics come from a few hundred calibration frames.

Notation for that layer, ``y = W x + b``: for heads, ``x`` concatenates the per-head attention
outputs, so head ``h`` owns the columns ``W_h`` of ``W``; for the MLP, ``x`` is the GELU
output. Statistics of ``x`` are accumulated in float64 over every token of the calibration
frames: the second moment ``M``, the mean ``mu`` and the centred covariance
``S = M / n - mu mu^T``.

* Head selection: removing the head set ``P`` changes the attention output by
  ``E||sum_{h in P} W_h x_h||^2 = sum_{i, j in P} G_ij`` with the head Gram
  ``G_ij = sum_{a in i, b in j} (W^T W)_ab M_ab / n``. The dropped set minimises it,
  exhaustively for up to 200k candidate sets and greedily with a one-swap search above.
* Hidden-unit selection (Wanda per unit): unit ``j`` scores ``||W[:, j]||^2 M_jj / n``, and the
  highest scores are kept.
* Refit: a damped ridge least-squares fit of the kept columns ``K`` to the unpruned output,
  ``min E||W' x_K + b' - (W x + b)||^2 + lam ||W' - W_K||^2``, solved in float64 as
  ``W' = W_K + W_P S_PK (S_KK + lam I)^-1`` and ``b' = b + W mu - W' mu_K``. The bias takes
  the dropped units' mean contribution. ``lam`` is ``damp`` times the mean of
  ``diag(S_KK)``.
* Sequential calibration: blocks are pruned in forward order, the attention of a block before
  its MLP, and the statistics of a layer are taken with everything before it already pruned
  and refit. The fit target is the unpruned layer applied to those inputs.

The pruning runs on the export copy of the encoder, as a mask: the dropped units' columns of
``proj`` / ``fc2`` are zero. The written checkpoint keeps the kept heads of ``qkv`` (laid out
``(3, heads, head_dim, channels)``) and ``proj``, and the kept hidden units of ``fc1`` and
``fc2``, in their original order, so the pruned model is selected by Hydra overrides alone.
Every other tensor is copied unchanged. The optimizer, scheduler and loop state are dropped,
since they describe the unpruned parameters, so the checkpoint is used with ``--weights``.

Calibration frames should come from the train split of the domain the pruned model is meant
for: the refit fits the statistics of its calibration frames, and frames from another domain
can cost accuracy there that target-domain frames keep. By default they come from the train
split of the run's own config; ``prune.calibration_config`` (plus
``prune.calibration_overrides``) and ``prune.calibration_split`` name another dataset and
split. The frames are read with the test transforms. One frame is taken from each of
``num_frames`` scenes spaced evenly over the split, at a golden-ratio position inside the
scene, so the selection is deterministic. The tool refuses calibration frames whose scene is
also in the validation or test split of the calibration dataset.

Usage, halving the heads of every stage at head width 16 and narrowing the MLP from 4 to 3
times the stage width::

    python -m autoware_ml.scripts.tools.ptv3.oneshot_prune \\
        --config-name multi/ptv3/t4dataset/<config> \\
        '++weights=[<trained>.ckpt]' +prune.output=<pruned>.ckpt \\
        '+prune.num_heads=[1,2,4,8,16]' +prune.mlp_ratio=3

Either ``prune.num_heads`` or ``prune.mlp_ratio`` may be left out. Optional keys:
``+prune.calibration_config`` (the run's config),
``'+prune.calibration_overrides=["<key>=<value>",...]'`` (none, each override quoted),
``+prune.calibration_split`` (``train``), ``+prune.num_frames`` (256), ``+prune.damp`` (0.01)
and ``+prune.device`` (``cuda``). The tool logs the overrides that build the pruned model,
e.g. ``model.encoder.enc_num_head=[1,2,4,8,16] +model.encoder.enc_head_dim=16
model.encoder.mlp_ratio=3 model.seg3d_head.mlp_ratio=4``. The seg3d head reads its MLP ratio
from the encoder's, so a narrower encoder MLP pins it back to the trained value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass
import itertools
import logging
import math
from pathlib import Path
import re
from typing import Any

import hydra
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import spconv.constants as spconv_constants
import torch
from torch import nn

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.model_builder import (
    build_data_preprocessor,
    build_model,
    resolve_weights_paths,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import SerializedAttention
from autoware_ml.models.segmentation3d.ptv3_base import (
    ENCODER_EXPORT_POOLING_FIELDS,
    PTv3BaseModel,
    _PTv3EncoderExportModule,
    prepare_ptv3_export_inputs,
)
from autoware_ml.types.dataset import SplitType
from autoware_ml.utils.runtime import get_config_path, set_seed

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()

_BLOCK_NAME = re.compile(r"^enc\.enc(?P<stage>\d+)\.block(?P<block>\d+)$")
# Checkpoint entries describing the training state of the unpruned parameters.
_TRAINING_STATE_KEYS = ("optimizer_states", "lr_schedulers", "loops", "callbacks")
# Above this many candidate drop sets the head selection is greedy.
_MAX_EXHAUSTIVE_SETS = 200_000
_GOLDEN_RATIO_FRACTION = 0.6180339887
_PRUNED_LAYERS = (
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "mlp.0.fc1.weight",
    "mlp.0.fc1.bias",
    "mlp.0.fc2.weight",
    "mlp.0.fc2.bias",
)


class _StopForward(Exception):
    """Raised by a statistics hook once it has the input of its layer."""


@dataclass
class InputStatistics:
    """Float64 sums over the tokens of the input of a linear layer.

    Attributes:
        second_moment: ``sum_n x_n x_n^T``.
        total: ``sum_n x_n``.
        count: Number of tokens.
    """

    second_moment: torch.Tensor
    total: torch.Tensor
    count: int

    @classmethod
    def empty(cls, features: int, device: torch.device) -> InputStatistics:
        """Return zero statistics of ``features``-wide inputs."""
        return cls(
            second_moment=torch.zeros(features, features, dtype=torch.float64, device=device),
            total=torch.zeros(features, dtype=torch.float64, device=device),
            count=0,
        )

    def update(self, inputs: torch.Tensor) -> None:
        """Add the tokens of ``inputs``, whose last dimension is the feature."""
        tokens = inputs.detach().reshape(-1, inputs.shape[-1]).double()
        self.second_moment.addmm_(tokens.t(), tokens)
        self.total += tokens.sum(0)
        self.count += tokens.shape[0]

    @property
    def mean(self) -> torch.Tensor:
        """Mean input ``mu``."""
        return self.total / self.count

    @property
    def covariance(self) -> torch.Tensor:
        """Centred covariance ``M / n - mu mu^T``."""
        mean = self.mean
        return self.second_moment / self.count - torch.outer(mean, mean)


def collect_input_statistics(
    layer: nn.Linear, run_model: Callable[[Any], object], frames: Iterable[Any]
) -> InputStatistics:
    """Accumulate the statistics of the input of ``layer`` over ``frames``.

    Each forward pass stops right after ``layer`` has received its input.

    Args:
        layer: Linear layer whose input is measured.
        run_model: Runs the model on one frame.
        frames: Calibration frames.

    Returns:
        The input statistics.
    """
    statistics = InputStatistics.empty(layer.in_features, layer.weight.device)

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        statistics.update(inputs[0])
        raise _StopForward

    handle = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for frame in frames:
                try:
                    run_model(frame)
                except _StopForward:
                    pass
    finally:
        handle.remove()
    return statistics


def head_gram(
    weight: torch.Tensor, second_moment: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Return the head Gram ``G_ij = E<W_i x_i, W_j x_j>`` of a projection.

    Args:
        weight: Projection weight ``W``, ``(out, num_heads * head_dim)``.
        second_moment: Mean second moment ``M / n`` of the projection input.
        num_heads: Number of heads.
        head_dim: Width of one head.

    Returns:
        The ``(num_heads, num_heads)`` Gram matrix.
    """
    weight = weight.double()
    products = (weight.t() @ weight) * second_moment.double()
    return products.reshape(num_heads, head_dim, num_heads, head_dim).sum((1, 3))


def select_heads(
    gram: torch.Tensor, keep: int, max_exhaustive_sets: int = _MAX_EXHAUSTIVE_SETS
) -> list[int]:
    """Return the heads to keep: the dropped set ``P`` minimises ``sum_{i, j in P} G_ij``.

    Args:
        gram: Head Gram matrix, see :func:`head_gram`.
        keep: Number of heads to keep.
        max_exhaustive_sets: Search every drop set up to this many, otherwise pick greedily
            and improve by single swaps.

    Returns:
        The kept heads, ascending.

    Raises:
        ValueError: If ``keep`` is not between 1 and the head count.
    """
    num_heads = gram.shape[0]
    if not 1 <= keep <= num_heads:
        raise ValueError(f"Cannot keep {keep} of {num_heads} heads.")
    drop = num_heads - keep
    values = gram.double().cpu().numpy()

    def energy(heads: Sequence[int]) -> float:
        heads = list(heads)
        return float(values[heads][:, heads].sum())

    if math.comb(num_heads, drop) <= max_exhaustive_sets:
        dropped = list(min(itertools.combinations(range(num_heads), drop), key=energy))
    else:
        dropped = []
        for _ in range(drop):
            candidates = [head for head in range(num_heads) if head not in dropped]
            dropped.append(min(candidates, key=lambda head: energy([*dropped, head])))
        improved = True
        while improved:
            improved = False
            current = energy(dropped)
            for index, head in itertools.product(
                range(drop), [head for head in range(num_heads) if head not in dropped]
            ):
                swapped = dropped.copy()
                swapped[index] = head
                if energy(swapped) < current - 1e-12 * abs(current):
                    dropped, improved = swapped, True
                    break
    return [head for head in range(num_heads) if head not in dropped]


def head_columns(heads: Sequence[int], head_dim: int) -> list[int]:
    """Return the input columns of a projection that belong to ``heads``."""
    return [head * head_dim + channel for head in heads for channel in range(head_dim)]


def select_hidden_units(weight: torch.Tensor, second_moment: torch.Tensor, keep: int) -> list[int]:
    """Return the MLP hidden units to keep, by the Wanda score ``||W[:, j]||^2 M_jj``.

    Args:
        weight: Weight of the layer after the hidden units, ``(out, hidden)``.
        second_moment: Mean second moment ``M / n`` of the hidden units.
        keep: Number of units to keep.

    Returns:
        The kept units, ascending.

    Raises:
        ValueError: If ``keep`` is not between 1 and the unit count.
    """
    if not 1 <= keep <= weight.shape[1]:
        raise ValueError(f"Cannot keep {keep} of {weight.shape[1]} hidden units.")
    scores = weight.double().square().sum(0) * torch.diagonal(second_moment.double())
    return sorted(torch.argsort(scores, descending=True)[:keep].tolist())


def refit_linear(
    weight: torch.Tensor,
    bias: torch.Tensor,
    statistics: InputStatistics,
    kept_columns: Sequence[int],
    damp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refit ``y = W x + b`` on the kept input columns by damped ridge least squares.

    Solves ``min E||W' x_K + b' - (W x + b)||^2 + lam ||W' - W_K||^2`` in float64:
    ``W' = W_K + W_P S_PK (S_KK + lam I)^-1`` and ``b' = b + W mu - W' mu_K``, with the
    centred input covariance ``S``, the dropped columns ``P`` and
    ``lam = damp * mean(diag(S_KK))``.

    Args:
        weight: Weight ``W``, ``(out, in)``.
        bias: Bias ``b``.
        statistics: Statistics of the layer input.
        kept_columns: Kept input columns ``K``.
        damp: Ridge strength relative to the mean kept input variance.

    Returns:
        ``(W', b')`` in the dtype of ``weight``; ``W'`` has the columns ``K`` only.
    """
    kept = torch.as_tensor(list(kept_columns), dtype=torch.long, device=weight.device)
    dropped_mask = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device)
    dropped_mask[kept] = False
    dropped = dropped_mask.nonzero().flatten()
    weight64, bias64 = weight.double(), bias.double()
    mean, covariance = statistics.mean, statistics.covariance
    covariance_kept = covariance[kept][:, kept]
    damping = damp * float(torch.diagonal(covariance_kept).mean())
    covariance_kept = covariance_kept + damping * torch.eye(
        len(kept), dtype=torch.float64, device=weight.device
    )
    cholesky = torch.linalg.cholesky(covariance_kept)
    # W_P S_PK (S_KK + lam I)^-1, solved as (S_KK + lam I) X^T = S_KP W_P^T.
    correction = torch.cholesky_solve(
        covariance[kept][:, dropped] @ weight64[:, dropped].t(), cholesky
    ).t()
    refit_weight = weight64[:, kept] + correction
    refit_bias = bias64 + weight64 @ mean - refit_weight @ mean[kept]
    return refit_weight.to(weight.dtype), refit_bias.to(bias.dtype)


def encoder_blocks(encoder: nn.Module) -> list[tuple[str, int, nn.Module]]:
    """Return ``(name, stage, block)`` of every encoder block with attention, in forward order.

    Blocks are the modules named ``enc.enc<stage>.block<index>``.
    """
    blocks = []
    for name, module in encoder.named_modules():
        match = _BLOCK_NAME.match(name)
        if match is not None and getattr(module, "attn", None) is not None:
            blocks.append((int(match["stage"]), int(match["block"]), name, module))
    blocks.sort(key=lambda item: item[:2])
    return [(name, stage, module) for stage, _, name, module in blocks]


@dataclass(frozen=True)
class AttentionPruning:
    """Kept heads of one block.

    Attributes:
        heads: Kept heads, ascending.
        num_heads: Trained head count.
        head_dim: Trained head width.
    """

    heads: list[int]
    num_heads: int
    head_dim: int


@dataclass(frozen=True)
class BlockPruning:
    """Kept units of one block.

    Attributes:
        attention: Kept heads, or ``None`` when the heads are not pruned.
        hidden_units: Kept MLP hidden units, ascending, or ``None`` when the MLP is not pruned.
    """

    attention: AttentionPruning | None = None
    hidden_units: list[int] | None = None


def _refit_masked(
    layer: nn.Linear, statistics: InputStatistics, kept_columns: Sequence[int], damp: float
) -> None:
    """Refit ``layer`` on ``kept_columns`` in place and zero its other input columns."""
    weight, bias = refit_linear(layer.weight, layer.bias, statistics, kept_columns, damp)
    layer.weight.zero_()
    layer.weight[:, list(kept_columns)] = weight
    layer.bias.copy_(bias)


@torch.no_grad()
def prune_encoder(
    encoder: nn.Module,
    run_model: Callable[[Any], object],
    frames: Sequence[Any],
    damp: float,
    num_heads: Sequence[int] | None = None,
    mlp_ratio: float | None = None,
) -> dict[str, BlockPruning]:
    """Prune ``encoder`` in place, as a mask, with sequential calibration.

    Every block of stage ``s`` keeps ``num_heads[s]`` heads and ``int(channels * mlp_ratio)``
    MLP hidden units. The dropped units' columns of ``attn.proj`` / ``mlp.0.fc2`` become zero
    and the kept ones hold the refit weights, see :func:`refit_linear`. A block's attention is
    pruned before its MLP.

    Args:
        encoder: PTv3 encoder, changed in place.
        run_model: Runs the model that contains ``encoder`` on one frame.
        frames: Calibration frames.
        damp: Ridge strength of the refit.
        num_heads: Heads kept per stage, or ``None`` to keep every head.
        mlp_ratio: Target MLP expansion ratio, or ``None`` to keep every hidden unit.

    Returns:
        The kept units of every block, by block name.

    Raises:
        ValueError: If ``num_heads`` does not have one entry per stage.
    """
    blocks = encoder_blocks(encoder)
    stage_count = max(stage for _, stage, _ in blocks) + 1
    if num_heads is not None and len(num_heads) != stage_count:
        raise ValueError(f"num_heads must have {stage_count} entries, got {len(num_heads)}.")
    pruning = {}
    for name, stage, block in blocks:
        attention_pruning = None
        if num_heads is not None:
            attention: SerializedAttention = block.attn
            proj = attention.proj
            statistics = collect_input_statistics(proj, run_model, frames)
            gram = head_gram(
                proj.weight,
                statistics.second_moment / statistics.count,
                attention.num_heads,
                attention.head_dim,
            )
            heads = select_heads(gram, num_heads[stage])
            _refit_masked(proj, statistics, head_columns(heads, attention.head_dim), damp)
            attention_pruning = AttentionPruning(heads, attention.num_heads, attention.head_dim)
            logger.info(
                "%s: kept heads %s of %d (%d tokens)",
                name,
                heads,
                attention.num_heads,
                statistics.count,
            )
        hidden_units = None
        if mlp_ratio is not None:
            fc2 = block.mlp.get_submodule("0.fc2")
            statistics = collect_input_statistics(fc2, run_model, frames)
            hidden_units = select_hidden_units(
                fc2.weight,
                statistics.second_moment / statistics.count,
                int(fc2.out_features * mlp_ratio),
            )
            _refit_masked(fc2, statistics, hidden_units, damp)
            logger.info(
                "%s: kept %d of %d MLP hidden units (%d tokens)",
                name,
                len(hidden_units),
                fc2.in_features,
                statistics.count,
            )
        pruning[name] = BlockPruning(attention_pruning, hidden_units)
    return pruning


def slice_pruned_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    masked_encoder: nn.Module,
    pruning: Mapping[str, BlockPruning],
    prefix: str = "encoder.",
) -> dict[str, torch.Tensor]:
    """Return the state dict of the pruned model.

    ``qkv`` keeps the rows of the kept heads of ``state_dict`` and ``fc1`` those of the kept
    hidden units. ``proj`` and ``fc2`` take the refit weights of ``masked_encoder`` on the kept
    columns. Every other tensor is copied.

    Args:
        state_dict: State dict of the unpruned model.
        masked_encoder: Encoder pruned by :func:`prune_encoder`.
        pruning: Kept units by block name, relative to the encoder.
        prefix: Key prefix of the encoder inside ``state_dict``.

    Returns:
        The pruned state dict; every tensor is a contiguous copy on the CPU.
    """
    pruned = {key: tensor.detach().cpu().clone() for key, tensor in state_dict.items()}
    modules = dict(masked_encoder.named_modules())
    for name, block in pruning.items():
        if block.hidden_units is not None:
            key = f"{prefix}{name}.mlp.0"
            units = torch.as_tensor(block.hidden_units, dtype=torch.long)
            pruned[f"{key}.fc1.weight"] = pruned[f"{key}.fc1.weight"][units].contiguous()
            pruned[f"{key}.fc1.bias"] = pruned[f"{key}.fc1.bias"][units].contiguous()
            fc2 = modules[f"{name}.mlp.0.fc2"]
            pruned[f"{key}.fc2.weight"] = fc2.weight.detach()[:, units].cpu().contiguous()
            pruned[f"{key}.fc2.bias"] = fc2.bias.detach().cpu().clone()
        kept = block.attention
        if kept is None:
            continue
        key = f"{prefix}{name}.attn"
        heads = torch.as_tensor(kept.heads, dtype=torch.long)
        qkv_weight = pruned[f"{key}.qkv.weight"]
        channels = qkv_weight.shape[1]
        pruned[f"{key}.qkv.weight"] = (
            qkv_weight.reshape(3, kept.num_heads, kept.head_dim, channels)[:, heads]
            .reshape(-1, channels)
            .contiguous()
        )
        if f"{key}.qkv.bias" in pruned:
            pruned[f"{key}.qkv.bias"] = (
                pruned[f"{key}.qkv.bias"]
                .reshape(3, kept.num_heads, kept.head_dim)[:, heads]
                .reshape(-1)
                .contiguous()
            )
        proj = modules[f"{name}.attn.proj"]
        columns = head_columns(kept.heads, kept.head_dim)
        pruned[f"{key}.proj.weight"] = proj.weight.detach()[:, columns].cpu().contiguous()
        pruned[f"{key}.proj.bias"] = proj.bias.detach().cpu().clone()
    return pruned


def calibration_frame_indices(scene_groups: Sequence[Sequence[int]], num_frames: int) -> list[int]:
    """Pick one frame from each of ``num_frames`` scenes spaced evenly over the split.

    Inside the ``j``-th picked scene the frame sits at the fraction ``(0.618 j) mod 1``.

    Args:
        scene_groups: Dataset indices of every scene, in split order; empty scenes are skipped.
        num_frames: Number of frames. Fewer come back when the split has fewer scenes.

    Returns:
        Dataset indices of the calibration frames.
    """
    groups = [group for group in scene_groups if group]
    last = len(groups) - 1
    scenes = sorted({round(index * last / max(num_frames - 1, 1)) for index in range(num_frames)})
    indices = []
    for position, scene in enumerate(scenes):
        group = groups[scene]
        indices.append(group[int(((position * _GOLDEN_RATIO_FRACTION) % 1.0) * len(group))])
    return indices


def pruned_model_overrides(
    pruning: Mapping[str, BlockPruning],
    mlp_ratio: float | None = None,
    seg3d_head_mlp_ratio: float | None = None,
) -> list[str]:
    """Return the Hydra overrides that build the pruned encoder.

    Args:
        pruning: Kept units by block name, every stage's blocks included.
        mlp_ratio: Target MLP expansion ratio, or ``None`` when the MLP is not pruned.
        seg3d_head_mlp_ratio: Trained MLP ratio of a seg3d head that reads the encoder's, to pin
            it back, or ``None`` without such a head.

    Returns:
        The overrides.
    """
    overrides = []
    attention = {
        int(_BLOCK_NAME.match(name)["stage"]): block.attention
        for name, block in pruning.items()
        if block.attention is not None
    }
    if attention:
        stage_count = max(attention) + 1
        heads = [len(attention[stage].heads) for stage in range(stage_count)]
        head_dims = [attention[stage].head_dim for stage in range(stage_count)]
        head_dim = (
            head_dims[0] if len(set(head_dims)) == 1 else f"[{','.join(map(str, head_dims))}]"
        )
        overrides += [
            f"model.encoder.enc_num_head=[{','.join(map(str, heads))}]",
            f"+model.encoder.enc_head_dim={head_dim}",
        ]
    if mlp_ratio is not None:
        overrides.append(f"model.encoder.mlp_ratio={mlp_ratio}")
        if seg3d_head_mlp_ratio is not None:
            overrides.append(f"model.seg3d_head.mlp_ratio={seg3d_head_mlp_ratio}")
    return overrides


def _use_strict_fp32() -> None:
    """Keep the calibration forward passes in full fp32, without TF32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    spconv_constants.SPCONV_ALLOW_TF32 = False


def check_calibration_scenes(
    calibration_scenes: Iterable[str], held_out_scenes: Mapping[str, set[str]]
) -> None:
    """Refuse calibration frames from scenes of a held-out split.

    Args:
        calibration_scenes: Scenario id of every calibration frame.
        held_out_scenes: Scenario ids of every held-out split, by split name.

    Raises:
        ValueError: If a calibration scene is in a held-out split.
    """
    calibration = set(calibration_scenes)
    for split, scenes in held_out_scenes.items():
        shared = sorted(calibration & scenes)
        if shared:
            raise ValueError(
                f"{len(shared)} calibration scenes are in the {split} split, e.g. {shared[0]}. "
                "Calibrate on the train split."
            )


def _calibration_config(cfg: DictConfig, options: Mapping[str, Any]) -> DictConfig:
    """Return the config of the calibration dataset: the run's own or a composed one."""
    name = options.get("calibration_config")
    overrides = [str(value) for value in options.get("calibration_overrides") or []]
    if name is None:
        if overrides:
            raise ValueError("'prune.calibration_overrides' needs 'prune.calibration_config'.")
        return copy.deepcopy(cfg)
    searchpath = list(HydraConfig.get().searchpath)
    if searchpath:
        overrides.insert(0, f"hydra.searchpath=[{','.join(searchpath)}]")
    return compose(config_name=str(name), overrides=overrides)


def _scenario_id(dataset: Any, index: int) -> str:
    source, sample = dataset.locate(index)
    records = dataset.datasets[source]
    row = records.dataset_records_dataframe.row(records.record_index(sample), named=True)
    return str(row["scenario_id"])


def _load_calibration_inputs(
    model: PTv3BaseModel,
    cfg: DictConfig,
    split: SplitType,
    num_frames: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, ...]]:
    """Return the export-encoder inputs of the calibration frames, on the CPU."""
    if split == SplitType.TRAIN:
        cfg.datasets.train.transforms = cfg.datasets.test.transforms
    datamodule = build_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup_splits((split,))
    dataset = datamodule.datasets[split]
    indices = calibration_frame_indices(dataset.scene_index_groups(), num_frames)
    held_out = {}
    for other in (SplitType.VAL, SplitType.TEST):
        if other == split:
            continue
        try:
            datamodule.setup_splits((other,))
        except ValueError as error:  # served by sources but empty: no scene to avoid
            logger.info("No %s records to check the calibration scenes against: %s", other, error)
            continue
        if other in datamodule.datasets:
            held_out[other.value] = {
                str(scenario)
                for records in datamodule.datasets[other].datasets
                for scenario in records.dataset_records_dataframe.get_column("scenario_id")
            }
    check_calibration_scenes((_scenario_id(dataset, index) for index in indices), held_out)
    frames = []
    with torch.no_grad():
        for index in indices:
            batch = dataset.collate_fn([dataset[index]])
            batch_inputs = model.on_after_batch_transfer(batch.to_device(device), dataloader_idx=0)
            inputs = prepare_ptv3_export_inputs(model, batch_inputs)
            args, _ = inputs.encoder_args(ENCODER_EXPORT_POOLING_FIELDS)
            frames.append(tuple(arg.detach().cpu() for arg in args))
    logger.info(
        "Calibration: %d frames of the %s split of %s, none in a scene of %s.",
        len(frames),
        split.value,
        OmegaConf.select(cfg, "database.version", default="the calibration config"),
        sorted(held_out) or "no other split",
    )
    return frames


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Prune a trained PTv3 encoder and write the pruned checkpoint.

    Args:
        cfg: Hydra configuration with one ``weights`` path and the ``prune`` options.

    Raises:
        ValueError: If not exactly one weights path, no ``prune.output`` or neither
            ``prune.num_heads`` nor ``prune.mlp_ratio`` is given.
        NotImplementedError: If the configured model is not a PTv3 model.
    """
    options = cfg.get("prune", None) or {}
    weights = resolve_weights_paths(cfg)
    if weights is None or len(weights) != 1:
        raise ValueError("Pass exactly one checkpoint: '++weights=[<path>]'.")
    if options.get("output") is None:
        raise ValueError("Pass '+prune.output=<path>'.")
    if options.get("num_heads") is None and options.get("mlp_ratio") is None:
        raise ValueError("Pass '+prune.num_heads=[...]', '+prune.mlp_ratio=<ratio>' or both.")
    num_heads = options.get("num_heads")
    num_heads = None if num_heads is None else [int(value) for value in num_heads]
    mlp_ratio = options.get("mlp_ratio")
    num_frames = int(options.get("num_frames", 256))
    calibration_split = SplitType(options.get("calibration_split", "train"))
    damp = float(options.get("damp", 0.01))
    device = torch.device(options.get("device", "cuda"))

    _use_strict_fp32()
    set_seed(cfg)
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        device=device,
        weights_path=weights,
        resume_checkpoint_path=None,
        set_eval=True,
        enforce_full_coverage=True,
    )
    if not isinstance(model, PTv3BaseModel):
        raise NotImplementedError(
            f"{Path(__file__).stem} supports only PTv3 models, got {type(model).__name__}."
        )
    frames = _load_calibration_inputs(
        model, _calibration_config(cfg, options), calibration_split, num_frames, device
    )
    sparse_shape, serialization_depth = model.compute_export_geometry(device)
    export_module = _PTv3EncoderExportModule(
        encoder=model._prepare_encoder_export(),
        voxel_encoder=model.voxel_encoder,
        sparse_shape=sparse_shape,
        serialized_depth=serialization_depth,
    ).eval()
    # The export copy may fold some layers; the pruned ones must still be the trained ones.
    trained = model.encoder.state_dict()
    for name, value in export_module.encoder.state_dict().items():
        if name.endswith(_PRUNED_LAYERS):
            if not torch.equal(value, trained[name]):
                raise RuntimeError(f"The export copy changed {name}.")

    def run_model(frame: tuple[torch.Tensor, ...]) -> object:
        return export_module(*(arg.to(device) for arg in frame))

    pruning = prune_encoder(
        export_module.encoder, run_model, frames, damp, num_heads=num_heads, mlp_ratio=mlp_ratio
    )
    checkpoint = torch.load(weights[0], map_location="cpu", weights_only=False)
    if set(checkpoint["state_dict"]) != set(model.state_dict()):
        raise ValueError("The checkpoint does not hold exactly the tensors of the model.")
    payload = {
        key: value
        for key, value in checkpoint.items()
        if key not in _TRAINING_STATE_KEYS and key != "state_dict"
    }
    payload["state_dict"] = slice_pruned_state_dict(
        checkpoint["state_dict"], export_module.encoder, pruning
    )
    payload["autoware_ml_prune"] = {
        "type": "ptv3_encoder_oneshot",
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "calibration_config": options.get("calibration_config"),
        "calibration_split": calibration_split.value,
        "num_frames": len(frames),
        "damp": damp,
        "kept_heads": {
            name: block.attention.heads
            for name, block in pruning.items()
            if block.attention is not None
        },
        "kept_hidden_units": {
            name: block.hidden_units
            for name, block in pruning.items()
            if block.hidden_units is not None
        },
    }
    output = Path(options["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    logger.info(
        "Wrote %s. Build the pruned model with: %s",
        output,
        " ".join(
            pruned_model_overrides(
                pruning,
                mlp_ratio,
                OmegaConf.select(cfg, "model.seg3d_head.mlp_ratio", default=None),
            )
        ),
    )


if __name__ == "__main__":
    main()
