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

"""Slice a PTv3 checkpoint onto a structurally pruned encoder.

The pruned encoder is the same PTv3 with, per stage, fewer blocks, a narrower MLP and
fewer attention heads at the trained head width. The slices keep a prefix of each:

* depth: blocks ``block0 .. block{k-1}`` of every stage; later blocks are dropped;
* MLP: the first ``channels * mlp_ratio`` hidden units, i.e. ``fc1.weight[:h]``,
  ``fc1.bias[:h]`` and ``fc2.weight[:, :h]``;
* heads: the first ``num_heads`` heads of q, k and v, with the ``qkv`` rows laid out
  ``(3, heads, head_dim, channels)``, and the matching ``proj.weight[:, :heads * head_dim]``.

Every other tensor, stage widths and the task heads included, is copied unchanged. The
pruned model is selected by Hydra overrides alone, e.g. for depth 1, MLP ratio 2 and
half the heads at head width 16::

    model.encoder.enc_depths=[1,1,1,1,1] model.encoder.mlp_ratio=2
    model.encoder.enc_num_head=[1,2,4,8,16] +model.encoder.enc_head_dim=16
    model.seg3d_head.mlp_ratio=4

The seg3d head reads its MLP ratio from the encoder's by interpolation, so it is pinned
back to the trained value. The written checkpoint holds the sliced ``state_dict`` and the
prune provenance; optimizer, scheduler and loop state are dropped, since they describe
the unpruned parameters. Use it with ``--weights``, not ``--resume-checkpoint``.

Usage::

    python -m autoware_ml.scripts.tools.ptv3.prune_checkpoint IN.ckpt OUT.ckpt \
        --depths 1 1 1 1 1 --mlp-ratio 2 --num-heads 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_BLOCK_KEY = re.compile(r"^(?P<stage>enc\.enc(?P<s>\d+))\.block(?P<b>\d+)\.(?P<rest>.+)$")
# Checkpoint entries describing the training state of the unpruned parameters.
_TRAINING_STATE_KEYS = ("optimizer_states", "lr_schedulers", "loops", "callbacks")


def _stage_layout(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict[int, int]:
    """Return the block count of every encoder stage found under ``prefix``."""
    depths: dict[int, int] = {}
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        match = _BLOCK_KEY.match(key[len(prefix) :])
        if match is not None:
            stage, block = int(match["s"]), int(match["b"])
            depths[stage] = max(depths.get(stage, 0), block + 1)
    if not depths:
        raise ValueError(f"No encoder blocks found under the prefix {prefix!r}.")
    if sorted(depths) != list(range(len(depths))):
        raise ValueError(f"Encoder stages are not contiguous: {sorted(depths)}.")
    return depths


def _per_stage(value: Sequence[Any] | None, stage_count: int, name: str) -> list[Any]:
    if value is None:
        return [None] * stage_count
    values = list(value)
    if len(values) == 1:
        return values * stage_count
    if len(values) != stage_count:
        raise ValueError(f"{name} must have 1 or {stage_count} entries, got {len(values)}.")
    return values


def prune_ptv3_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    depths: Sequence[int] | None = None,
    mlp_ratio: float | None = None,
    num_heads: Sequence[int] | None = None,
    head_dim: int | None = None,
    prefix: str = "encoder.",
) -> dict[str, torch.Tensor]:
    """Slice the encoder tensors of a PTv3 state dict onto a pruned encoder.

    Args:
        state_dict: State dict of the trained model.
        depths: Blocks kept per stage (a prefix of each stage), one value for every stage
            or one per stage. ``None`` keeps every block.
        mlp_ratio: Target MLP expansion ratio. ``None`` keeps the trained width.
        num_heads: Attention heads kept per stage (a prefix), one value for every stage or
            one per stage. ``None`` keeps every head.
        head_dim: Trained per-head width, required with ``num_heads``. The checkpoint
            holds no head count, so the ``qkv`` layout cannot be inferred from it.
        prefix: Key prefix of the encoder inside the state dict.

    Returns:
        The sliced state dict; every tensor is a contiguous copy.

    Raises:
        ValueError: When a target is larger than the trained model or does not match the
            trained layout, or the state dict holds no PTv3 encoder blocks.
    """
    trained_depths = _stage_layout(state_dict, prefix)
    stage_count = len(trained_depths)
    keep_depths = _per_stage(depths, stage_count, "depths")
    keep_heads = _per_stage(num_heads, stage_count, "num_heads")
    if num_heads is not None and head_dim is None:
        raise ValueError("head_dim is required to slice attention heads.")
    for stage, keep in enumerate(keep_depths):
        if keep is not None and not 1 <= keep <= trained_depths[stage]:
            raise ValueError(
                f"Stage {stage} has {trained_depths[stage]} trained blocks, cannot keep {keep}."
            )

    pruned: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        match = _BLOCK_KEY.match(key[len(prefix) :]) if key.startswith(prefix) else None
        if match is None:
            pruned[key] = tensor.clone()
            continue
        stage, block, rest = int(match["s"]), int(match["b"]), match["rest"]
        if keep_depths[stage] is not None and block >= keep_depths[stage]:
            continue
        block_prefix = key[: -len(rest)]
        pruned[key] = _slice_block_tensor(
            rest,
            tensor,
            fc1_weight=state_dict.get(block_prefix + "mlp.0.fc1.weight"),
            mlp_ratio=mlp_ratio,
            heads=keep_heads[stage],
            head_dim=head_dim,
            key=key,
        )
    return pruned


def _slice_block_tensor(
    rest: str,
    tensor: torch.Tensor,
    *,
    fc1_weight: torch.Tensor | None,
    mlp_ratio: float | None,
    heads: int | None,
    head_dim: int | None,
    key: str,
) -> torch.Tensor:
    """Slice one tensor of a kept block; ``rest`` is its key below ``blockN.``.

    ``fc1_weight`` is the block's trained ``mlp.0.fc1.weight``, which sets the stage width
    the kept MLP width is derived from.
    """
    if mlp_ratio is not None and rest in ("mlp.0.fc1.weight", "mlp.0.fc1.bias", "mlp.0.fc2.weight"):
        assert fc1_weight is not None
        hidden, channels = fc1_weight.shape
        keep = int(channels * mlp_ratio)  # as MLP() sizes its hidden layer
        if not 1 <= keep <= hidden:
            raise ValueError(f"{key}: cannot slice the MLP from {hidden} to {keep} hidden units.")
        if rest == "mlp.0.fc2.weight":
            return tensor[:, :keep].contiguous().clone()
        return tensor[:keep].contiguous().clone()

    if heads is not None and rest in ("attn.qkv.weight", "attn.qkv.bias"):
        assert head_dim is not None
        trained_heads, remainder = divmod(tensor.shape[0], 3 * head_dim)
        if remainder or not 1 <= heads <= trained_heads:
            raise ValueError(
                f"{key}: {tensor.shape[0]} rows are not 3 x heads x {head_dim} with at least "
                f"{heads} heads."
            )
        tail = tensor.shape[1:]
        sliced = tensor.reshape(3, trained_heads, head_dim, *tail)[:, :heads]
        return sliced.reshape(3 * heads * head_dim, *tail).contiguous().clone()
    if heads is not None and rest == "attn.proj.weight":
        assert head_dim is not None
        return tensor[:, : heads * head_dim].contiguous().clone()
    return tensor.clone()


def prune_ptv3_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    depths: Sequence[int] | None = None,
    mlp_ratio: float | None = None,
    num_heads: Sequence[int] | None = None,
    head_dim: int | None = None,
    prefix: str = "encoder.",
) -> dict[str, Any]:
    """Return a weights-only checkpoint whose ``state_dict`` is sliced for the pruned encoder.

    Args:
        checkpoint: Lightning checkpoint payload of the trained model.
        depths: See :func:`prune_ptv3_state_dict`.
        mlp_ratio: See :func:`prune_ptv3_state_dict`.
        num_heads: See :func:`prune_ptv3_state_dict`.
        head_dim: See :func:`prune_ptv3_state_dict`.
        prefix: See :func:`prune_ptv3_state_dict`.

    Returns:
        Checkpoint payload with the sliced ``state_dict`` and an ``autoware_ml_prune``
        provenance record; the unpruned optimizer, scheduler and loop state are dropped.
    """
    state_dict = prune_ptv3_state_dict(
        checkpoint["state_dict"],
        depths=depths,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        head_dim=head_dim,
        prefix=prefix,
    )
    pruned = {k: v for k, v in checkpoint.items() if k not in _TRAINING_STATE_KEYS}
    pruned["state_dict"] = state_dict
    pruned["autoware_ml_prune"] = {
        "type": "ptv3_encoder_prefix_slice",
        "depths": None if depths is None else list(depths),
        "mlp_ratio": mlp_ratio,
        "num_heads": None if num_heads is None else list(num_heads),
        "head_dim": head_dim,
        "prefix": prefix,
        "source_epoch": checkpoint.get("epoch"),
        "source_global_step": checkpoint.get("global_step"),
    }
    return pruned


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="Trained Lightning checkpoint.")
    parser.add_argument("output", type=Path, help="Path of the pruned checkpoint to write.")
    parser.add_argument(
        "--depths", type=int, nargs="+", help="Blocks kept per stage (one value or one per stage)."
    )
    parser.add_argument("--mlp-ratio", type=float, help="Target MLP expansion ratio.")
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        help="Attention heads kept per stage (one value or one per stage).",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=16,
        help="Trained per-head width, enc_channels / enc_num_head (16 for every PTv3 config).",
    )
    parser.add_argument("--prefix", default="encoder.", help="State-dict prefix of the encoder.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Read a checkpoint, slice its encoder and write the pruned checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    pruned = prune_ptv3_checkpoint(
        checkpoint,
        depths=args.depths,
        mlp_ratio=args.mlp_ratio,
        num_heads=args.num_heads,
        head_dim=args.head_dim if args.num_heads is not None else None,
        prefix=args.prefix,
    )
    before = sum(
        v.numel() for k, v in checkpoint["state_dict"].items() if k.startswith(args.prefix)
    )
    after = sum(v.numel() for k, v in pruned["state_dict"].items() if k.startswith(args.prefix))
    logger.info(
        "Encoder tensors: %d -> %d keys, %d -> %d elements (%.1f%%).",
        sum(k.startswith(args.prefix) for k in checkpoint["state_dict"]),
        sum(k.startswith(args.prefix) for k in pruned["state_dict"]),
        before,
        after,
        100.0 * after / before,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned, args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
