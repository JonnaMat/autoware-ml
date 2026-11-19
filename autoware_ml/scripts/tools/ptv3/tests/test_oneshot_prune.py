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

"""Unit tests for the one-shot pruning of the PTv3 encoder."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import itertools
import subprocess
import sys

import pytest
import torch
from torch import nn

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    MLP,
    Point,
    PointSequential,
    SerializedAttention,
)
from autoware_ml.scripts.tools.ptv3.oneshot_prune import (
    InputStatistics,
    calibration_frame_indices,
    check_calibration_scenes,
    collect_input_statistics,
    head_columns,
    head_gram,
    prune_encoder,
    pruned_model_overrides,
    refit_linear,
    select_heads,
    select_hidden_units,
    slice_pruned_state_dict,
)

CHANNELS = 16


class _Block(nn.Module):
    """A PTv3 block without the sparse positional convolution: attention, then MLP."""

    def __init__(self, num_heads: int, head_dim: int | None, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = PointSequential(nn.LayerNorm(CHANNELS))
        self.attn = SerializedAttention(
            channels=CHANNELS,
            num_heads=num_heads,
            patch_size=8,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            order_index=0,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
            head_dim=head_dim,
        )
        self.norm2 = PointSequential(nn.LayerNorm(CHANNELS))
        self.mlp = PointSequential(MLP(CHANNELS, mlp_ratio, 0.0))

    def forward(self, point: Point) -> Point:
        shortcut = point.feat
        point = self.attn(self.norm1(point))
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.mlp(self.norm2(point))
        point.feat = shortcut + point.feat
        return point


class _Encoder(nn.Module):
    """Two stages of two blocks, named like the PTv3 encoder (``enc.enc<s>.block<b>``)."""

    def __init__(
        self,
        num_heads: Sequence[int] = (4, 2),
        head_dim: Sequence[int | None] = (None, None),
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.enc = nn.ModuleDict(
            {
                f"enc{stage}": nn.ModuleDict(
                    {
                        f"block{block}": _Block(num_heads[stage], head_dim[stage], mlp_ratio)
                        for block in range(2)
                    }
                )
                for stage in range(2)
            }
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        order = torch.arange(feat.shape[0]).unsqueeze(0)
        point = Point(
            feat=feat,
            grid_coord=torch.zeros(feat.shape[0], 3, dtype=torch.long),
            offset=torch.tensor([feat.shape[0]]),
            serialized_order=order,
            serialized_inverse=order.clone(),
        )
        for stage in self.enc.values():
            for block in stage.values():
                point = block(point)
        return point.feat


def _frames(count: int, seed: int) -> list[torch.Tensor]:
    """Correlated, off-centre token features, as the inputs of a trained encoder are."""
    generator = torch.Generator().manual_seed(seed)
    mixing = torch.randn(CHANNELS, CHANNELS, generator=generator)
    return [torch.randn(24, CHANNELS, generator=generator) @ mixing + 0.5 for _ in range(count)]


def _trained_encoder() -> _Encoder:
    torch.manual_seed(0)
    return _Encoder().eval()


def _first_units_pruning(
    encoder: _Encoder, num_heads: Sequence[int] | None, mlp_ratio: float | None
) -> None:
    """Drop all but the first heads or hidden units of every block, without a refit."""
    with torch.no_grad():
        for stage_index, stage in enumerate(encoder.enc.values()):
            for block in stage.values():
                if num_heads is not None:
                    keep = num_heads[stage_index] * block.attn.head_dim
                    block.attn.proj.weight[:, keep:] = 0.0
                if mlp_ratio is not None:
                    block.mlp.get_submodule("0.fc2").weight[:, int(CHANNELS * mlp_ratio) :] = 0.0


def test_statistics_accumulate_every_token_in_float64() -> None:
    """
    Input: two batches of tokens, one of them 3-d, fed to the input of a linear layer.
    Expected: the float64 sums over all tokens, and the hook stops the forward pass.
    Check: second moment, mean and covariance against a direct computation.
    """
    layer = nn.Linear(3, 2)
    batches = [torch.randn(5, 3), torch.randn(2, 4, 3)]
    after = []

    def run(batch: torch.Tensor) -> None:
        layer(batch)
        after.append(batch)

    statistics = collect_input_statistics(layer, run, batches)
    tokens = torch.cat([batch.reshape(-1, 3) for batch in batches]).double()

    assert after == []
    assert statistics.count == 13
    assert statistics.second_moment.dtype == torch.float64
    torch.testing.assert_close(statistics.second_moment, tokens.t() @ tokens)
    torch.testing.assert_close(statistics.mean, tokens.mean(0))
    torch.testing.assert_close(
        statistics.covariance, torch.cov(tokens.t(), correction=0), rtol=1e-9, atol=1e-12
    )


def test_refit_with_nothing_dropped_is_the_identity() -> None:
    """
    Input: a linear layer refit on all of its input columns, also inside the encoder pruned to
        its trained head counts and MLP ratio.
    Expected: the weights, the bias and the encoder outputs are unchanged.
    Check: allclose at fp32 rounding.
    """
    torch.manual_seed(0)
    weight, bias = torch.randn(6, 8), torch.randn(6)
    statistics = InputStatistics.empty(8, torch.device("cpu"))
    statistics.update(torch.randn(50, 8) + 1.0)
    refit_weight, refit_bias = refit_linear(weight, bias, statistics, range(8), damp=0.01)
    torch.testing.assert_close(refit_weight, weight)
    torch.testing.assert_close(refit_bias, bias)

    encoder = _trained_encoder()
    reference = copy.deepcopy(encoder)
    frames = _frames(4, seed=1)
    pruning = prune_encoder(encoder, encoder, frames, 0.01, num_heads=[4, 2], mlp_ratio=4.0)

    assert [block.attention.heads for block in pruning.values()] == [[0, 1, 2, 3]] * 2 + [
        [0, 1]
    ] * 2
    assert [block.hidden_units for block in pruning.values()] == [list(range(64))] * 4
    with torch.no_grad():
        for frame in frames:
            torch.testing.assert_close(encoder(frame), reference(frame), rtol=1e-5, atol=1e-5)


_SHAPES = [
    pytest.param(
        [2, 1],
        None,
        ["model.encoder.enc_num_head=[2,1]", "+model.encoder.enc_head_dim=[4,8]"],
        id="heads",
    ),
    pytest.param(
        None,
        3.0,
        ["model.encoder.mlp_ratio=3.0", "model.seg3d_head.mlp_ratio=4"],
        id="mlp",
    ),
    pytest.param(
        [2, 1],
        2.5,
        [
            "model.encoder.enc_num_head=[2,1]",
            "+model.encoder.enc_head_dim=[4,8]",
            "model.encoder.mlp_ratio=2.5",
            "model.seg3d_head.mlp_ratio=4",
        ],
        id="heads+mlp",
    ),
]


@pytest.mark.parametrize(("num_heads", "mlp_ratio", "overrides"), _SHAPES)
def test_sliced_model_equals_the_masked_model(
    num_heads: list[int] | None, mlp_ratio: float | None, overrides: list[str]
) -> None:
    """
    Input: the encoder pruned as a mask to 2 / 1 heads, to a narrower MLP or both, and its
        pruned state dict loaded into an encoder built with those head counts at the trained
        head widths 4 / 8 and that MLP ratio.
    Expected: the two encoders compute the same function; the overrides name that encoder.
    Check: strict load, and outputs equal within 1e-5 on new frames.
    """
    encoder = _trained_encoder()
    pruning = prune_encoder(
        encoder, encoder, _frames(8, seed=1), 0.01, num_heads=num_heads, mlp_ratio=mlp_ratio
    )
    state = slice_pruned_state_dict(encoder.state_dict(), encoder, pruning, prefix="")
    sliced = _Encoder(
        num_heads=num_heads or (4, 2),
        head_dim=(4, 8) if num_heads else (None, None),
        mlp_ratio=mlp_ratio or 4.0,
    ).eval()
    sliced.load_state_dict(state, strict=True)

    assert pruned_model_overrides(pruning, mlp_ratio, seg3d_head_mlp_ratio=4) == overrides
    with torch.no_grad():
        for frame in _frames(4, seed=2):
            torch.testing.assert_close(sliced(frame), encoder(frame), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("num_heads", "mlp_ratio", "layer"),
    [([2, 1], None, "attn.proj"), (None, 2.0, "mlp.0.fc2")],
    ids=["heads", "mlp"],
)
def test_refit_reduces_the_output_error_of_first_units_pruning(
    num_heads: list[int] | None, mlp_ratio: float | None, layer: str
) -> None:
    """
    Input: the encoder pruned to 2 / 1 heads or to half the MLP hidden units by selection plus
        refit, and by keeping the first heads or units without a refit, on random correlated
        frames.
    Expected: the refit is closer to the unpruned encoder, both at the pruned layer's output
        in the first block on the calibration frames and at the encoder output on new frames.
    Check: the summed squared errors.
    """
    reference = _trained_encoder()
    calibration = _frames(16, seed=1)
    refit = copy.deepcopy(reference)
    prune_encoder(refit, refit, calibration, 0.01, num_heads=num_heads, mlp_ratio=mlp_ratio)
    first = copy.deepcopy(reference)
    _first_units_pruning(first, num_heads, mlp_ratio)

    def layer_output(encoder: _Encoder, frame: torch.Tensor) -> torch.Tensor:
        outputs = []
        module = encoder.enc.enc0.block0.get_submodule(layer)
        handle = module.register_forward_hook(
            lambda _module, _inputs, output: outputs.append(output)
        )
        encoder(frame)
        handle.remove()
        return outputs[0]

    def error(encoder: _Encoder, frames: list[torch.Tensor], output) -> float:
        with torch.no_grad():
            return sum(
                float((output(encoder, frame) - output(reference, frame)).square().sum())
                for frame in frames
            )

    assert error(refit, calibration, layer_output) < 0.5 * error(first, calibration, layer_output)
    held_out = _frames(8, seed=2)
    assert error(refit, held_out, _Encoder.__call__) < error(first, held_out, _Encoder.__call__)


def test_hidden_unit_selection_keeps_the_highest_wanda_scores() -> None:
    """
    Input: a layer over 4 hidden units whose column norms and input energies give the scores
        4, 1, 9 and 0.25.
    Expected: keeping 2 units keeps units 0 and 2, ascending.
    Check: the kept units, and the error for an impossible count.
    """
    weight = torch.tensor([[2.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.0]])
    second_moment = torch.diag(torch.tensor([1.0, 1.0, 9.0, 1.0]))

    assert select_hidden_units(weight, second_moment, 2) == [0, 2]
    with pytest.raises(ValueError, match="Cannot keep"):
        select_hidden_units(weight, second_moment, 5)


def test_head_selection_minimises_the_dropped_energy() -> None:
    """
    Input: a random positive semi-definite head Gram of 7 heads, keeping 3.
    Expected: the exhaustive search returns the best drop set; the greedy search with swaps
        finds it too on this small case.
    Check: against a brute force over all drop sets.
    """
    torch.manual_seed(0)
    factor = torch.randn(7, 5, dtype=torch.float64)
    gram = factor @ factor.t()
    best = min(
        itertools.combinations(range(7), 4),
        key=lambda dropped: float(gram[list(dropped)][:, list(dropped)].sum()),
    )
    expected = [head for head in range(7) if head not in best]

    assert select_heads(gram, 3) == expected
    assert select_heads(gram, 3, max_exhaustive_sets=1) == expected
    with pytest.raises(ValueError, match="Cannot keep"):
        select_heads(gram, 0)


def test_head_gram_is_the_attention_output_energy_of_head_pairs() -> None:
    """
    Input: a projection over 3 heads of width 2 and random inputs.
    Expected: ``G_ij`` is the mean inner product of the outputs of heads ``i`` and ``j``.
    Check: against the per-head outputs.
    """
    torch.manual_seed(0)
    weight, inputs = torch.randn(4, 6, dtype=torch.float64), torch.randn(30, 6, dtype=torch.float64)
    gram = head_gram(weight, inputs.t() @ inputs / 30, num_heads=3, head_dim=2)
    outputs = [
        inputs[:, head_columns([head], 2)] @ weight[:, head_columns([head], 2)].t()
        for head in range(3)
    ]
    expected = torch.tensor(
        [[float((outputs[i] * outputs[j]).sum() / 30) for j in range(3)] for i in range(3)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(gram, expected)


def test_calibration_frames_come_from_evenly_spaced_scenes() -> None:
    """
    Input: 10 scenes (one empty) of 5 frames, and requests for 3 and for 20 frames.
    Expected: the first and last non-empty scene and one in between, one frame each, at
        golden-ratio positions; at most one frame per scene.
    Check: the exact indices.
    """
    groups = [list(range(5 * scene, 5 * scene + 5)) for scene in range(10)]
    groups[3] = []

    assert calibration_frame_indices(groups, 3) == [0, 5 * 5 + 3, 45 + 1]
    assert len(calibration_frame_indices(groups, 20)) == 9


def test_calibration_scenes_must_stay_out_of_the_held_out_splits() -> None:
    """
    Input: calibration frames from scenes a and b, against held-out splits without them and
        with scene b in the test split.
    Expected: the first passes; the second is refused, naming the split.
    Check: no error, then a ValueError.
    """
    check_calibration_scenes(["a", "b", "a"], {"val": {"c"}, "test": {"d"}})
    with pytest.raises(ValueError, match="1 calibration scenes are in the test split, e.g. b"):
        check_calibration_scenes(["a", "b"], {"val": {"c"}, "test": {"b", "d"}})


def test_module_runs_hydra_main() -> None:
    """
    Input: ``python -m autoware_ml.scripts.tools.ptv3.oneshot_prune --help``.
    Expected: the module starts its Hydra main.
    Check: exit code 0 and Hydra's help text.
    """
    result = subprocess.run(
        [sys.executable, "-m", "autoware_ml.scripts.tools.ptv3.oneshot_prune", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Powered by Hydra" in result.stdout
