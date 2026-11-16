"""Unit tests for slicing a PTv3 checkpoint onto a pruned encoder."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    MLP,
    Point,
    PointTransformerV3Encoder,
    SerializedAttention,
)
from autoware_ml.scripts.tools.ptv3.prune_checkpoint import (
    main,
    prune_ptv3_checkpoint,
    prune_ptv3_state_dict,
)


def _attention(channels: int, num_heads: int, head_dim: int | None = None) -> SerializedAttention:
    return SerializedAttention(
        channels=channels,
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
    ).eval()


def _point(feat: torch.Tensor) -> Point:
    count = feat.shape[0]
    return Point(
        {
            "feat": feat.clone(),
            "grid_coord": torch.zeros(count, 3, dtype=torch.int32),
            "serialized_order": torch.randperm(count).reshape(1, count),
            "serialized_inverse": torch.arange(count).reshape(1, count),
            "offset": torch.tensor([count], dtype=torch.long),
        }
    )


def _encoder(depths: tuple[int, ...], heads: tuple[int, ...], mlp_ratio: float, head_dim=None):
    return PointTransformerV3Encoder(
        in_channels=4,
        order=("z", "z-trans"),
        stride=(2,),
        enc_depths=depths,
        enc_channels=(16, 32),
        enc_num_head=heads,
        enc_patch_size=(8, 8),
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enc_head_dim=head_dim,
    )


def test_default_head_dim_keeps_the_ptv3_attention() -> None:
    torch.manual_seed(0)
    default = _attention(32, 4)
    explicit = _attention(32, 4, head_dim=8)
    assert default.qkv.weight.shape == (96, 32) and default.proj.weight.shape == (32, 32)
    assert default.scale == explicit.scale == 8**-0.5
    explicit.load_state_dict(default.state_dict(), strict=True)
    feat = torch.randn(8, 32)
    torch.manual_seed(1)
    expected = default(_point(feat)).feat
    torch.manual_seed(1)
    torch.testing.assert_close(explicit(_point(feat)).feat, expected, rtol=0, atol=0)


def test_head_dim_decouples_the_attention_width_from_the_channels() -> None:
    attention = _attention(32, 2, head_dim=8)
    assert attention.qkv.weight.shape == (48, 32)
    assert attention.proj.weight.shape == (32, 16)
    assert attention(_point(torch.randn(8, 32))).feat.shape == (8, 32)


def test_head_slice_computes_the_original_with_the_dropped_heads_silenced() -> None:
    torch.manual_seed(0)
    original = _attention(32, 4)
    state = {f"encoder.enc.enc0.block0.attn.{k}": v for k, v in original.state_dict().items()}
    sliced = prune_ptv3_state_dict(state, num_heads=[2], head_dim=8)
    pruned = _attention(32, 2, head_dim=8)
    pruned.load_state_dict({k.rsplit("attn.", 1)[1]: v for k, v in sliced.items()}, strict=True)
    with torch.no_grad():
        original.proj.weight[:, 16:] = 0  # heads 2 and 3 no longer reach the output
    feat = torch.randn(8, 32)
    torch.manual_seed(1)
    expected = original(_point(feat)).feat
    torch.manual_seed(1)
    torch.testing.assert_close(pruned(_point(feat)).feat, expected)


def test_mlp_slice_computes_the_original_with_the_dropped_units_silenced() -> None:
    torch.manual_seed(0)
    original = MLP(16, 4, 0.0).eval()
    state = {f"encoder.enc.enc0.block0.mlp.0.{k}": v for k, v in original.state_dict().items()}
    sliced = prune_ptv3_state_dict(state, mlp_ratio=2)
    pruned = MLP(16, 2, 0.0).eval()
    pruned.load_state_dict({k.rsplit("mlp.0.", 1)[1]: v for k, v in sliced.items()}, strict=True)
    with torch.no_grad():
        original.fc2.weight[:, 32:] = 0
    feat = torch.randn(5, 16)
    torch.testing.assert_close(pruned(feat), original(feat))


def test_pruned_checkpoint_loads_strictly_and_keeps_other_tensors(tmp_path: Path) -> None:
    pytest.importorskip("spconv")
    torch.manual_seed(0)
    original = _encoder((2, 3), (2, 4), 4.0)
    state = {f"encoder.{k}": v for k, v in original.state_dict().items()}
    state["seg3d_head.weight"] = torch.randn(3)
    checkpoint = {"state_dict": state, "optimizer_states": [{}], "epoch": 7}
    source = tmp_path / "in.ckpt"
    target = tmp_path / "out.ckpt"
    torch.save(checkpoint, source)

    main([str(source), str(target), "--depths", "1", "2", "--mlp-ratio", "2",
          "--num-heads", "1", "2", "--head-dim", "8"])  # fmt: skip

    written = torch.load(target, weights_only=False)
    assert "optimizer_states" not in written and written["epoch"] == 7
    assert written["autoware_ml_prune"]["depths"] == [1, 2]
    pruned = _encoder((1, 2), (1, 2), 2.0, head_dim=8)
    pruned.load_state_dict(
        {
            k.removeprefix("encoder."): v
            for k, v in written["state_dict"].items()
            if k.startswith("encoder.")
        },
        strict=True,
    )
    assert torch.equal(written["state_dict"]["seg3d_head.weight"], state["seg3d_head.weight"])
    kept = "encoder.enc.enc1.block1.cpe.1.weight"
    assert torch.equal(written["state_dict"][kept], state[kept])


def test_unpruned_targets_copy_the_checkpoint() -> None:
    state = {"encoder.enc.enc0.block0.attn.qkv.weight": torch.randn(48, 16)}
    pruned = prune_ptv3_checkpoint({"state_dict": state})["state_dict"]
    assert torch.equal(pruned["encoder.enc.enc0.block0.attn.qkv.weight"], state[next(iter(state))])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"depths": [3]}, "cannot keep 3"),
        ({"num_heads": [1]}, "head_dim is required"),
        ({"num_heads": [4], "head_dim": 8}, "at least 4 heads"),
        ({"mlp_ratio": 8}, "cannot slice the MLP"),
    ],
)
def test_prune_rejects_targets_the_checkpoint_cannot_provide(kwargs, match) -> None:
    state = {
        "encoder.enc.enc0.block0.attn.qkv.weight": torch.randn(48, 16),
        "encoder.enc.enc0.block0.mlp.0.fc1.weight": torch.randn(64, 16),
        "encoder.enc.enc0.block1.attn.qkv.weight": torch.randn(48, 16),
    }
    with pytest.raises(ValueError, match=match):
        prune_ptv3_state_dict(state, **kwargs)
