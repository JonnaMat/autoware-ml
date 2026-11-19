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

"""The ``enc_norm`` option of the PTv3 encoder: LayerNorm (default) or a foldable BatchNorm."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_ptv3_encoder,
    build_seg3d_model,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointBatchNorm,
    PointTransformerV3Encoder,
    ProgressiveLayerToBatchNorm,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

NORM_PATHS = ("cpe.2", "norm0.0", "norm1.0", "norm2.0")


def _build_block(norm: str, *, enable_conv: bool = True, qkv_bias: bool = True) -> Block:
    return Block(
        channels=16,
        num_heads=2,
        patch_size=4,
        mlp_ratio=2.0,
        qkv_bias=qkv_bias,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        order_index=0,
        cpe_indice_key="stage0",
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enable_conv=enable_conv,
        norm=norm,
    )


def _block_norms(encoder: PointTransformerV3Encoder) -> list[nn.Module]:
    norms = []
    for block in encoder.modules():
        if isinstance(block, Block):
            modules = dict(block.named_modules())
            norms.extend(modules[path] for path in NORM_PATHS if path in modules)
    return norms


def _jitter_batch_norm(norm: PointBatchNorm) -> None:
    nn.init.uniform_(norm.weight, 0.5, 1.5)
    nn.init.uniform_(norm.bias, -0.5, 0.5)
    nn.init.uniform_(norm.running_mean, -2.0, 2.0)
    nn.init.uniform_(norm.running_var, 0.1, 3.0)


def test_default_encoder_keeps_layer_norm_and_its_parameters() -> None:
    """
    Input: the test encoder built with the default arguments and with ``enc_norm="ln"``, from
    the same seed.
    Expected: the default is LayerNorm, so the option changes nothing unless set.
    Check: every block norm (cpe.2, norm1, norm2) is exactly ``nn.LayerNorm``; both state dicts
    have the same keys and equal tensors.
    """
    torch.manual_seed(0)
    default = build_ptv3_encoder()
    torch.manual_seed(0)
    explicit = build_ptv3_encoder(enc_norm="ln")
    norms = _block_norms(default)
    assert len(norms) == 3 * 2
    assert all(type(norm) is nn.LayerNorm for norm in norms)
    default_state, explicit_state = default.state_dict(), explicit.state_dict()
    assert list(default_state) == list(explicit_state)
    for key, value in default_state.items():
        assert torch.equal(value, explicit_state[key]), key


def test_batch_norm_encoder_only_adds_running_statistics() -> None:
    """
    Input: the test encoder with ``enc_norm="ln"`` and ``enc_norm="bn"``.
    Expected: the BatchNorm replaces every block LayerNorm under the same name, so a LayerNorm
    checkpoint provides its weight and bias.
    Check: every block norm is a PointBatchNorm; the shared keys have equal shapes and the extra
    keys are only running_mean / running_var / num_batches_tracked of those norms.
    """
    layer_norm = build_ptv3_encoder(enc_norm="ln").state_dict()
    batch_norm_encoder = build_ptv3_encoder(enc_norm="bn")
    batch_norm = batch_norm_encoder.state_dict()
    assert all(isinstance(norm, PointBatchNorm) for norm in _block_norms(batch_norm_encoder))
    assert set(layer_norm) <= set(batch_norm)
    for key, value in layer_norm.items():
        assert batch_norm[key].shape == value.shape, key
    extra = set(batch_norm) - set(layer_norm)
    assert len(extra) == 3 * 3 * 2
    assert {key.rsplit(".", 1)[1] for key in extra} == {
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }
    assert {key.rsplit(".", 1)[0] + ".weight" for key in extra} <= set(layer_norm)


def test_unknown_block_norm_is_rejected() -> None:
    """
    Input: a block asked for an unknown normalization.
    Expected: a configuration error instead of a silent LayerNorm.
    Check: ValueError naming the norm.
    """
    with pytest.raises(ValueError, match="rms"):
        _build_block("rms")


def test_point_batch_norm_is_an_affine_in_eval_and_uses_batch_statistics_in_training() -> None:
    """
    Input: a PointBatchNorm with non-trivial running statistics and affine parameters, applied
    to 500 points in eval and in training mode.
    Expected: eval is the per-channel affine ``scale * x + shift`` of ``eval_affine``; training
    normalizes each channel over all points.
    Check: eval output matches the affine within 1e-5; the training output divided by the weight
    after removing the bias has per-channel mean 0 and variance 1.
    """
    torch.manual_seed(0)
    norm = PointBatchNorm(16, eps=1e-5, momentum=0.01)
    _jitter_batch_norm(norm)
    x = torch.randn(500, 16) * 3.0 + 1.0
    norm.eval()
    scale, shift = norm.eval_affine()
    torch.testing.assert_close(norm(x), (scale * x.double() + shift).float(), rtol=1e-5, atol=1e-5)
    norm.train()
    standardized = (norm(x) - norm.bias) / norm.weight
    torch.testing.assert_close(standardized.mean(0), torch.zeros(16), rtol=0, atol=1e-5)
    torch.testing.assert_close(
        standardized.var(0, unbiased=False), torch.ones(16), rtol=0, atol=1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast on CUDA")
def test_point_batch_norm_runs_in_fp32_under_autocast() -> None:
    """
    Input: a PointBatchNorm on CUDA applied to bf16 features under bf16 autocast.
    Expected: the statistics are taken in fp32, as LayerNorm does under autocast.
    Check: the output is fp32 and equals the fp32 layer applied to the upcast input.
    """
    torch.manual_seed(0)
    norm = PointBatchNorm(16).cuda().eval()
    _jitter_batch_norm(norm)
    x = torch.randn(100, 16, device="cuda").bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = norm(x)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, norm(x.float()))


@pytest.mark.parametrize(("enable_conv", "qkv_bias"), [(True, True), (True, False), (False, True)])
def test_fold_batch_norms_preserves_every_folded_path(enable_conv: bool, qkv_bias: bool) -> None:
    """
    Input: a BatchNorm block (with and without CPE, with and without a qkv bias) whose norms
    carry non-trivial statistics, folded in eval mode.
    Expected: the fold moves each norm into its neighbouring linear layer: cpe.2 into cpe.1,
    norm1 into attn.qkv, norm2 into mlp.fc1; norm0 (no CPE) stays a BatchNorm.
    Check: ``cpe.1 -> cpe.2``, ``norm1 -> qkv`` and ``norm2 -> fc1`` of the original equal the
    folded linear layers alone within 1e-5 on random points; the folded norms are Identity.
    """
    torch.manual_seed(0)
    block = _build_block("bn", enable_conv=enable_conv, qkv_bias=qkv_bias)
    for module in block.modules():
        if isinstance(module, PointBatchNorm):
            _jitter_batch_norm(module)
    block.eval()
    folded = copy.deepcopy(block)
    folded.fold_batch_norms()
    x = torch.randn(200, 16) * 2.0 + 0.5
    with torch.no_grad():
        if enable_conv:
            reference = block.cpe._modules["2"](block.cpe._modules["1"](x))
            torch.testing.assert_close(folded.cpe._modules["1"](x), reference, rtol=1e-5, atol=1e-5)
            assert isinstance(folded.cpe._modules["2"], nn.Identity)
        else:
            assert isinstance(folded.norm0._modules["0"], PointBatchNorm)
        reference = block.attn.qkv(block.norm1._modules["0"](x))
        torch.testing.assert_close(folded.attn.qkv(x), reference, rtol=1e-5, atol=1e-5)
        fc1 = block.mlp._modules["0"].fc1
        reference = fc1(block.norm2._modules["0"](x))
        torch.testing.assert_close(folded.mlp._modules["0"].fc1(x), reference, rtol=1e-5, atol=1e-5)
    assert isinstance(folded.norm1._modules["0"], nn.Identity)
    assert isinstance(folded.norm2._modules["0"], nn.Identity)


def test_fold_batch_norms_leaves_layer_norm_blocks_untouched() -> None:
    """
    Input: a default LayerNorm block, folded.
    Expected: only BatchNorm blocks fold.
    Check: the state dict is unchanged and the norms are still LayerNorm.
    """
    block = _build_block("ln").eval()
    state = {key: value.clone() for key, value in block.state_dict().items()}
    block.fold_batch_norms()
    for key, value in block.state_dict().items():
        assert torch.equal(value, state[key]), key
    assert type(block.norm1._modules["0"]) is nn.LayerNorm


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_export_folds_the_block_batch_norms(monkeypatch) -> None:
    """
    Input: a small PTv3 segmentation model with ``enc_norm="bn"`` and non-trivial statistics,
    exported with and without the BatchNorm fold.
    Expected: the export encoder holds no BatchNorm in its blocks and computes what the
    unfolded eval model computes.
    Check: encoder export outputs match within 1e-5; no PointBatchNorm is left in the folded
    encoder; the live model's state_dict is unchanged.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = build_seg3d_model(enc_norm="bn")
    for module in model.encoder.modules():
        if isinstance(module, PointBatchNorm):
            _jitter_batch_norm(module)
    model = model.to(device).eval()
    batch = build_inputs(device=device)
    state = {key: value.clone() for key, value in model.state_dict().items()}

    folded = model.build_export_specs(batch)
    monkeypatch.setattr(Block, "fold_batch_norms", lambda block: None)
    reference = model.build_export_specs(batch)

    with torch.no_grad():
        folded_outputs = folded["ptv3_encoder"].module(*folded["ptv3_encoder"].args)
        reference_outputs = reference["ptv3_encoder"].module(*reference["ptv3_encoder"].args)
    for folded_output, reference_output in zip(folded_outputs, reference_outputs, strict=True):
        if folded_output.is_floating_point():
            torch.testing.assert_close(folded_output, reference_output, rtol=1e-5, atol=1e-5)
    encoder_modules = list(folded["ptv3_encoder"].module.modules())
    assert any(isinstance(module, Block) for module in encoder_modules)
    assert not any(isinstance(module, PointBatchNorm) for module in encoder_modules)
    assert any(
        isinstance(module, PointBatchNorm) for module in reference["ptv3_encoder"].module.modules()
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key]), key


def test_progressive_norm_starts_as_layer_norm_and_ends_as_batch_norm() -> None:
    """
    Input: a ProgressiveLayerToBatchNorm with decay_steps 4, non-trivial LN and BN parameters,
    run in eval before training, then for 2 and 4 training forward passes.
    Expected: lam = 1 - step / 4: the layer is the LayerNorm at step 0, a blend in between and the
    BatchNorm alone from step 4 on, in training and in eval.
    Check: eval output at step 0 equals the LN; the training output at step 2 equals
    0.5 LN + 0.5 BN(batch stats); from step 4 the train and eval outputs equal the BN's.
    """
    torch.manual_seed(0)
    norm = ProgressiveLayerToBatchNorm(16, decay_steps=4)
    nn.init.uniform_(norm.ln.weight, 0.5, 1.5)
    nn.init.uniform_(norm.ln.bias, -0.5, 0.5)
    _jitter_batch_norm(norm.bn)
    x = torch.randn(300, 16) * 2.0 + 0.5
    norm.eval()
    torch.testing.assert_close(norm(x), norm.ln(x))
    norm.train()
    norm(x)
    bn_train = copy.deepcopy(norm.bn)
    blended = norm(x)
    assert int(norm.step) == 2
    torch.testing.assert_close(blended, 0.5 * norm.ln(x) + 0.5 * bn_train(x), rtol=1e-5, atol=1e-5)
    norm(x)
    bn_train = copy.deepcopy(norm.bn)
    torch.testing.assert_close(norm(x), bn_train(x), rtol=1e-5, atol=1e-5)
    norm.eval()
    torch.testing.assert_close(norm(x), norm.bn(x))


def test_progressive_norm_step_survives_a_state_dict_round_trip() -> None:
    """
    Input: a progressive norm trained 3 steps, its state loaded into a fresh one that already ran a
    forward pass.
    Expected: the step counter is checkpointed and the host-side mirror re-reads it.
    Check: the state dict holds ln.*, bn.* and step; the loaded layer reports step 3 and its next
    training pass gives lam = 1 - 4 / 8.
    """
    source = ProgressiveLayerToBatchNorm(8, decay_steps=8).train()
    x = torch.randn(50, 8)
    for _ in range(3):
        source(x)
    state = source.state_dict()
    assert {"ln.weight", "ln.bias", "bn.weight", "bn.running_mean", "step"} <= set(state)
    target = ProgressiveLayerToBatchNorm(8, decay_steps=8).eval()
    target(x)
    target.load_state_dict(state)
    target.train()
    target(x)
    assert int(target.step) == 4
    assert float(target.ln_weight()) == pytest.approx(0.5)


def test_progressive_blocks_fold_only_after_the_blend() -> None:
    """
    Input: a block with norm="ln2bn" (decay 2) in eval, folded before and after 2 training passes.
    Expected: only the pure BatchNorm folds; a part-LayerNorm blend must not be exported as an
    affine.
    Check: ValueError before the blend ends; afterwards cpe.1 / qkv / fc1 reproduce the unfolded
    paths within 1e-5 and the norms are Identity.
    """
    torch.manual_seed(0)
    block = Block(
        channels=16,
        num_heads=2,
        patch_size=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        order_index=0,
        cpe_indice_key="stage0",
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        norm="ln2bn",
        norm_decay_steps=2,
    ).eval()
    with pytest.raises(ValueError, match="blend"):
        copy.deepcopy(block).fold_batch_norms()
    x = torch.randn(200, 16) * 2.0 + 0.5
    norms = [m for m in block.modules() if isinstance(m, ProgressiveLayerToBatchNorm)]
    assert len(norms) == 3
    for norm in norms:
        _jitter_batch_norm(norm.bn)
        norm.train()
        norm(x)
        norm(x)
        norm.eval()
    folded = copy.deepcopy(block)
    folded.fold_batch_norms()
    with torch.no_grad():
        reference = block.cpe._modules["2"](block.cpe._modules["1"](x))
        torch.testing.assert_close(folded.cpe._modules["1"](x), reference, rtol=1e-5, atol=1e-5)
        reference = block.attn.qkv(block.norm1._modules["0"](x))
        torch.testing.assert_close(folded.attn.qkv(x), reference, rtol=1e-5, atol=1e-5)
        fc1 = block.mlp._modules["0"].fc1
        reference = fc1(block.norm2._modules["0"](x))
        torch.testing.assert_close(folded.mlp._modules["0"].fc1(x), reference, rtol=1e-5, atol=1e-5)
    assert isinstance(folded.norm1._modules["0"], nn.Identity)


def test_progressive_encoder_requires_a_decay_length() -> None:
    """
    Input: the test encoder with enc_norm="ln2bn" without and with enc_norm_decay_steps.
    Expected: the blend length must be given explicitly.
    Check: ValueError without it; with it every block norm is a ProgressiveLayerToBatchNorm.
    """
    with pytest.raises(ValueError, match="enc_norm_decay_steps"):
        build_ptv3_encoder(enc_norm="ln2bn")
    encoder = PointTransformerV3Encoder(
        **{**_encoder_kwargs(), "enc_norm": "ln2bn", "enc_norm_decay_steps": 10}
    )
    norms = _block_norms(encoder)
    assert norms and all(isinstance(norm, ProgressiveLayerToBatchNorm) for norm in norms)


def _encoder_kwargs() -> dict:
    return dict(
        in_channels=4,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(1, 2),
        enc_patch_size=(4, 4),
        mlp_ratio=2.0,
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
    )
