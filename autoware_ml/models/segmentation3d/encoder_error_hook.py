"""Test-time error injection at the PTv3 encoder outputs (RDDR-1486 error-rule calibration).

EXPERIMENT CODE, not for upstream. Off unless one of the env flags is set; with no flag set
:func:`apply` returns the point unchanged and :func:`install` changes nothing.

Env flags
  PTV3_ERR_NOISE_K       float k > 0: add Gaussian noise to every encoder stage output point_feat_s,
                         sigma_s = k * m_s / sqrt(2/pi), so E|noise_s| = k * m_s, where m_s is the
                         per-output mean_abs of the base fp16 TensorRT engine vs the fp32 PyTorch
                         reference (level 5; default: EXP-112 prs, mean of 3 builds).
  PTV3_ERR_MEAN_ABS      5 comma-separated m_s overriding the default.
  PTV3_ERR_SEED          int (default 0). The noise of call i (one encoder forward = one batch) is
                         drawn from a generator seeded with seed * 1_000_003 + i: same batches, same noise.
  PTV3_ERR_LN            "ln_scale": every encoder Block's norm1 LayerNorm is replaced by an emulation
                         of the ln_scale ONNX rewrite (tools/rewrite_equiv.py p_ln_scale) as a weakly
                         typed fp16 TensorRT build runs it: the row statistics mean, centred, square,
                         var, +eps, sqrt, 1/x and the scaled rows r1 * rstd are each rounded to fp16
                         (reductions accumulate in fp32); the centring and gamma/beta, folded into the
                         qkv weights by the rewrite, are applied exactly in fp32 to the fp16 rows.
                         Everything else stays fp32: the arm isolates the rewrite's own error.
  PTV3_ERR_LOG           path: one JSON line per call with the injected per-output mean/max |noise|,
                         the per-output mean |feat| / rms, and the voxel counts.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
from torch import nn

# EXP-112 prs (base = main + PR #293 + #295 export, thor2), fp32ref per-output mean_abs, mean of 3 builds.
BASE_MEAN_ABS = (0.0017218, 0.0028871, 0.0043473, 0.0042391, 0.0044379)

_state: dict[str, Any] = {"calls": 0, "installed": False}


def _k() -> float:
    return float(os.environ.get("PTV3_ERR_NOISE_K", "0") or 0)


def _mean_abs() -> tuple[float, ...]:
    s = os.environ.get("PTV3_ERR_MEAN_ABS")
    return tuple(float(x) for x in s.split(",")) if s else BASE_MEAN_ABS


def active() -> bool:
    return _k() > 0 or bool(os.environ.get("PTV3_ERR_LN"))


class LnScaleFp16(nn.Module):
    """norm1 as the ln_scale rewrite computes it in a weakly typed fp16 engine (see module doc)."""

    def __init__(self, ln: nn.LayerNorm) -> None:
        super().__init__()
        assert len(ln.normalized_shape) == 1, ln.normalized_shape
        self.ln = ln

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.float16
        x16 = x.to(h)
        mu = x16.float().mean(-1, keepdim=True).to(h)
        c = (x16.float() - mu.float()).to(h)
        sq = (c.float() * c.float()).to(h)
        var = sq.float().mean(-1, keepdim=True).to(h)
        ve = (var.float() + torch.tensor(self.ln.eps, dtype=h).float()).to(h)
        std = ve.float().sqrt().to(h)
        rstd = (1.0 / std.float()).to(h)
        y = (x16.float() * rstd.float()).to(h).to(x.dtype)
        y = y - y.mean(-1, keepdim=True)  # (I - 11^T/C) folded into W'
        w = self.ln.weight if self.ln.weight is not None else 1.0
        b = self.ln.bias if self.ln.bias is not None else 0.0
        return y * w + b


def install(encoder: nn.Module) -> int:
    """Swap every encoder Block's norm1 LayerNorm for LnScaleFp16 when PTV3_ERR_LN=ln_scale."""
    mode = os.environ.get("PTV3_ERR_LN")
    if not mode or _state["installed"]:
        return 0
    if mode != "ln_scale":
        raise ValueError(f"PTV3_ERR_LN={mode!r}: only 'ln_scale' is implemented")
    n = 0
    for m in encoder.modules():
        seq = getattr(m, "norm1", None)
        if seq is None or not hasattr(m, "attn"):
            continue
        for name, sub in list(seq._modules.items()):
            if isinstance(sub, nn.LayerNorm):
                seq._modules[name] = LnScaleFp16(sub)
                n += 1
    _state["installed"] = True
    _state["ln_replaced"] = n
    print(f"[encoder_error_hook] PTV3_ERR_LN=ln_scale: replaced {n} norm1 LayerNorms", flush=True)
    return n


def stage_points(deepest: Any) -> list[Any]:
    stages = [deepest]
    while "pooling_parent" in stages[-1]:
        stages.append(stages[-1]["pooling_parent"])
    stages.reverse()
    return stages


def noise_like(feats: list[torch.Tensor], call: int) -> list[torch.Tensor]:
    """Per-stage Gaussian noise for call `call` (deterministic), E|n_s| = k * m_s."""
    k, m = _k(), _mean_abs()
    if len(m) != len(feats):
        raise ValueError(f"{len(feats)} encoder outputs but {len(m)} mean_abs values")
    g = torch.Generator(device=feats[0].device)
    g.manual_seed(int(os.environ.get("PTV3_ERR_SEED", "0")) * 1_000_003 + call)
    c = 1.0 / math.sqrt(2.0 / math.pi)
    return [torch.randn(f.shape, generator=g, device=f.device, dtype=torch.float32).mul_(k * ms * c).to(f.dtype)
            for f, ms in zip(feats, m)]


def apply(point: Any) -> Any:
    """Add the configured noise to every stage output of the encoder (in place on the Point chain)."""
    if _k() <= 0 and not os.environ.get("PTV3_ERR_LOG"):
        return point
    call = _state["calls"]
    _state["calls"] += 1
    stages = stage_points(point)
    rec: dict[str, Any] = {"call": call, "k": _k(), "ln": os.environ.get("PTV3_ERR_LN", ""),
                           "n": [int(s.feat.shape[0]) for s in stages]}
    with torch.no_grad():
        rec["feat_mean_abs"] = [float(s.feat.float().abs().mean()) for s in stages]
        rec["feat_rms"] = [float(s.feat.float().pow(2).mean().sqrt()) for s in stages]
        if _k() > 0:
            ns = noise_like([s.feat for s in stages], call)
            rec["noise_mean_abs"] = [float(n.float().abs().mean()) for n in ns]
            rec["noise_max_abs"] = [float(n.float().abs().max()) for n in ns]
            tot = sum(n.numel() for n in ns)
            rec["noise_all_mean_abs"] = sum(float(n.float().abs().sum()) for n in ns) / tot
            rec["noise_all_max_abs"] = max(rec["noise_max_abs"])
            for s, n in zip(stages, ns):
                s.feat = s.feat + n
                if "sparse_conv_feat" in s.keys():
                    s.sparse_conv_feat = s.sparse_conv_feat.replace_feature(s.feat)
    log = os.environ.get("PTV3_ERR_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return point
