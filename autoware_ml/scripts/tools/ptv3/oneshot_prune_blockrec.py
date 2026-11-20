#!/usr/bin/env python3
"""oneshot_prune.py [all] [--model ptv3|litept] [--shapes ...] [--methods ...]: one-shot STRUCTURED pruning of the v0.2.0 PTv3
encoder with calibration-based selection and closed-form weight reconstruction (no finetuning).

Runs inside Docker autoware-ml:ptv3-export through tools/oneshot_prune.sh, with the autoware-ml
worktree ~/src/autoware-ml-wt/oneshot-prune (local branch exp/RDDR-1486-oneshot-prune = main 3a5ba6c
+ the enc_head_dim encoder option of 7601054) at /workspace. Idea #1 of
notes/llm-compressor-survey-2026-11-19.md (rows 1+2, with 10/18 inside).

Shapes (uniform, the EXP-105 / work/prune-sweep shapes whose latency is measured):
  heads_r50  enc_num_head 2/4/8/16/32 -> 1/2/4/8/16 at head width 16 (+enc_head_dim=16)
  mlp_r25    MLP hidden 4C -> 3C   (mlp_ratio=3)
  mlp_r50    MLP hidden 4C -> 2C   (mlp_ratio=2)
  h2m2       heads_r50 + mlp_r50 combined (EXP-105 mlp_r50_heads_r50; --shapes h2m2, not in the default set);
             per block the attention is pruned first, then the MLP with that attention already pruned
  heads_r75  enc_num_head -> 1/1/2/4/8 (stage 0 floored at 1 head); heads_r50_mlp_r25 / heads_r50_mlp_r50
             COMBINED like h2m2 (heads_r50_mlp_r50 = h2m2); heads_r75_mlp_r50 (h4m2, EXP-150) likewise;
             none of these in the default set
Methods per shape:
  firstk  keep the first k heads / hidden units (= prune_checkpoint.py's slicer; the control)
  score   keep by a calibration score: heads by the exact head Gram of the attention output
          (min p^T G p over the drop set), MLP units by Wanda |W|.||x|| aggregated per unit
  refit   score's selection, then ridge LS refit (float64) of the following projection (attn.proj /
          mlp fc2) on the kept inputs plus the dropped units' mean contribution moved into the bias
score and refit calibrate block by block in order with all upstream blocks already pruned (llm-compressor
sequential pipeline): statistics of block i are taken on the model whose blocks < i are pruned by the
same method.
  blockrec      (2026-11-19, review 7 E6(b); not in the default methods) refit, then block-wise
                self-reconstruction toward the UNPRUNED model: for block i = 0..9 in forward order, with the
                input of block i taken from the PRUNED stream (blocks < i already reconstructed) and the
                target from the UNPRUNED stream (asymmetric, corrects upstream error; AdaRound/BRECQ-style):
                (1) attn.proj ridge LS (float64, closed form) on the kept head columns so that
                    shortcut_p + proj(o_p) matches the unpruned post-attention residual h_u;
                (2) mlp fc2 ridge LS (closed form) on the kept hidden units so that the block output
                    matches the unpruned block output y_u (fc2 statistics after step 1).
                Both regularised toward the current weights with --damp (as refit). Label-free.
  blockrecadam  blockrec, then --blockrec-steps Adam steps (fp32, lr --blockrec-lr) per block on the
                block's qkv / proj / fc1 / fc2 / norm1 / norm2 (cpe and everything outside the block frozen)
                minimising ||block(x_p) - y_u||^2 / ||y_u||^2, one calibration frame per step; dropped proj /
                fc2 columns are kept at 0 (gradient mask + re-zero). Frames calib[::16] are held out from
                the steps; if their block error does not improve, the block's Adam result is reverted.
                Every statistic and step streams full frames (no point subsampling).
  blockrecfar   (2026-11-19, EXP-156) blockrec with every token's term in the two LS fits weighted by its
                horizontal distance from the ego (--range-weights, default 50:4,90:8: weight 1 below 50 m,
                4 in 50-90 m, 8 beyond 90 m; distance from the token's grid_coord and its stage's
                sparse_shape and point_cloud_range). Same frames; selection and the per-layer refit stay
                unweighted; no Adam. Aims the reconstruction at the far-range bins of the range table.
With --calib-set fy26h1|mix every blockrec* method also records per-block errors per range bin (<50, 50-90,
>=90 m) on calib[::16], and every arm records the hold-out proxy per range bin (holdout_<set>.range_bins,
pooled rel L2 over all hold-out tokens of the bin, per stage). Records only: the weights do not change.

--model litept (2026-11-20): LitePT v0.3.0 instead of PTv3; only the attention + MLP blocks (encoder stages 3-4)
are pruned, the conv-only stages 0-2 and the decoder stay as trained (see LITEPT below). Calibration must be
--calib-set fy26h1 --holdout-sets fy26h1; point_feat_0..2 of the proxy are then exactly the reference's.

The pruned network is computed on the full-size model as a MASK (dropped head / hidden-unit columns
of proj / fc2 set to 0, which is exactly the removal), then the kept units are permuted to the front of
qkv / proj / fc1 / fc2 and the prefix slicer copied from prune_checkpoint.py (7601054) cuts the
checkpoint. Outputs per arm in OUT/<shape>_<method>/: pruned.ckpt (weights-only Lightning payload with
an autoware_ml_prune record), overrides.txt (Hydra overrides per config), md5.txt, selection.json.
The proxy builds every pruned model from its checkpoint through Hydra with the overrides (strict
load: enforce_full_coverage + load_state_dict(strict=True)), runs the export encoder graph in fp32
(TF32 off) on held-out TRAIN frames and reports relative L2 of point_feat_0..4 vs the unpruned model,
and the max deviation of the sliced model from the masked model (surgery check).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
import spconv.constants as spconv_constants  # noqa: E402

spconv_constants.SPCONV_ALLOW_TF32 = False

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from autoware_ml.builders.database_builder import build_datamodule  # noqa: E402
from autoware_ml.builders.model_builder import build_data_preprocessor, build_model  # noqa: E402
from autoware_ml.models.segmentation3d.ptv3_base import (  # noqa: E402
    ENCODER_EXPORT_POOLING_FIELDS,
    _PTv3EncoderExportModule,
    prepare_ptv3_export_inputs,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import collect_encoder_stage_points  # noqa: E402
from autoware_ml.types.dataset import SplitType  # noqa: E402
from autoware_ml.utils.cli.helpers import resolve_config_reference  # noqa: E402
from autoware_ml.utils.point_cloud.structures import Point  # noqa: E402

# ONESHOT_CKPT / ONESHOT_OUT: other hosts (gw: v0.2.0 under /pool/training_result, same md5)
CKPT = os.environ.get("ONESHOT_CKPT", "/home/jonna-matthiesen/src/autoware-ml/checkpoints/ptv3_v0.2.0/checkpoints/best.ckpt")
# ONESHOT_CKPT_MD5: another source checkpoint (2026-11-19: PTv3 v0.3.0 best.ckpt b200b937..., same
# architecture as v0.2.0); the proxy reference is then that checkpoint's unpruned model.
CKPT_MD5 = os.environ.get("ONESHOT_CKPT_MD5", "ea2d37164831ecffdf47206bdee401ce")
OUT = os.environ.get("ONESHOT_OUT", "/home/jonna-matthiesen/research/ptv3-thor-inf-opt/work/oneshot-prune")
CONFIG = "multi/ptv3/t4dataset/voxel012_b8_24e_t4dataset_122m_pseudo_10hz_500k"
BASE_OVERRIDES = [
    "logger=null",
    "database=t4dataset/t4dataset_j6gen2_pseudo",
    "~datamodule.sources.jpntaxi",
    "~datamodule.sources.joint",
    "~datamodule.sources.mannequin",
]
# Calibration frames come from the TRAIN split of that datamodule, read with the TEST transforms
# (no augmentation): the train source's dataset config is swapped for the test one.
CALIB_OVERRIDES = ["datamodule.sources.main.train_dataset=${datasets.test}"]
# --calib-set / --holdout-sets (2026-11-19, notes/oneshot-followup-2026-11-19.md "FY26H1-calibrated refit"):
# the frames may also come from the FY26H1 Kognic TRAIN split (database t4dataset_j6gen2_det3d_seg3d_fy26h1,
# scenario list db_j6gen2_det3d_seg3d_v1 md5 ccd34e04, 316/40/38 scenes), read through the Kognic finetune
# leaf with the TEST transforms (its single source `main`; the transforms equal the pseudo set's except
# for the data root). The MODEL is always built from CONFIG. Default (pseudo) = the original behaviour.
#   pseudo  256 calib + 20 hold-out from pseudo_large TRAIN (the original set)
#   fy26h1  256 calib + 20 hold-out from FY26H1 TRAIN, same selection rule
#   mix     calib = pseudo[0::2] + fy26h1[0::2] of those two 256-frame sets (128 + 128)
FY26_CONFIG = "multi/ptv3/t4dataset/kognic_ft_v020_t4dataset_122m_j6gen2_det3d_seg3d"
FY26_OVERRIDES = ["logger=null", "datamodule.sources.main.train_dataset=${datasets.test}"]
CALIB_SETS = ("pseudo", "fy26h1", "mix")
HOLDOUT_SETS = ("pseudo", "fy26h1")
HEAD_DIM = 16
TRAINED_HEADS = [2, 4, 8, 16, 32]
TRAINED_MLP_RATIO = 4

SHAPES: dict[str, dict[str, Any]] = {
    "heads_r50": {"num_heads": [1, 2, 4, 8, 16], "mlp_ratio": None},
    "mlp_r25": {"num_heads": None, "mlp_ratio": 3},
    "mlp_r50": {"num_heads": None, "mlp_ratio": 2},
    # combined heads -50 % + MLP -50 % (EXP-105 mlp_r50_heads_r50); not in the default run
    "h2m2": {"num_heads": [1, 2, 4, 8, 16], "mlp_ratio": 2},
    # 2026-11-19 follow-up (notes/oneshot-followup-2026-11-19.md), not in the default run:
    # heads -75 % (2/4/8/16/32 -> 1/1/2/4/8); stage 0 cannot go below 1 head, so it stays at -50 %
    "heads_r75": {"num_heads": [1, 1, 2, 4, 8], "mlp_ratio": None},
    # COMBINED arms (attention then MLP per block, sequential calibration as h2m2)
    "heads_r50_mlp_r25": {"num_heads": [1, 2, 4, 8, 16], "mlp_ratio": 3},
    "heads_r50_mlp_r50": {"num_heads": [1, 2, 4, 8, 16], "mlp_ratio": 2},  # = h2m2 under its arm name
    # 2026-11-19 (EXP-150): heads -75 % + MLP -50 % combined (heads4 + mlp2 = "h4m2"), as heads_r50_mlp_r50
    "heads_r75_mlp_r50": {"num_heads": [1, 1, 2, 4, 8], "mlp_ratio": 2},
    "h4m2": {"num_heads": [1, 1, 2, 4, 8], "mlp_ratio": 2},  # = heads_r75_mlp_r50 under its arm name (EXP-150)
}
DEFAULT_SHAPES = ("heads_r50", "mlp_r25", "mlp_r50")
# Blocks that carry attention + MLP, per encoder stage (PTv3: every block of every stage).
ATTN_STAGES = (0, 1, 2, 3, 4)
ENC_DEPTHS = (2, 2, 2, 2, 2)
PRUNE_TYPE = "ptv3_encoder_oneshot"

# --model litept (2026-11-20, EXP-161 prep): LitePT v0.3.0 (max_schmeller run 7fd7c74d, best.ckpt f2ae8808).
# Encoder stages 0-2 conv-only (cpe, no attention/MLP: untouched), stages 3-4 attention + MLP with 16 / 32 heads
# of width 16, mlp_ratio 4, qk_norm, RoPE base 100, depths 2. QK-norm is nn.LayerNorm(head_dim) SHARED by all
# heads of a block (weight/bias shape (16,)), so a head cut leaves q_norm/k_norm unchanged, as RoPE (per head,
# parameter-free, same head_dim). Shapes keep the PTv3 names but act on stages 3-4 only; enc_num_head of the
# conv-only stages stays at the trained 2/4/8 (no attention is built there). Needs a tree whose LitePTEncoder
# takes enc_head_dim (gw autoware-ml-litept-prune d7ecc34). The decoder (seg3d head, dec_depths 0,0,1,1, heads
# 8/16) is never pruned: --decoder is rejected (PTv3Seg3DDecoderHead has no head-width option and the tool's
# calibration path is the export encoder only).
LITEPT = {
    "CONFIG": "multi/litept/t4dataset/voxel012_b8_100e_t4dataset_122m",
    # the LitePT leaf has a single source on t4dataset_base; the model is built from it, never its data
    "BASE_OVERRIDES": ["logger=null"],
    "TRAINED_HEADS": [2, 4, 8, 16, 32],
    "ATTN_STAGES": (3, 4),
    "SHAPES": {
        "heads_r50": {"num_heads": [2, 4, 8, 8, 16], "mlp_ratio": None},
        "mlp_r50": {"num_heads": None, "mlp_ratio": 2},
        "heads_r50_mlp_r50": {"num_heads": [2, 4, 8, 8, 16], "mlp_ratio": 2},
    },
    "DEFAULT_SHAPES": ("heads_r50", "mlp_r50", "heads_r50_mlp_r50"),
    "PRUNE_TYPE": "litept_encoder_oneshot",
}


def use_litept() -> None:
    """Switch the module-level model profile to LitePT (before anything is built)."""
    global CONFIG, BASE_OVERRIDES, TRAINED_HEADS, ATTN_STAGES, SHAPES, DEFAULT_SHAPES, PRUNE_TYPE
    CONFIG, BASE_OVERRIDES = LITEPT["CONFIG"], list(LITEPT["BASE_OVERRIDES"])
    TRAINED_HEADS, ATTN_STAGES = list(LITEPT["TRAINED_HEADS"]), LITEPT["ATTN_STAGES"]
    SHAPES, DEFAULT_SHAPES, PRUNE_TYPE = dict(LITEPT["SHAPES"]), LITEPT["DEFAULT_SHAPES"], LITEPT["PRUNE_TYPE"]
METHODS = ("firstk", "score", "refit")
EXTRA_METHODS = ("blockrec", "blockrecadam", "blockrecfar")  # opt-in only (--methods), see the docstring
RANGE_EDGES = (50.0, 90.0)  # range-table bins 0-50 / 50-90 / 90-121 m (horizontal distance from the ego)
RANGE_BIN_NAMES = ("0_50", "50_90", "90_inf")
PCR_XY: list[float] | None = None  # point_cloud_range xmin, ymin, xmax, ymax; set in main() from the config


def shape_overrides(shape: str) -> dict[str, list[str]]:
    """Hydra overrides that build the pruned model, per config family."""
    s = SHAPES[shape]
    enc = []
    if s["num_heads"] is not None:
        enc += [f"model.encoder.enc_num_head=[{','.join(map(str, s['num_heads']))}]",
                f"+model.encoder.enc_head_dim={HEAD_DIM}"]
    if s["mlp_ratio"] is not None:
        enc += [f"model.encoder.mlp_ratio={s['mlp_ratio']}"]
    # The seg3d head interpolates its MLP ratio from the encoder's; pin it to the trained value.
    return {"det3d": enc, "seg3d": enc + [f"model.seg3d_head.mlp_ratio={TRAINED_MLP_RATIO}"],
            "multi": enc + [f"model.seg3d_head.mlp_ratio={TRAINED_MLP_RATIO}"]}


# ==== prefix slicer: copied from autoware_ml/scripts/tools/ptv3/prune_checkpoint.py @ 7601054 ======
_BLOCK_KEY = re.compile(r"^(?P<stage>enc\.enc(?P<s>\d+))\.block(?P<b>\d+)\.(?P<rest>.+)$")
_TRAINING_STATE_KEYS = ("optimizer_states", "lr_schedulers", "loops", "callbacks")


def _slice_block_tensor(rest, tensor, *, fc1_weight, mlp_ratio, heads, head_dim, key):
    if mlp_ratio is not None and rest in ("mlp.0.fc1.weight", "mlp.0.fc1.bias", "mlp.0.fc2.weight"):
        hidden, channels = fc1_weight.shape
        keep = int(channels * mlp_ratio)
        if not 1 <= keep <= hidden:
            raise ValueError(f"{key}: cannot slice the MLP from {hidden} to {keep} hidden units.")
        if rest == "mlp.0.fc2.weight":
            return tensor[:, :keep].contiguous().clone()
        return tensor[:keep].contiguous().clone()
    if heads is not None and rest in ("attn.qkv.weight", "attn.qkv.bias"):
        trained_heads, remainder = divmod(tensor.shape[0], 3 * head_dim)
        if remainder or not 1 <= heads <= trained_heads:
            raise ValueError(f"{key}: bad qkv layout for {heads} heads.")
        tail = tensor.shape[1:]
        sliced = tensor.reshape(3, trained_heads, head_dim, *tail)[:, :heads]
        return sliced.reshape(3 * heads * head_dim, *tail).contiguous().clone()
    if heads is not None and rest == "attn.proj.weight":
        return tensor[:, : heads * head_dim].contiguous().clone()
    return tensor.clone()


def prune_ptv3_state_dict(state_dict, *, mlp_ratio=None, num_heads=None, head_dim=None,
                          prefix="encoder."):
    """Prefix slice (depth pruning of the original removed: not used here)."""
    pruned = {}
    for key, tensor in state_dict.items():
        match = _BLOCK_KEY.match(key[len(prefix):]) if key.startswith(prefix) else None
        if match is None:
            pruned[key] = tensor.clone()
            continue
        stage, rest = int(match["s"]), match["rest"]
        block_prefix = key[: -len(rest)]
        pruned[key] = _slice_block_tensor(
            rest, tensor, fc1_weight=state_dict.get(block_prefix + "mlp.0.fc1.weight"),
            mlp_ratio=mlp_ratio, heads=None if num_heads is None else num_heads[stage],
            head_dim=head_dim, key=key)
    return pruned
# ==================================================================================================


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def compose_cfg(extra_overrides: Sequence[str]):
    root, name, extra = resolve_config_reference(CONFIG, "tasks")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=root):
        return compose(config_name=name,
                       overrides=[*extra, *BASE_OVERRIDES, *extra_overrides, f"+weights=[{CKPT}]"])


def build(cfg, weights: str, dev):
    m = build_model(cfg, data_preprocessor=build_data_preprocessor(cfg), device=dev,
                    weights_path=[weights], resume_checkpoint_path=None, set_eval=True,
                    enforce_full_coverage=True)
    return m.eval()


def export_module(model, inp):
    return _PTv3EncoderExportModule(encoder=model._prepare_encoder_export(),
                                    voxel_encoder=model.voxel_encoder,
                                    sparse_shape=inp["sparse_shape"],
                                    serialized_depth=inp["depth"]).eval()


# ---- calibration set ------------------------------------------------------------------------------
def compose_fy26_cfg():
    root, name, extra = resolve_config_reference(FY26_CONFIG, "tasks")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=root):
        return compose(config_name=name, overrides=[*extra, *FY26_OVERRIDES, f"weights=[{CKPT}]"])


def fy26_split_check(cfg, frames: list[dict]) -> dict:
    """Every frame's scenario must be in the scenario list's train split and in neither val nor test."""
    import yaml

    lists = {}
    for sc in cfg.database.scenarios.values():  # scenarios.<db name> -> T4Scenarios config
        for p in sc.dataset_params:
            f = Path(sc.scenario_root_path) / f"{p.dataset_name}.yaml"
            d = yaml.safe_load(f.read_text())
            for split in ("train", "val", "test"):
                lists.setdefault(split, set()).update(str(s).split("/")[0] for s in d.get(split) or [])
            lists.setdefault("_files", set()).add(f"{f} {md5(f)}")
    ids = {str(f["id"]["scenario_id"]).split("/")[0] for f in frames}
    rec = {"scenario_files": sorted(lists["_files"]),
           "n_train_val_test": [len(lists[s]) for s in ("train", "val", "test")],
           "n_frame_scenes": len(ids), "not_in_train": len(ids - lists["train"]),
           "in_val": len(ids & lists["val"]), "in_test": len(ids & lists["test"])}
    assert rec["not_in_train"] == rec["in_val"] == rec["in_test"] == 0, rec
    return rec


def build_frames(model, dev, n_calib: int, n_hold: int, data: str = "pseudo") -> tuple[list[dict], list[dict], dict]:
    cfg = compose_cfg(CALIB_OVERRIDES) if data == "pseudo" else compose_fy26_cfg()
    dm = build_datamodule(cfg)
    dm.prepare_data()
    dm.setup_splits((SplitType.TRAIN,))
    ds = dm.datasets[SplitType.TRAIN]
    transforms = []
    for sub in ds.datasets:
        t = getattr(sub, "transforms", None)
        pipe = getattr(t, "pipeline", getattr(t, "transforms", None))
        transforms.append([type(p).__name__ for p in pipe] if pipe is not None else repr(t)[:400])
    groups = [g for g in ds.scene_index_groups() if g]
    G = len(groups)
    calib_scenes = sorted({round(i * (G - 1) / max(n_calib - 1, 1)) for i in range(n_calib)})
    taken = set(calib_scenes)
    hold_scenes = []
    for i in range(n_hold):
        g = int((i + 0.5) * G / n_hold)
        while g in taken:
            g = (g + 1) % G
        taken.add(g)
        hold_scenes.append(g)

    def frame_id(idx: int) -> dict:
        src, si = ds.locate(idx)
        sub = ds.datasets[src]
        row = sub.dataset_records_dataframe.row(sub.record_index(si), named=True)
        keep = ("scenario_id", "scenario_name", "sample_id", "sample_index", "timestamp_seconds")
        return {"split_index": idx, **{k: row[k] for k in keep if k in row}}

    def load(scene_list, salt):
        out = []
        for j, g in enumerate(scene_list):
            grp = groups[g]
            pos = int((((j + salt) * 0.6180339887) % 1.0) * len(grp))
            idx = grp[pos]
            batch = ds.collate_fn([ds[idx]])
            with torch.no_grad():
                bi = model.on_after_batch_transfer(batch.to_device(dev), dataloader_idx=0)
                pin = prepare_ptv3_export_inputs(model, bi)
                args, names = pin.encoder_args(ENCODER_EXPORT_POOLING_FIELDS)
            out.append({"id": {**frame_id(idx), "scene_group": g},
                        "args": [a.detach().cpu() for a in args], "names": list(names),
                        "sparse_shape": pin.sparse_shape, "depth": pin.serialization_depth,
                        "n_voxels": int(args[0].shape[0])})
            if j % 32 == 0:
                print(f"  frame {j}/{len(scene_list)}: {out[-1]['id']['scenario_name'][:60]} "
                      f"N={out[-1]['n_voxels']}", flush=True)
        return out

    t0 = time.time()
    calib = load(calib_scenes, 0)
    hold = load(hold_scenes, 7)
    info = {"split": "train", "dataset_len": len(ds), "n_scenes": G, "transforms": transforms,
            "overrides": [*BASE_OVERRIDES, *CALIB_OVERRIDES], "load_seconds": round(time.time() - t0, 1)}
    if data != "pseudo":
        info.update({"data": data, "config": FY26_CONFIG, "overrides": FY26_OVERRIDES,
                     "database": cfg.database.version, "split_check": fy26_split_check(cfg, calib + hold)})
        for f in calib + hold:
            f["id"]["source"] = data
    return calib, hold, info


def to_dev(frame, dev):
    return [a.to(dev, non_blocking=True) for a in frame["args"]]


class _Stop(Exception):
    pass


# ---- statistics -----------------------------------------------------------------------------------
def collect_stats(mod, layer_name: str, frames, dev) -> dict:
    """Uncentred second moment M = sum x x^T, sum x and count of the INPUT of `layer_name` (float64)."""
    layer = dict(mod.named_modules())[layer_name]
    acc: dict[str, Any] = {"M": None, "s": None, "n": 0}

    def hook(_m, inp):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
        if acc["M"] is None:
            acc["M"] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
            acc["s"] = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
        acc["M"].addmm_(x.t(), x)
        acc["s"] += x.sum(0)
        acc["n"] += x.shape[0]
        raise _Stop

    h = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for f in frames:
                try:
                    mod(*to_dev(f, dev))
                except _Stop:
                    pass
    finally:
        h.remove()
    return acc


def select_heads(W: torch.Tensor, M: torch.Tensor, n_heads: int, keep: int) -> tuple[list[int], dict]:
    """Drop set P (|P| = H - keep) minimising p^T G p, G_ij = E<W_i o_i, W_j o_j> (exact head Gram)."""
    A = (W.t() @ W) * M  # (D, D): A_ab = (W^T W)_ab M_ab
    G = A.reshape(n_heads, HEAD_DIM, n_heads, HEAD_DIM).sum((1, 3))  # (H, H)
    d = n_heads - keep
    Gc = G.cpu().numpy()
    ncomb = math.comb(n_heads, d)
    if ncomb <= 200_000:
        best, best_v = None, math.inf
        for P in itertools.combinations(range(n_heads), d):
            v = Gc[np.ix_(P, P)].sum()
            if v < best_v:
                best, best_v = P, v
        P, how = list(best), "exhaustive"
    else:  # greedy forward + 1-swap local search
        P = []
        for _ in range(d):
            rest = [h for h in range(n_heads) if h not in P]
            P.append(min(rest, key=lambda h: Gc[np.ix_(P + [h], P + [h])].sum()))
        improved = True
        while improved:
            improved = False
            cur = Gc[np.ix_(P, P)].sum()
            for i, j in itertools.product(range(d), [h for h in range(n_heads) if h not in P]):
                Q = P.copy()
                Q[i] = j
                v = Gc[np.ix_(Q, Q)].sum()
                if v < cur - 1e-12 * abs(cur):
                    P, cur, improved = Q, v, True
                    break
        how = "greedy+swap"
    P = sorted(P)
    kept = [h for h in range(n_heads) if h not in P]
    total = Gc.sum()
    return kept, {"how": how, "drop_energy": float(Gc[np.ix_(P, P)].sum()), "total_energy": float(total),
                  "diag": [float(x) for x in np.diag(Gc)]}


def select_units(W: torch.Tensor, M: torch.Tensor, keep: int) -> tuple[list[int], dict]:
    """Wanda per hidden unit: S_j = sum_rows (|W_rj| ||x_j||)^2 = ||W[:, j]||^2 * sum_n x_nj^2."""
    S = (W ** 2).sum(0) * torch.diagonal(M)
    order = torch.argsort(S, descending=True)
    kept = sorted(order[:keep].tolist())
    return kept, {"score_kept_frac": float(S[order[:keep]].sum() / S.sum())}


def refit(W: torch.Tensor, b: torch.Tensor, M: torch.Tensor, s: torch.Tensor, n: int,
          kept_cols: list[int], damp: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Ridge LS refit of y = W x + b on the kept inputs K (float64), regularised toward W_K:
       min_{W',b'} E||W'x_K + b' - (W x + b)||^2 + lam ||W' - W_K||^2
       => W' = W_K + W_P S_PK (S_KK + lam I)^-1,  b' = b + W mu - W' mu_K
       with S = centred covariance, P = dropped inputs, lam = damp * mean diag(S_KK)."""
    D = W.shape[1]
    K = torch.tensor(kept_cols, device=W.device)
    Pm = torch.ones(D, dtype=torch.bool, device=W.device)
    Pm[K] = False
    P = Pm.nonzero().flatten()
    mu = s / n
    S = M / n - torch.outer(mu, mu)
    W64, b64 = W.double(), b.double()
    SKK = S[K][:, K]
    lam = damp * float(torch.diagonal(SKK).mean())
    SKK = SKK + lam * torch.eye(len(K), dtype=torch.float64, device=W.device)
    L = torch.linalg.cholesky(SKK)
    # X = W_P S_PK SKK^-1  ->  solve SKK X^T = S_KP W_P^T
    corr = torch.cholesky_solve((S[K][:, P] @ W64[:, P].t()), L).t()
    Wk = W64[:, K] + corr
    bk = b64 + W64 @ mu - Wk @ mu[K]
    # in-sample relative error of the sublayer output (centred part), before and after the refit
    def err(Wc_full):
        E = Wc_full - W64
        return float(torch.trace(E @ S @ E.t()) / torch.trace(W64 @ S @ W64.t()))
    Wmask = torch.zeros_like(W64)
    Wmask[:, K] = W64[:, K]
    Wfit = torch.zeros_like(W64)
    Wfit[:, K] = Wk
    mean_shift = float((W64[:, P] @ mu[P]).norm() / (b64 + W64 @ mu).norm())
    info = {"lambda": lam, "cov_err_masked": err(Wmask), "cov_err_refit": err(Wfit),
            "dropped_mean_rel": mean_shift,
            "max_abs_W": float(W.abs().max()), "max_abs_W_refit": float(Wk.abs().max()),
            "max_abs_b": float(b.abs().max()), "max_abs_b_refit": float(bk.abs().max())}
    return Wk.float(), bk.float(), info


# ---- block-wise self-reconstruction toward the unpruned stream (methods blockrec, blockrecadam) ----------
BLOCKREC_TRAIN = ("attn.qkv", "attn.proj", "mlp.0.fc1", "mlp.0.fc2", "norm1", "norm2")


def _run_hooked(mod, frame, dev, hooks, grad: bool = False) -> None:
    """Run `mod` on one frame with temporary hooks [(module, 'pre'|'post', fn)]; fn may raise _Stop."""
    hs = [m.register_forward_pre_hook(fn) if kind == "pre" else m.register_forward_hook(fn) for m, kind, fn in hooks]
    try:
        with torch.set_grad_enabled(grad):
            try:
                mod(*to_dev(frame, dev))
            except _Stop:
                pass
    finally:
        for h in hs:
            h.remove()


def _ls_asym(ZZ: torch.Tensor, ZT: torch.Tensor, TT: float, W0: torch.Tensor, b0: torch.Tensor,
             damp: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """min_{W,b} sum ||W z + b - t||^2 + lam ||W - W0||^2 from the sums ZZ = sum [z;1][z;1]^T,
    ZT = sum [z;1] t^T, TT = sum ||t||^2 (float64); lam = damp * n * mean centred var(z) (= refit's)."""
    d = W0.shape[1]
    n = float(ZZ[d, d])
    mu = ZZ[:d, d] / n
    var = torch.diagonal(ZZ[:d, :d]) / n - mu * mu
    lam = damp * n * float(var.mean())
    R = torch.zeros_like(ZZ)
    R.diagonal()[:d] = lam
    X0 = torch.cat([W0.double().t(), b0.double()[None]], 0)  # (d+1, out)
    X = torch.linalg.solve(ZZ + R, ZT + R @ X0)

    def sse(Xc):
        return float(torch.trace(Xc.t() @ ZZ @ Xc) - 2 * torch.trace(Xc.t() @ ZT) + TT)
    return X[:d].t().float(), X[d].float(), {"lambda_sum": lam, "sse_before": sse(X0), "sse_after": sse(X)}


def _block_target(ref_blk, ref_mod, frame, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpruned stream: (post-attention residual h_u = norm2 input, block output y_u) of one block."""
    cap = {}

    def pre_n2(_m, inp):
        cap["h"] = inp[0].feat.detach()

    def post_blk(_m, _inp, out):
        cap["y"] = out.feat.detach()
        raise _Stop
    _run_hooked(ref_mod, frame, dev, [(ref_blk.norm2, "pre", pre_n2), (ref_blk, "post", post_blk)])
    return cap["h"], cap["y"]


def _block_input(blk, mod, frame, dev):
    """Pruned stream: the Point entering block `blk` (shallow copy; the block reassigns, never mutates)."""
    cap = {}

    def pre(_m, inp):
        cap["p"] = Point(inp[0])
        raise _Stop
    _run_hooked(mod, frame, dev, [(blk, "pre", pre)])
    return cap["p"]


def _token_range(point) -> torch.Tensor:
    """Horizontal distance (m) from the ego of every token of a Point at any stage: the export path carries
    integer grid_coord per stage (pooled coord = grid index), cell = point_cloud_range extent / sparse_shape."""
    ss = torch.as_tensor(point.sparse_shape).reshape(-1).double().cpu()
    assert ss.numel() == 3 and ss[0] == ss[1], f"sparse_shape {ss.tolist()} not x,y,z with x == y"
    lo = torch.tensor(PCR_XY[:2], dtype=torch.float64)
    cell = (torch.tensor(PCR_XY[2:], dtype=torch.float64) - lo) / ss[:2]
    g = point.grid_coord[:, :2].double()
    r = ((g + 0.5) * cell.to(g.device) + lo.to(g.device)).norm(dim=1)
    assert float(r.max()) < 176.0, float(r.max())  # |xy| <= 122.88 * sqrt(2)
    return r


def _range_bin(r: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(r, torch.tensor(RANGE_EDGES, dtype=r.dtype, device=r.device), right=True)


def _range_weight(r: torch.Tensor, spec) -> torch.Tensor:
    """Per-token weight: 1, or the weight of the last threshold of spec [(r_min, w), ...] that r reaches."""
    w = torch.ones_like(r)
    for thr, wt in sorted(spec):
        w[r >= thr] = wt
    return w


def _block_err(blk, ref_blk, mod, ref_mod, frames, dev, bins: bool = False):
    """Relative block-output error on frames; with bins=True also per range bin (pooled over tokens)."""
    num = den = 0.0
    nb = np.zeros(len(RANGE_BIN_NAMES))
    db = np.zeros(len(RANGE_BIN_NAMES))
    with torch.no_grad():
        for f in frames:
            _, y_u = _block_target(ref_blk, ref_mod, f, dev)
            p_in = _block_input(blk, mod, f, dev)
            out = blk(Point(p_in))
            y = out.feat
            e = ((y.double() - y_u.double()) ** 2).sum(1)
            t = (y_u.double() ** 2).sum(1)
            num += float(e.sum())
            den += float(t.sum())
            if bins:
                b = _range_bin(_token_range(out))
                for i in range(len(RANGE_BIN_NAMES)):
                    nb[i] += float(e[b == i].sum())
                    db[i] += float(t[b == i].sum())
    if not bins:
        return num / den
    return num / den, [float(n / d) if d > 0 else None for n, d in zip(nb, db)]


def blockrec(mod, ref_mod, kept_cols: dict, calib, dev, damp: float, adam_steps: int, adam_lr: float,
             log, range_weights=None, range_bins: bool = False) -> dict:
    """Sequential block-wise reconstruction of the masked pruned export module `mod` toward `ref_mod`
    (unpruned). kept_cols[bn] = {"attn": proj input columns kept, "mlp": fc2 input columns kept}.
    range_weights [(r_min, w), ...]: weight every token's LS term by its distance (method blockrecfar).
    range_bins: also record the block errors per range bin. Changes mod's block parameters in place;
    returns a record per block."""
    mods, rmods = dict(mod.named_modules()), dict(ref_mod.named_modules())
    val = calib[::16]
    train = [f for i, f in enumerate(calib) if i % 16]
    recs = {}
    for _, _, bn in block_names():
        t0 = time.time()
        blk, rblk = mods[bn], rmods[bn]
        rec: dict[str, Any] = {}
        if range_bins:
            rec["err_before"], rec["err_before_bins"] = _block_err(blk, rblk, mod, ref_mod, val, dev, bins=True)
        else:
            rec["err_before"] = _block_err(blk, rblk, mod, ref_mod, val, dev)
        tok = np.zeros(len(RANGE_BIN_NAMES))
        wsum = np.zeros(len(RANGE_BIN_NAMES))
        # (1) attn.proj: shortcut_p + proj(o_p) -> h_u ; (2) fc2: shortcut2_p + fc2(z_p) -> y_u
        for part, lname, stop_mod in (("attn", f"{bn}.attn.proj", blk.norm2), ("mlp", f"{bn}.mlp.0.fc2", None)):
            lin = mods[lname]
            K = torch.tensor(kept_cols[bn][part], device=lin.weight.device)
            d = len(K)
            ZZ = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=lin.weight.device)
            ZT = torch.zeros(d + 1, lin.weight.shape[0], dtype=torch.float64, device=lin.weight.device)
            TT = YY = 0.0
            for f in calib:
                h_u, y_u = _block_target(rblk, ref_mod, f, dev)
                cap = {}

                def pre_lin(_m, inp):
                    cap["z"] = inp[0].detach()

                def pre_n2(_m, inp):
                    cap["s"] = inp[0].feat.detach()
                    cap["p"] = inp[0]
                    raise _Stop

                def post_blk(_m, _inp, out):
                    cap["s"] = out.feat.detach()
                    cap["p"] = out
                    raise _Stop
                hooks = [(lin, "pre", pre_lin)] + ([(stop_mod, "pre", pre_n2)] if part == "attn" else [(blk, "post", post_blk)])
                _run_hooked(mod, f, dev, hooks)
                torch.set_grad_enabled(False)  # (job 3209 OOM: t required grad, ZT.addmm_ chained the graph)
                z = cap["z"].double()
                shortcut = cap["s"].double() - torch.nn.functional.linear(z, lin.weight.double(), lin.bias.double())
                t = (h_u if part == "attn" else y_u).double() - shortcut
                za = torch.cat([z[:, K], torch.ones(z.shape[0], 1, dtype=torch.float64, device=z.device)], 1)
                if range_weights is None:
                    ZZ.addmm_(za.t(), za)
                    ZT.addmm_(za.t(), t)
                    TT += float((t * t).sum())
                    YY += float(((h_u if part == "attn" else y_u).double() ** 2).sum())
                else:  # weighted LS: every sum gets the token's weight (lam scales with the weighted n)
                    r = _token_range(cap["p"])
                    w = _range_weight(r, range_weights)
                    assert w.shape[0] == za.shape[0], (w.shape, za.shape)
                    zw = za * w[:, None]
                    ZZ.addmm_(zw.t(), za)
                    ZT.addmm_(zw.t(), t)
                    TT += float((w * (t * t).sum(1)).sum())
                    YY += float((w * ((h_u if part == "attn" else y_u).double() ** 2).sum(1)).sum())
                    if part == "attn":
                        b = _range_bin(r)
                        for i in range(len(RANGE_BIN_NAMES)):
                            tok[i] += float((b == i).sum())
                            wsum[i] += float(w[b == i].sum())
                torch.set_grad_enabled(True)
            Wk, bk, info = _ls_asym(ZZ, ZT, TT, lin.weight.data[:, K], lin.bias.data, damp)
            newW = torch.zeros_like(lin.weight.data)
            newW[:, K] = Wk
            with torch.no_grad():
                lin.weight.copy_(newW)
                lin.bias.copy_(bk)
            rec[f"{part}_rel_before"] = info["sse_before"] / YY
            rec[f"{part}_rel_after"] = info["sse_after"] / YY
            rec[f"{part}_max_abs_W"] = float(Wk.abs().max())
        if range_weights is not None:
            rec["token_share_bins"] = [float(x) for x in tok / tok.sum()]
            rec["weight_share_bins"] = [float(x) for x in wsum / wsum.sum()]
        if range_bins:
            rec["err_closed"], rec["err_closed_bins"] = _block_err(blk, rblk, mod, ref_mod, val, dev, bins=True)
        else:
            rec["err_closed"] = _block_err(blk, rblk, mod, ref_mod, val, dev)
        if adam_steps > 0:
            params, masks = [], []
            for name, p in blk.named_parameters():
                if any(name.startswith(t + ".") for t in BLOCKREC_TRAIN):
                    params.append(p)
                    m = None
                    if name in ("attn.proj.weight", "mlp.0.fc2.weight"):
                        m = torch.zeros_like(p)
                        m[:, kept_cols[bn]["attn" if name.startswith("attn") else "mlp"]] = 1
                    masks.append(m)
            backup = [p.detach().clone() for p in params]
            req = {id(p): p.requires_grad for p in mod.parameters()}
            for p in mod.parameters():
                p.requires_grad_(False)
            for p in params:
                p.requires_grad_(True)
            opt = torch.optim.Adam(params, lr=adam_lr)
            g = torch.Generator().manual_seed(0)
            losses = []
            for step in range(adam_steps):
                f = train[int(torch.randint(len(train), (1,), generator=g))]
                _, y_u = _block_target(rblk, ref_mod, f, dev)
                p_in = _block_input(blk, mod, f, dev)
                with torch.enable_grad():
                    y = blk(Point(p_in)).feat
                    loss = ((y - y_u) ** 2).sum() / (y_u ** 2).sum()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                for p, m in zip(params, masks):
                    if m is not None and p.grad is not None:
                        p.grad.mul_(m)
                opt.step()
                with torch.no_grad():
                    for p, m in zip(params, masks):
                        if m is not None:
                            p.mul_(m)
                losses.append(float(loss))
            for p in mod.parameters():
                p.requires_grad_(req[id(p)])
            rec["adam_loss_first20"] = float(np.mean(losses[:20]))
            rec["adam_loss_last20"] = float(np.mean(losses[-20:]))
            rec["err_adam"] = _block_err(blk, rblk, mod, ref_mod, val, dev)
            rec["adam_kept"] = rec["err_adam"] < rec["err_closed"]
            if not rec["adam_kept"]:
                with torch.no_grad():
                    for p, b in zip(params, backup):
                        p.copy_(b)
        rec["seconds"] = round(time.time() - t0, 1)
        recs[bn] = rec
        log(f"  blockrec {bn}: " + " ".join(f"{k}={v:.4g}" for k, v in rec.items() if isinstance(v, float))
            + (f" adam_kept={rec['adam_kept']}" if "adam_kept" in rec else "")
            + "".join(f" {k}={[None if x is None else round(x, 4) for x in v]}" for k, v in rec.items()
                      if k.endswith("_bins")))
    return recs


# ---- one arm ---------------------------------------------------------------------------------------
def block_names(depths=None):
    """Encoder blocks with attention + MLP (all of PTv3's; LitePT: stages 3-4 only), in forward order."""
    depths = ENC_DEPTHS if depths is None else depths
    return [(s, b, f"encoder.enc.enc{s}.block{b}") for s in range(len(depths)) if s in ATTN_STAGES
            for b in range(depths[s])]


def prune_arm(model, shape: str, method: str, calib, dev, damp: float, log, ref_model=None,
              blockrec_steps: int = 0, blockrec_lr: float = 1e-5, range_weights=None,
              range_bins: bool = False) -> tuple[dict, dict, Any]:
    """Return (pruned state_dict, selection record, masked export module).
    blockrec / blockrecadam: refit, then blockrec() toward the export module of `ref_model` (unpruned)."""
    spec = SHAPES[shape]
    rec_method = method
    if method in EXTRA_METHODS:
        method = "refit"
    mod = export_module(model, calib[0])
    mods = dict(mod.named_modules())
    sel: dict[str, Any] = {}
    # parts pruned per block, in forward order (attention before MLP, so the MLP statistics of a block
    # are taken with that block's attention already pruned); a single-part shape keys records by block
    parts = (["attn"] if spec["num_heads"] is not None else []) + (["mlp"] if spec["mlp_ratio"] is not None else [])

    def key(bn, part):
        return bn if len(parts) == 1 else f"{bn}.{part}"

    for s, b, bn in block_names():
        for part in parts:
            t0 = time.time()
            rec: dict[str, Any] = {}
            if part == "attn":
                layer, H, k = f"{bn}.attn.proj", TRAINED_HEADS[s], spec["num_heads"][s]
            else:
                layer = f"{bn}.mlp.0.fc2"
                H = mods[layer].weight.shape[1]
                k = int(mods[layer].weight.shape[0] * spec["mlp_ratio"])
            lin = mods[layer]
            W, bias = lin.weight.data, lin.bias.data
            if method == "firstk":
                kept = list(range(k))
            else:
                st = collect_stats(mod, layer, calib, dev)
                rec["n_tokens"] = st["n"]
                if part == "attn":
                    kept, info = select_heads(W.double(), st["M"] / st["n"], H, k)
                else:
                    kept, info = select_units(W.double(), st["M"] / st["n"], k)
                rec.update(info)
            cols = [h * HEAD_DIM + c for h in kept for c in range(HEAD_DIM)] if part == "attn" else kept
            if method == "refit":
                Wk, bk, info = refit(W, bias, st["M"], st["s"], st["n"], cols, damp)
                rec.update(info)
            else:
                Wk, bk = W[:, cols].clone(), bias.clone()
            newW = torch.zeros_like(W)
            newW[:, cols] = Wk
            lin.weight.data.copy_(newW)
            lin.bias.data.copy_(bk)
            rec["kept"] = kept
            rec["seconds"] = round(time.time() - t0, 1)
            sel[key(bn, part)] = rec
            log(f"  {shape}/{rec_method} {key(bn, part)}: keep {len(kept)}/{H} "
                + " ".join(f"{k_}={v:.4g}" for k_, v in rec.items() if isinstance(v, float)) + f" ({rec['seconds']}s)")

    if rec_method in EXTRA_METHODS:
        kept_cols = {}
        for s, b, bn in block_names():
            kc = {"attn": list(range(mods[f"{bn}.attn.proj"].weight.shape[1])),
                  "mlp": list(range(mods[f"{bn}.mlp.0.fc2"].weight.shape[1]))}
            if "attn" in parts:
                kc["attn"] = [h * HEAD_DIM + c for h in sel[key(bn, "attn")]["kept"] for c in range(HEAD_DIM)]
            if "mlp" in parts:
                kc["mlp"] = list(sel[key(bn, "mlp")]["kept"])
            kept_cols[bn] = kc
        # the block parameters blockrec may change must be the model's own (prepare_for_export copies them)
        msd = model.state_dict()
        trained = [n for n, _ in mod.named_parameters()
                   if any(f".{t}." in n for t in BLOCKREC_TRAIN) and ".block" in n]
        for n in trained:
            if not n.endswith(("attn.proj.weight", "attn.proj.bias", "mlp.0.fc2.weight", "mlp.0.fc2.bias")):
                assert torch.equal(dict(mod.named_parameters())[n].detach(), msd[n]), f"{n} differs in the export copy"
        ref_mod = export_module(ref_model, calib[0])
        steps = blockrec_steps if rec_method == "blockrecadam" else 0
        br = blockrec(mod, ref_mod, kept_cols, calib, dev, damp, steps, blockrec_lr, log,
                      range_weights=range_weights if rec_method == "blockrecfar" else None, range_bins=range_bins)
        del ref_mod
        for bn, r in br.items():
            sel[key(bn, parts[-1])]["blockrec"] = r
        with torch.no_grad():  # write the reconstructed block parameters back into the model
            mp = dict(model.named_parameters())
            for n, p in mod.named_parameters():
                if n in trained:
                    mp[n].copy_(p.detach())

    # ---- permute kept units to the front of the original tensors, then prefix-slice -----------------
    ck_sd = model.state_dict()
    sd = {k: v.detach().cpu().clone() for k, v in ck_sd.items()}
    for s, b, bn in block_names():
        if "attn" in parts:
            kept = sel[key(bn, "attn")]["kept"]
            H = TRAINED_HEADS[s]
            perm = kept + [h for h in range(H) if h not in kept]
            qw = sd[f"{bn}.attn.qkv.weight"]
            sd[f"{bn}.attn.qkv.weight"] = qw.reshape(3, H, HEAD_DIM, -1)[:, perm].reshape(qw.shape).contiguous()
            qb = sd[f"{bn}.attn.qkv.bias"]
            sd[f"{bn}.attn.qkv.bias"] = qb.reshape(3, H, HEAD_DIM)[:, perm].reshape(qb.shape).contiguous()
            cols = [h * HEAD_DIM + c for h in perm for c in range(HEAD_DIM)]
            lin = mods[f"{bn}.attn.proj"]
            sd[f"{bn}.attn.proj.weight"] = lin.weight.data[:, cols].detach().cpu().contiguous()
            sd[f"{bn}.attn.proj.bias"] = lin.bias.data.detach().cpu().clone()
        if "mlp" in parts:
            kept = sel[key(bn, "mlp")]["kept"]
            Hd = sd[f"{bn}.mlp.0.fc1.weight"].shape[0]
            perm = kept + [j for j in range(Hd) if j not in kept]
            sd[f"{bn}.mlp.0.fc1.weight"] = sd[f"{bn}.mlp.0.fc1.weight"][perm].contiguous()
            sd[f"{bn}.mlp.0.fc1.bias"] = sd[f"{bn}.mlp.0.fc1.bias"][perm].contiguous()
            lin = mods[f"{bn}.mlp.0.fc2"]
            sd[f"{bn}.mlp.0.fc2.weight"] = lin.weight.data[:, perm].detach().cpu().contiguous()
            sd[f"{bn}.mlp.0.fc2.bias"] = lin.bias.data.detach().cpu().clone()
    pruned = prune_ptv3_state_dict(sd, mlp_ratio=spec["mlp_ratio"], num_heads=spec["num_heads"],
                                   head_dim=HEAD_DIM if spec["num_heads"] is not None else None)
    return pruned, sel, mod


def run_outputs(mod, frames, dev) -> list[list[torch.Tensor]]:
    outs = []
    with torch.no_grad():
        for f in frames:
            outs.append([o.detach().float().cpu() for o in mod(*to_dev(f, dev))])
    return outs


def rel_l2(a: list[list[torch.Tensor]], ref: list[list[torch.Tensor]]) -> dict:
    per_k = []
    for k in range(len(ref[0])):
        r = [float((x[k].double() - y[k].double()).norm() / y[k].double().norm()) for x, y in zip(a, ref)]
        per_k.append(r)
    arr = np.array(per_k)  # (5, frames)
    return {"rel_l2_mean": [float(x) for x in arr.mean(1)], "rel_l2_max": [float(x) for x in arr.max(1)],
            "rel_l2_mean_all": float(arr.mean())}


def stage_ranges(mod, frames, dev) -> list[list[torch.Tensor]]:
    """Per frame, per encoder stage: horizontal distance (m) of every output token (rows of point_feat_k)."""
    out = []
    with torch.no_grad():
        for f in frames:
            stages = collect_encoder_stage_points(mod.run_encoder(*to_dev(f, dev)))
            out.append([_token_range(s).cpu() for s in stages])
    return out


def rel_l2_bins(a, ref, rngs) -> dict:
    """Pooled relative L2 per stage and range bin: sqrt(sum ||a - ref||^2 / sum ||ref||^2) over the bin's tokens."""
    res = {}
    for i, name in enumerate(RANGE_BIN_NAMES):
        per_k, ntok = [], []
        for k in range(len(ref[0])):
            num = den = 0.0
            n = 0
            for x, y, r in zip(a, ref, rngs):
                assert r[k].shape[0] == y[k].shape[0], (r[k].shape, y[k].shape)
                m = _range_bin(r[k]) == i
                num += float(((x[k][m].double() - y[k][m].double()) ** 2).sum())
                den += float((y[k][m].double() ** 2).sum())
                n += int(m.sum())
            per_k.append(math.sqrt(num / den) if den > 0 else None)
            ntok.append(n)
        res[name] = {"rel_l2": per_k, "n_tokens": ntok,
                     "rel_l2_mean_stages": float(np.mean([x for x in per_k if x is not None]))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all"])
    ap.add_argument("--model", default="ptv3", choices=["ptv3", "litept"],
                    help="ptv3 (default, the original behaviour) or litept (attention stages 3-4 only; see LITEPT)")
    ap.add_argument("--decoder", action="store_true",
                    help="also prune the decoder attention blocks: NOT supported (rejected), see LITEPT")
    ap.add_argument("--shapes", nargs="+", default=None, choices=list(SHAPES),
                    help="default: heads_r50 mlp_r25 mlp_r50 (ptv3) / heads_r50 mlp_r50 heads_r50_mlp_r50 (litept)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--n-holdout", type=int, default=20)
    ap.add_argument("--damp", type=float, default=0.01, help="ridge lambda / mean diag(S_KK)")
    ap.add_argument("--blockrec-steps", type=int, default=300, help="Adam steps per block (method blockrecadam)")
    ap.add_argument("--blockrec-lr", type=float, default=1e-5, help="Adam lr (method blockrecadam)")
    ap.add_argument("--range-weights", default="50:4,90:8",
                    help="method blockrecfar: r_min:weight,... (horizontal distance, m; weight 1 below the first)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--calib-set", default="pseudo", choices=CALIB_SETS,
                    help="calibration frames: pseudo (default, original), fy26h1, mix (128 + 128)")
    ap.add_argument("--holdout-sets", nargs="+", choices=HOLDOUT_SETS, default=None,
                    help="proxy hold-out sets, first = primary (default: pseudo for --calib-set pseudo, "
                         "else fy26h1 pseudo)")
    args = ap.parse_args()
    if args.decoder:
        sys.exit("--decoder: decoder attention pruning is not implemented (PTv3Seg3DDecoderHead has no head-width "
                 "option; the calibration path is the export encoder only)")
    if args.model == "litept":
        if "ONESHOT_CKPT" not in os.environ or "ONESHOT_CKPT_MD5" not in os.environ:
            sys.exit("--model litept needs ONESHOT_CKPT and ONESHOT_CKPT_MD5 (the LitePT source checkpoint)")
        use_litept()
        # the pseudo frames are read through CONFIG's datamodule (the PTv3 stage-3 leaf); LitePT uses FY26H1 only
        args.holdout_sets = args.holdout_sets or ["fy26h1"]
        if args.calib_set != "fy26h1" or args.holdout_sets != ["fy26h1"]:
            sys.exit("--model litept: only --calib-set fy26h1 with --holdout-sets fy26h1 is supported")
        bad = set(args.shapes or ()) - set(SHAPES)
        if bad:
            sys.exit(f"--model litept: shapes {sorted(bad)} not defined; choose from {list(SHAPES)}")
    if args.shapes is None:
        args.shapes = list(DEFAULT_SHAPES)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_all = time.time()
    if md5(Path(CKPT)) != CKPT_MD5:
        sys.exit("checkpoint md5 mismatch")
    dev = torch.device("cuda")
    log(f"== oneshot_prune {time.strftime('%Y-%m-%d %H:%M:%S')} worktree HEAD {os.environ.get('ONESHOT_WT_HEAD')} "
        f"args {vars(args)}")
    base_cfg = compose_cfg([])
    global PCR_XY
    pcr = [float(x) for x in base_cfg.point_cloud_range]
    PCR_XY = [pcr[0], pcr[1], pcr[3], pcr[4]]
    range_weights = [(float(a), float(b)) for a, b in (x.split(":") for x in args.range_weights.split(","))]
    base_model = build(base_cfg, CKPT, dev)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    # the model's state dict must be the checkpoint's (so the permuted/sliced dict is complete)
    msd = base_model.state_dict()
    assert set(msd) == set(ckpt["state_dict"]), set(msd) ^ set(ckpt["state_dict"])
    for k, v in ckpt["state_dict"].items():
        assert torch.equal(msd[k].cpu(), v), k

    t0 = time.time()
    hsets = args.holdout_sets or (["pseudo"] if args.calib_set == "pseudo" else ["fy26h1", "pseudo"])
    default_data = args.calib_set == "pseudo" and hsets == ["pseudo"]
    if default_data:  # the original path, unchanged
        calib, hold, info = build_frames(base_model, dev, args.n_calib, args.n_holdout)
        holds = {"pseudo": hold}
    else:
        need = sorted(set(hsets) | ({"pseudo", "fy26h1"} if args.calib_set == "mix" else {args.calib_set}))
        built = {d: build_frames(base_model, dev, args.n_calib, args.n_holdout, d) for d in need}
        for f in built.get("pseudo", ([], [], None))[0] + built.get("pseudo", ([], [], None))[1]:
            f["id"]["source"] = "pseudo"
        if args.calib_set == "mix":  # every other frame of each 256-frame set (128 + 128)
            calib = built["pseudo"][0][0::2] + built["fy26h1"][0][0::2]
        else:
            calib = built[args.calib_set][0]
        holds = {d: built[d][1] for d in hsets}
        hold = holds[hsets[0]]
        info = {"calib_set": args.calib_set, "holdout_sets": hsets, "data": {d: b[2] for d, b in built.items()}}
        # one export graph serves every frame: sparse shape and serialisation depth must agree
        allf = calib + [f for h in holds.values() for f in h]
        geom = {(str([int(x) for x in f["sparse_shape"]]), str(f["depth"])) for f in allf}
        assert len(geom) == 1, geom
        info["export_geometry"] = sorted(geom)[0]
    info["build_seconds"] = round(time.time() - t0, 1)
    rec = {**info, "calib": [f["id"] | {"n_voxels": f["n_voxels"]} for f in calib],
           "holdout": [f["id"] | {"n_voxels": f["n_voxels"]} for f in hold]}
    if not default_data:
        rec.update({f"holdout_{d}": [f["id"] | {"n_voxels": f["n_voxels"]} for f in h] for d, h in holds.items()})
    (out / "calibration_set.json").write_text(json.dumps(rec, indent=1, default=str))
    log(f"calibration: {len(calib)} frames, holdout {len(hold)} frames ({info['build_seconds']} s); "
        + (f"transforms {info['transforms'][0]}" if default_data else
           f"calib_set {args.calib_set} holdout_sets {hsets} "
           + " ".join(f"{d}: len {i['dataset_len']} scenes {i['n_scenes']} split_check {i.get('split_check')}"
                      for d, i in info["data"].items())))
    for h in holds.values():
        assert not {f['id']['scenario_id'] for f in calib} & {f['id']['scenario_id'] for f in h}

    # export modules prepare a COPY: check proj/fc2/qkv/fc1 are copied unchanged
    ref_mod = export_module(base_model, calib[0])
    rm = dict(ref_mod.named_parameters())
    for _, _, bn in block_names():
        for t in ("attn.qkv.weight", "attn.proj.weight", "attn.proj.bias", "mlp.0.fc1.weight", "mlp.0.fc2.weight",
                  "mlp.0.fc2.bias"):
            assert torch.equal(rm[f"{bn}.{t}"], msd[f"{bn}.{t}"]), f"{bn}.{t} changed by prepare_for_export"
    ref_hold = run_outputs(ref_mod, hold, dev)
    ref_rerun = run_outputs(ref_mod, hold, dev)
    noise = rel_l2(ref_rerun, ref_hold)
    log(f"reference run-to-run rel L2: {noise['rel_l2_mean']}")
    ref_holds = {d: (ref_hold if d == hsets[0] else run_outputs(ref_mod, h, dev)) for d, h in holds.items()}
    hold_rngs = {}
    if not default_data:  # token distances per hold-out frame and stage (geometry is an input: same for every arm)
        hold_rngs = {d: stage_ranges(ref_mod, h, dev) for d, h in holds.items()}
        for d, rr in hold_rngs.items():
            shares = []
            for k in range(len(rr[0])):
                b = _range_bin(torch.cat([r[k] for r in rr]))
                shares.append("/".join(f"{float((b == i).double().mean()):.3f}" for i in range(len(RANGE_BIN_NAMES))))
            log(f"holdout {d} token share per range bin {RANGE_BIN_NAMES}, pf0..pf4: {shares}")
    del ref_mod

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.setdefault("_reference_run_to_run", noise)
    if noise != results["_reference_run_to_run"]:
        results[f"_reference_run_to_run_{'+'.join(args.shapes)}"] = noise
    for shape in args.shapes:
        for method in args.methods:
            arm = f"{shape}_{method}"
            d = out / arm
            d.mkdir(exist_ok=True)
            t0 = time.time()
            log(f"== {arm}")
            work_model = build(compose_cfg([]), CKPT, dev)  # fresh unpruned copy per arm
            pruned_sd, sel, masked_mod = prune_arm(work_model, shape, method, calib, dev, args.damp, log,
                                                   ref_model=base_model, blockrec_steps=args.blockrec_steps,
                                                   blockrec_lr=args.blockrec_lr, range_weights=range_weights,
                                                   range_bins=not default_data)
            t_prune = time.time() - t0
            payload = {k: v for k, v in ckpt.items() if k not in _TRAINING_STATE_KEYS and k != "state_dict"}
            payload["state_dict"] = pruned_sd
            payload["autoware_ml_prune"] = {
                "type": f"{PRUNE_TYPE}_{method}", "shape": shape,
                "mlp_ratio": SHAPES[shape]["mlp_ratio"], "num_heads": SHAPES[shape]["num_heads"],
                "head_dim": HEAD_DIM if SHAPES[shape]["num_heads"] else None, "prefix": "encoder.",
                "damp": args.damp if method in ("refit", *EXTRA_METHODS) else None,
                "n_calib": len(calib) if method != "firstk" else 0,
                "kept": {k_: r["kept"] for k_, r in sel.items()},
                "source_ckpt_md5": CKPT_MD5, "source_epoch": ckpt.get("epoch"),
                "source_global_step": ckpt.get("global_step"), "tool": "tools/oneshot_prune.py",
                "autoware_ml_worktree_head": os.environ.get("ONESHOT_WT_HEAD")}
            if args.calib_set != "pseudo":
                payload["autoware_ml_prune"]["calib_set"] = args.calib_set
            if method in EXTRA_METHODS:
                payload["autoware_ml_prune"]["blockrec"] = {
                    "target": "unpruned block output (pruned-stream input)", "closed_form": ["attn.proj", "mlp.0.fc2"],
                    "adam_steps": args.blockrec_steps if method == "blockrecadam" else 0,
                    "adam_lr": args.blockrec_lr if method == "blockrecadam" else None}
                if method == "blockrecfar":
                    payload["autoware_ml_prune"]["blockrec"]["range_weights"] = range_weights
            torch.save(payload, d / "pruned.ckpt")
            ov = shape_overrides(shape)
            (d / "overrides.txt").write_text("".join(f"{k}: {' '.join(v)}\n" for k, v in ov.items()))
            (d / "selection.json").write_text(json.dumps(sel, indent=1))
            ck_md5 = md5(d / "pruned.ckpt")
            (d / "md5.txt").write_text(f"{ck_md5}  pruned.ckpt\n")

            # proxy: build the pruned model from the checkpoint with its overrides (strict), run hold-out
            masked_out = run_outputs(masked_mod, hold, dev)
            del masked_mod, work_model
            pm = build(compose_cfg(ov["multi"]), str(d / "pruned.ckpt"), dev)
            pm.load_state_dict(pruned_sd, strict=True)
            pmod = export_module(pm, hold[0])
            p_out = run_outputs(pmod, hold, dev)
            res = rel_l2(p_out, ref_hold)
            surgery = rel_l2(p_out, masked_out)
            res.update({"sliced_vs_masked_rel_l2_max": max(surgery["rel_l2_max"]),
                        "md5": ck_md5, "prune_seconds": round(t_prune, 1),
                        "total_seconds": round(time.time() - t0, 1),
                        "n_params_encoder": sum(v.numel() for k, v in pruned_sd.items() if k.startswith("encoder.")),
                        "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            if not default_data:  # every hold-out set, named (the primary one also at the top level)
                for hd, h in holds.items():
                    o_h = p_out if hd == hsets[0] else run_outputs(pmod, h, dev)
                    res[f"holdout_{hd}"] = res_h = rel_l2(o_h, ref_holds[hd])
                    res_h["range_bins"] = rb = rel_l2_bins(o_h, ref_holds[hd], hold_rngs[hd])
                    log(f"  {arm} holdout {hd}: rel L2 point_feat_0..4 = "
                        f"{[round(x, 4) for x in res_h['rel_l2_mean']]} mean {res_h['rel_l2_mean_all']:.4f}; per bin "
                        + " ".join(f"{b}: {[None if x is None else round(x, 4) for x in v['rel_l2']]}"
                                   for b, v in rb.items()))
                res["calib_set"], res["holdout_sets"] = args.calib_set, hsets
            if method in EXTRA_METHODS:
                payload_br = {"steps": args.blockrec_steps if method == "blockrecadam" else 0,
                              "lr": args.blockrec_lr if method == "blockrecadam" else None}
                if method == "blockrecfar":
                    payload_br["range_weights"] = range_weights
                res["blockrec"] = payload_br
                brs = [r["blockrec"] for r in sel.values() if "blockrec" in r]
                res["blockrec_err_before_mean"] = float(np.mean([r["err_before"] for r in brs]))
                res["blockrec_err_closed_mean"] = float(np.mean([r["err_closed"] for r in brs]))
                if method == "blockrecadam":
                    res["blockrec_err_adam_mean"] = float(np.mean([min(r["err_adam"], r["err_closed"]) for r in brs]))
                    res["blockrec_adam_kept_blocks"] = sum(bool(r["adam_kept"]) for r in brs)
            if method in ("refit", *EXTRA_METHODS):
                res["max_abs_W_ratio"] = max(r["max_abs_W_refit"] / r["max_abs_W"] for r in sel.values())
                res["max_abs_b_refit"] = max(r["max_abs_b_refit"] for r in sel.values())
            results[arm] = res
            results_path.write_text(json.dumps(results, indent=1))
            log(f"  {arm}: rel L2 point_feat_0..4 = {[round(x, 4) for x in res['rel_l2_mean']]} "
                f"mean {res['rel_l2_mean_all']:.4f}; sliced-vs-masked {res['sliced_vs_masked_rel_l2_max']:.2e}; "
                f"md5 {ck_md5}; {res['total_seconds']} s")
            del pm, pmod
            torch.cuda.empty_cache()
    tkey = "_total_seconds" if tuple(args.shapes) == DEFAULT_SHAPES else f"_total_seconds_{'+'.join(args.shapes)}"
    results[tkey] = round(time.time() - t_all, 1)
    results_path.write_text(json.dumps(results, indent=1))
    log(f"done in {results[tkey]} s")


if __name__ == "__main__":
    main()
