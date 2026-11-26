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

"""Unit tests for the LitePT encoder export with norm1 as row statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import onnx
import torch
from torch import nn

from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_litept_seg3d_head,
    build_seg3d_model,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    QKV_ROW_SCALE_KEY,
    Block,
    LayerNormRowStatistics,
    LitePTEncoder,
    Norm1RowStatistics,
    fold_norm1_row_statistics,
)
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.models.segmentation3d.ptv3 import PTv3Segmentation3DModel
from autoware_ml.models.segmentation3d.ptv3_base import (
    ENCODER_GRAPH,
    norm1_row_statistic_node_names,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import export_to_onnx
from autoware_ml.utils.point_cloud.structures import Point

REQUIRES_SPARSE_CUDA = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(), "LitePT export needs spconv and CUDA"
)
#: The export settings of the PTv3 base configs.
BASE_ONNX_CONFIG = {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
#: The operators of one block's row statistics, in graph order.
ROW_STATISTIC_OPS = ["ReduceMean", "Sub", "Mul", "ReduceMean", "Add", "Sqrt", "Div"]


def randomize_layer_norms(module: nn.Module, seed: int = 0) -> None:
    """Give every LayerNorm a non-trivial affine, so the fold has something to fold.

    Args:
        module: Module whose LayerNorms change in place.
        seed: Seed of the generator.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.LayerNorm):
                shape = child.weight.shape
                child.weight.copy_(1.0 + 0.3 * torch.randn(shape, generator=generator))
                child.bias.copy_(0.3 * torch.randn(shape, generator=generator))


def build_attention_block(pre_norm: bool = True) -> Block:
    """Return a LitePT-style attention-only block with RoPE and random LayerNorm affines.

    Args:
        pre_norm: Whether the block normalises before its attention.

    Returns:
        The block in eval mode.
    """
    block = Block(
        channels=24,
        num_heads=2,
        patch_size=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=pre_norm,
        order_index=0,
        cpe_indice_key="stage0",
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enable_conv=False,
        enable_attn=True,
        rope_base=100.0,
    )
    randomize_layer_norms(block)
    return block.eval()


def build_litept_model(
    export_norm1_row_stats: bool = False, patch_sizes: Sequence[int] = (2, 2, 2)
) -> PTv3Segmentation3DModel:
    """Return a small LitePT segmentation model with two attention blocks in its last stage.

    Args:
        export_norm1_row_stats: Whether the exported encoder folds norm1 into qkv.
        patch_sizes: Attention window of every level, the input level first.

    Returns:
        The segmentation model on the CPU, LayerNorm affines randomized.
    """
    torch.manual_seed(0)
    encoder = LitePTEncoder(
        in_channels=SweepSplitVoxelFeatureEncoder.out_channels,
        order=("z",),
        stride=(2, 2),
        enc_depths=(1, 1, 2),
        enc_channels=(12, 24, 24),
        enc_num_head=(1, 2, 2),
        enc_patch_size=tuple(patch_sizes),
        mlp_ratio=2.0,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,
        enable_flash=False,
        enc_conv=(True, True, False),
        enc_attn=(False, False, True),
        enc_rope_base=100.0,
    )
    randomize_layer_norms(encoder)
    return PTv3Segmentation3DModel(
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        encoder=encoder,
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_litept_seg3d_head(),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        grid_size=1.0,
        point_cloud_range=list(POINT_CLOUD_RANGE),
        export_norm1_row_stats=export_norm1_row_stats,
    )


class TestNorm1Fold(unittest.TestCase):
    """The fold of a block's norm1 into its qkv, in PyTorch."""

    def test_row_statistics_normalise_like_layer_norm(self) -> None:
        """Test that the centred rows times the reciprocal standard deviation are LayerNorm's.

        Input: random rows with a large mean, and the row statistics with LayerNorm's epsilon.
        Expected: ``(x - mean) * rstd`` equals ``layer_norm(x)`` without an affine.
        Check: assert_close at fp32 tolerance.
        """
        rows = 3.0 + torch.randn(500, 24)
        rstd = LayerNormRowStatistics(torch.tensor(1e-5), torch.tensor(1.0))(rows)
        self.assertEqual(rstd.shape, (500, 1))
        normalised = (rows - rows.mean(dim=-1, keepdim=True)) * rstd
        torch.testing.assert_close(
            normalised, nn.functional.layer_norm(rows, (24,), eps=1e-5), rtol=1e-5, atol=1e-5
        )

    def test_folded_qkv_of_scaled_rows_equals_qkv_of_layer_norm(self) -> None:
        """Test that ``qkv'(x[order] * rstd[order])`` is ``qkv(LN(x)[order])``.

        Input: an attention block with random LayerNorm affines, random rows and a row order
            with repeats (the attention windows are padded by repeating rows), in float64.
        Expected: the folded block's norm1 leaves the features as they are and attaches the
            row scale; the folded projection of the scaled gathered rows equals the original
            projection of the gathered normalised rows.
        Check: assert_close at float64 rounding.
        """
        block = build_attention_block().double()
        rows = 2.0 + torch.randn(40, 24, dtype=torch.float64)
        order = torch.cat([torch.randperm(40), torch.tensor([3, 3, 7])])
        with torch.no_grad():
            reference = block.attn.qkv(block.norm1(Point({"feat": rows.clone()})).feat[order])
            block.fold_norm1_into_qkv()
            point = block.norm1(Point({"feat": rows.clone()}))
            folded = block.attn.qkv(point.feat[order] * point[QKV_ROW_SCALE_KEY][order])
        self.assertIsInstance(block.norm1._modules["0"], Norm1RowStatistics)
        self.assertTrue(torch.equal(point.feat, rows))
        torch.testing.assert_close(folded, reference, rtol=1e-12, atol=1e-12)

    def test_fold_refuses_blocks_without_a_pre_norm_attention(self) -> None:
        """Test that only a pre-norm attention block can fold its norm1.

        Input: a post-norm attention block, and a folded block folded a second time.
        Expected: ValueError for both.
        Check: assertRaises.
        """
        with self.assertRaises(ValueError):
            build_attention_block(pre_norm=False).fold_norm1_into_qkv()
        block = build_attention_block()
        block.fold_norm1_into_qkv()
        with self.assertRaises(ValueError):
            block.fold_norm1_into_qkv()

    def test_fold_reaches_every_attention_block_and_no_other(self) -> None:
        """Test that the encoder fold changes the attention blocks only.

        Input: the export copy of a LitePT encoder with one convolution-only block in each of
            two stages and two attention blocks in the last.
        Expected: two blocks folded; their norm1 is the row statistics, the other blocks
            keep theirs.
        Check: the returned count and the norm1 module types.
        """
        encoder = build_litept_model().encoder.prepare_for_export(("z",))
        self.assertEqual(fold_norm1_row_statistics(encoder), 2)
        for block in (child for child in encoder.modules() if isinstance(child, Block)):
            if block.attn is None:
                self.assertIsNone(block.norm1)
            else:
                self.assertIsInstance(block.norm1._modules["0"], Norm1RowStatistics)


class TestOption(unittest.TestCase):
    """``export_norm1_row_stats`` on the task model."""

    def test_option_is_off_by_default_litept_only_and_reaches_the_encoder_graph(self) -> None:
        """Test that only a LitePT model folds, only on request, and only in the export copy.

        Input: the LitePT model with the default settings and with the option; the PTv3 test
            model with the option.
        Expected: off by default; with the option the encoder graph's module has row
            statistics in its attention blocks while the model's own encoder keeps its
            LayerNorms; the graphs keep their names, inputs and outputs; PTv3 refuses it.
        Check: the attribute, the norm1 modules and the graphs' names and I/O.
        """
        default = build_litept_model()
        folded = build_litept_model(export_norm1_row_stats=True)
        self.assertFalse(default.export_norm1_row_stats)
        self.assertTrue(folded.export_norm1_row_stats)
        default_graphs = {graph.name: graph for graph in default.deployment_graphs()}
        folded_graphs = {graph.name: graph for graph in folded.deployment_graphs()}

        def norm1_types(module: nn.Module) -> list[type]:
            return [
                type(block.norm1._modules["0"])
                for block in module.modules()
                if isinstance(block, Block) and block.attn is not None
            ]

        self.assertEqual(norm1_types(default_graphs[ENCODER_GRAPH].module), [nn.LayerNorm] * 2)
        self.assertEqual(norm1_types(folded_graphs[ENCODER_GRAPH].module), [Norm1RowStatistics] * 2)
        self.assertEqual(norm1_types(folded.encoder), [nn.LayerNorm] * 2)
        self.assertEqual(set(default_graphs), set(folded_graphs))
        for name, graph in default_graphs.items():
            self.assertEqual(graph.input_names, folded_graphs[name].input_names, name)
            self.assertEqual(graph.output_names, folded_graphs[name].output_names, name)
        with self.assertRaises(ValueError):
            PTv3Segmentation3DModel(
                log_dict_configs=LogDictConfigs(prog_bar=False),
                data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
                encoder=build_seg3d_model().encoder,
                voxel_encoder=SweepSplitVoxelFeatureEncoder(),
                seg3d_head=build_seg3d_model().seg3d_head,
                optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
                grid_size=1.0,
                point_cloud_range=list(POINT_CLOUD_RANGE),
                export_norm1_row_stats=True,
            )


@REQUIRES_SPARSE_CUDA
class TestEncoderExport(unittest.TestCase):
    """The exported LitePT encoder with norm1 as row statistics."""

    def setUp(self) -> None:
        """Build one batch, and the LitePT model without and with the option on CUDA."""
        self.device = torch.device("cuda")
        self.batch = build_inputs(device=self.device)
        self.models = {
            folded: build_litept_model(folded).to(self.device).eval() for folded in (False, True)
        }

    def test_encoder_graph_outputs_match_the_unfolded_export(self) -> None:
        """Test that the folded encoder graph returns the unfolded graph's stage features.

        Input: the encoder graph modules without and with the option, on the same traced
            inputs.
        Expected: every stage feature agrees to fp32 rounding.
        Check: assert_close with 1e-5 tolerances.
        """
        outputs = {}
        for folded, model in self.models.items():
            spec = model.build_export_specs(self.batch)[ENCODER_GRAPH]
            with torch.no_grad():
                outputs[folded] = spec.module(*spec.args)
        self.assertEqual(len(outputs[True]), len(outputs[False]))
        for unfolded_feat, folded_feat in zip(outputs[False], outputs[True], strict=True):
            torch.testing.assert_close(folded_feat, unfolded_feat, rtol=1e-5, atol=1e-5)

    def test_export_replaces_norm1_by_named_row_statistics(self) -> None:
        """Test the exported operators and the names of the statistic nodes.

        Input: the encoder graph without and with the option, exported with the base configs'
            ONNX settings.
        Expected: per attention block, the folded graph has one LayerNormalization fewer and
            the seven statistic nodes, a second Gather and a Mul more; the statistic nodes are
            exactly the nodes under ``/stats_fp32/``, seven per block in operator order, their
            constants initializers shared by the blocks (no Identity), and the graph inputs and
            outputs are unchanged.
        Check: operator counts, node names and the serialized graph inputs and outputs.
        """
        graphs = {}
        with tempfile.TemporaryDirectory() as directory:
            for folded, model in self.models.items():
                path = Path(directory) / f"{folded}.onnx"
                spec = model.build_export_specs(self.batch)[ENCODER_GRAPH]
                export_to_onnx(spec, OmegaConf.create(BASE_ONNX_CONFIG), path)
                graphs[folded] = onnx.load(path)
        blocks = 2
        ops = {
            folded: Counter(node.op_type for node in m.graph.node) for folded, m in graphs.items()
        }
        per_block = Counter(ROW_STATISTIC_OPS) + Counter({"Gather": 1, "Mul": 1})
        self.assertEqual(
            ops[True],
            ops[False]
            - Counter({"LayerNormalization": blocks})
            + Counter({op: blocks * count for op, count in per_block.items()}),
        )
        names = norm1_row_statistic_node_names(graphs[True])
        self.assertEqual(norm1_row_statistic_node_names(graphs[False]), [])
        nodes = {node.name: node for node in graphs[True].graph.node}
        self.assertEqual(len(names), 7 * blocks)
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual([nodes[name].op_type for name in names], ROW_STATISTIC_OPS * blocks)
        for name in names:
            self.assertIn("/norm1/norm1.0/stats_fp32/", name)
        initializers = {tensor.name for tensor in graphs[True].graph.initializer}
        produced = {
            output
            for node in graphs[True].graph.node
            if node.op_type != "Constant"
            for output in node.output
        }
        for name in names:
            for value in nodes[name].input:
                self.assertTrue(value in produced or value in initializers, (name, value))
        self.assertEqual(
            {nodes[name].input[1] for name in names if nodes[name].op_type == "Add"}
            | {nodes[name].input[0] for name in names if nodes[name].op_type == "Div"},
            {name for name in initializers if "stats_fp32" in name},
        )
        self.assertEqual(len({name for name in initializers if "stats_fp32" in name}), 2)
        for field in ("input", "output"):
            self.assertEqual(
                [tensor.SerializeToString() for tensor in getattr(graphs[False].graph, field)],
                [tensor.SerializeToString() for tensor in getattr(graphs[True].graph, field)],
                field,
            )
