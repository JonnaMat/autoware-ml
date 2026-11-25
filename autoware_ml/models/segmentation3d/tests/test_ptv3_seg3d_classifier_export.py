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

"""The seg3d head export applies the classifier before the finest unpooling's gather."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import onnx
from omegaconf import OmegaConf
import torch
from torch import nn

from autoware_ml.deployment.graphs import export_spec_for_graph
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_litept_seg3d_model,
    build_seg3d_head,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import SerializedUnpooling
from autoware_ml.models.segmentation3d.heads.ptv3 import PTv3Seg3DDecoderHead
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3Seg3DHeadExportModule,
    link_stage_points,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.deploy import export_to_onnx
from autoware_ml.utils.point_cloud.structures import Point

ENC_CHANNELS = (8, 16, 32)
DEC_CHANNELS = (24, 16)
NUM_CLASSES = 5
STAGE_VOXELS = (40, 17, 6)


def _jitter_affine_parameters(module: nn.Module) -> None:
    """Give every bias and batch norm a non-trivial value, as training would."""
    for child in module.modules():
        if isinstance(child, nn.BatchNorm1d):
            nn.init.uniform_(child.weight, 0.5, 1.5)
            nn.init.uniform_(child.running_mean, -0.5, 0.5)
            nn.init.uniform_(child.running_var, 0.5, 1.5)
        if getattr(child, "bias", None) is not None:
            nn.init.uniform_(child.bias, -0.5, 0.5)


def _build_head(
    enc_channels: tuple[int, ...] = ENC_CHANNELS,
    dec_depths: tuple[int, ...] = (0, 0),
    dec_channels: tuple[int, ...] = DEC_CHANNELS,
    dec_patch_size: tuple[int, ...] = (4, 4),
) -> PTv3Seg3DDecoderHead:
    """Return a two-level decoder head with non-trivial batch norms and biases."""
    head = PTv3Seg3DDecoderHead(
        num_classes=NUM_CLASSES,
        ignore_index=-1,
        order=("z",),
        enc_channels=enc_channels,
        dec_depths=dec_depths,
        dec_channels=dec_channels,
        dec_num_head=(1, 1),
        dec_patch_size=dec_patch_size,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        mixed_voxel_weight=0.0,
    )
    _jitter_affine_parameters(head)
    return head


def _build_blockless_head() -> PTv3Seg3DDecoderHead:
    """Return an eval-mode two-level decoder head without blocks, so it runs on the CPU."""
    torch.manual_seed(0)
    return _build_head().eval()


def _stage_inputs() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return random per-stage features and pooling clusters for the blockless head."""
    generator = torch.Generator().manual_seed(1)
    feats = [
        torch.randn(count, channels, generator=generator) * 2.0
        for count, channels in zip(STAGE_VOXELS, ENC_CHANNELS, strict=True)
    ]
    clusters = [
        torch.randint(0, coarse, (fine,), generator=generator)
        for fine, coarse in zip(STAGE_VOXELS[:-1], STAGE_VOXELS[1:], strict=True)
    ]
    return feats, clusters


class TestUnpoolingDeferredFusion(unittest.TestCase):
    """SerializedUnpooling.defer_fusion leaves the fusion to the module after it."""

    def test_deferred_fusion_reproduces_the_fused_features(self) -> None:
        """
        Input: an eval-mode SerializedUnpooling (16 input, 8 skip, 12 output channels) with
        non-trivial batch norms and biases, and a coarse level of 7 points unpooled to 30
        parent points through a random cluster, in fp32, with and without ``defer_fusion``.
        Expected: the deferred parent keeps the projected skip features and carries the
        projected coarse features and the inverse, whose fusion is the regular output.
        Check: ``feat + unpooling_feat[unpooling_inverse]`` of the deferred parent equals the
        regular parent's ``feat`` bitwise, and the regular parent carries neither key.
        """
        torch.manual_seed(2)
        unpooling = SerializedUnpooling(16, 8, 12)
        _jitter_affine_parameters(unpooling)
        unpooling.eval()
        coarse_feat = torch.randn(7, 16) * 2.0
        skip_feat = torch.randn(30, 8) * 2.0
        cluster = torch.randint(0, 7, (30,))

        def coarse_point() -> Point:
            point = Point(feat=coarse_feat.clone())
            point["pooling_parent"] = Point(feat=skip_feat.clone())
            point["pooling_inverse"] = cluster
            return point

        with torch.no_grad():
            fused = unpooling(coarse_point())
            unpooling.defer_fusion = True
            deferred = unpooling(coarse_point())
        self.assertNotIn("unpooling_feat", fused)
        self.assertNotIn("unpooling_inverse", fused)
        self.assertTrue(torch.equal(deferred["unpooling_inverse"], cluster))
        refused = deferred.feat + deferred["unpooling_feat"][deferred["unpooling_inverse"]]
        self.assertTrue(torch.equal(refused, fused.feat))


class TestSeg3dHeadClassifierBeforeUnpooling(unittest.TestCase):
    """The export copy of the seg3d head classifies before the finest unpooling's gather."""

    def test_only_a_blockless_finest_stage_enables_the_rewrite(self) -> None:
        """
        Input: a head whose finest decoder stage has no blocks and one whose finest stage has
        a block, both prepared for export.
        Expected: only the first can move the classifier in front of the finest gather (the
        second's blocks sit between the gather and the classifier); the live heads keep the
        regular forward.
        Check: ``defer_fusion`` of the finest unpooling of the export copies and of the live
        heads, and of the coarser unpooling.
        """
        blockless = _build_blockless_head()
        with_block = build_seg3d_head(dec_depths=(1,))
        blockless_export = blockless.prepare_for_export(["z"])
        self.assertTrue(blockless_export.dec.dec0.up.defer_fusion)
        self.assertFalse(blockless_export.dec.dec1.up.defer_fusion)
        self.assertFalse(with_block.prepare_for_export(["z"]).dec.dec0.up.defer_fusion)
        self.assertFalse(blockless.dec.dec0.up.defer_fusion)
        self.assertFalse(with_block.dec.dec0.up.defer_fusion)

    def test_export_module_matches_the_eager_head(self) -> None:
        """
        Input: an eval-mode blockless two-level head with non-trivial batch norms and biases,
        random stage features (40, 17 and 6 voxels) and pooling clusters, in fp32.
        Expected: the export module computes the head's logits with the classifier moved in
        front of the finest gather, an exact rewrite up to fp32 rounding, and leaves the head
        and its parameters unchanged.
        Check: its probabilities match the softmax of the eager head's logits within 1e-5, its
        labels are their argmax, and the head's state dict is unchanged.
        """
        head = _build_blockless_head()
        state = {key: value.clone() for key, value in head.state_dict().items()}
        feats, clusters = _stage_inputs()
        module = PTv3Seg3DHeadExportModule(
            head, ["z"], torch.tensor([64, 64, 64]), strides=(2, 2)
        ).eval()
        with torch.no_grad():
            reference = torch.softmax(head(link_stage_points(feats, clusters)), dim=1)
            labels, probs = module(*feats, *clusters)
        torch.testing.assert_close(probs, reference, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(labels, probs.argmax(dim=1))
        for key, value in head.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_exported_graph_gathers_class_logits_at_the_finest_stage(self) -> None:
        """
        Input: the blockless head's export module, exported to ONNX as the deploy configs do
        (TorchScript exporter, opset 17, no constant folding), with and without the rewrite.
        Expected: with the rewrite, the gather through ``pooling_cluster_0`` moves the
        ``num_classes``-channel logits of the coarse level instead of the decoder-width
        features; the classifier runs once per level (with the bias on the finest one only),
        so the graph gains one bias-free matrix product and is otherwise the same.
        Check: the channel count of the tensor each graph gathers through
        ``pooling_cluster_0`` (from shape inference), the node op types the rewrite adds and
        removes, and the wiring: the gather reads the bias-free product of the coarse level
        with the classifier weight, and the Add sums it with the finest level's classifier
        Gemm, which carries the bias.
        """
        head = _build_blockless_head()
        feats, clusters = _stage_inputs()
        args = (*feats, *clusters)
        onnx_cfg = OmegaConf.create(
            {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
        )
        rewritten = PTv3Seg3DHeadExportModule(head, ["z"], torch.tensor([64, 64, 64]), (2, 2))
        reference = deepcopy(rewritten)
        reference.seg3d_head.dec.dec0.up.defer_fusion = False

        with tempfile.TemporaryDirectory() as tmp:
            graphs = {}
            for name, module in (("rewritten", rewritten), ("reference", reference)):
                path = Path(tmp) / f"{name}.onnx"
                spec = export_spec_for_graph(module.eval().build_deployment_graph(), args)
                export_to_onnx(spec, onnx_cfg, path)
                graphs[name] = onnx.shape_inference.infer_shapes(onnx.load(path)).graph

        def gathered_channels(graph: onnx.GraphProto) -> list[int]:
            shapes = {
                info.name: [dim.dim_value for dim in info.type.tensor_type.shape.dim]
                for info in graph.value_info
            }
            return [
                shapes[node.input[0]][-1]
                for node in graph.node
                if node.op_type == "Gather" and node.input[1] == "pooling_cluster_0"
            ]

        self.assertEqual(gathered_channels(graphs["rewritten"]), [NUM_CLASSES])
        self.assertEqual(gathered_channels(graphs["reference"]), [DEC_CHANNELS[0]])
        rewritten_ops = Counter(node.op_type for node in graphs["rewritten"].node)
        reference_ops = Counter(node.op_type for node in graphs["reference"].node)
        # The bias-free product exports as MatMul over the transposed classifier weight.
        self.assertEqual(rewritten_ops - reference_ops, Counter({"MatMul": 1, "Transpose": 1}))
        self.assertEqual(reference_ops - rewritten_ops, Counter())
        nodes = list(graphs["rewritten"].node)
        producers = {output: node for node in nodes for output in node.output}
        (gather,) = [node for node in nodes if node.input[1:] == ["pooling_cluster_0"]]
        coarse_product = producers[gather.input[0]]
        self.assertEqual(coarse_product.op_type, "MatMul")
        transpose = producers[coarse_product.input[1]]
        self.assertEqual(transpose.op_type, "Transpose")
        weight = transpose.input[0]
        initializers = {tensor.name for tensor in graphs["rewritten"].initializer}
        self.assertIn(weight, initializers)
        (skip_product,) = [
            node for node in nodes if node.op_type == "Gemm" and node.input[1] == weight
        ]
        self.assertEqual(len(skip_product.input), 3)
        (fusion,) = [node for node in nodes if gather.output[0] in node.input]
        self.assertEqual(fusion.op_type, "Add")
        self.assertEqual(set(fusion.input), {skip_product.output[0], gather.output[0]})

    @unittest.skipUnless(
        IS_SPCONV_AVAILABLE and torch.cuda.is_available(), "PTv3 export needs spconv and CUDA"
    )
    def test_export_with_a_coarser_block_stage_matches_the_regular_head(self) -> None:
        """
        Input: a small LitePT segmentation model whose head has a decoder block on the coarser
        level and none on the finest, with non-trivial batch norms and biases, and its seg3d
        head export spec built from one batch on the GPU, in fp32.
        Expected: the coarser level decodes as before and only the finest unpooling and the
        classifier are rewritten, an exact rewrite up to fp32 rounding.
        Check: the export module's probabilities match a copy of it with the rewrite turned
        off within 1e-5, and its labels are their argmax.
        """
        torch.manual_seed(0)
        device = torch.device("cuda")
        model = build_litept_seg3d_model()
        model.seg3d_head = _build_head(
            enc_channels=(12, 24, 24),
            dec_depths=(0, 1),
            dec_channels=(12, 24),
            dec_patch_size=(2, 2),
        )
        _jitter_affine_parameters(model)
        model = model.to(device).eval()
        spec = model.build_export_specs(build_inputs(device=device))["ptv3_seg3d_head"]
        self.assertTrue(spec.module.seg3d_head.dec.dec0.up.defer_fusion)
        reference = deepcopy(spec.module)
        reference.seg3d_head.dec.dec0.up.defer_fusion = False
        with torch.no_grad():
            labels, probs = spec.module(*spec.args)
            _, reference_probs = reference(*spec.args)
        torch.testing.assert_close(probs, reference_probs, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(labels, probs.argmax(dim=1))


if __name__ == "__main__":
    unittest.main()
