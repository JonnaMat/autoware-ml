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

"""Unit tests for the PTv3 deployment graphs."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import onnx
import torch

from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.deployment.graphs import GraphTensor
from autoware_ml.deployment.tests.deployment_assertions import (
    assert_deployment_reproduces_forward,
    assert_specs_export_onnx_with_the_declared_io,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    CLASS_NAMES,
    build_inputs,
    build_litept_seg3d_model,
    build_seg3d_model,
    build_trans_model,
    whole_level_patch_sizes,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET3D_HEAD_GRAPH,
    ENCODER_EXPORT_INPUT_NAMES,
    ENCODER_GRAPH,
    SEG3D_HEAD_GRAPH,
    SEG3D_HEAD_OUTPUT_NAMES,
    encoder_graph_input_names,
    rulebook_names,
    seg3d_head_export_input_names,
    stage_feature_names,
    stage_submanifold_conv,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.deploy import export_to_onnx

REQUIRES_SPARSE_CUDA = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(), "PTv3 export needs spconv and CUDA"
)


class TestDeclarationKeepsTheModelMode(unittest.TestCase):
    """Declaring the graphs changes nothing on the model."""

    def test_declaring_the_graphs_leaves_the_live_voxel_encoder_in_training_mode(self) -> None:
        """Test that declaring the graphs does not switch the module the graphs share with the model.

        Input: the segmentation and the detection model in training mode.
        Expected: the voxel encoder, the one live module the encoder graph wraps, is still in
            training mode after the declaration; the framework switches modes at export only.
        Check: ``voxel_encoder.training`` after ``deployment_declaration()``.
        """
        for model in (build_seg3d_model(), build_trans_model()):
            model.train()
            model.deployment_declaration()
            self.assertTrue(model.voxel_encoder.training, type(model).__name__)


@REQUIRES_SPARSE_CUDA
class TestSeg3dDeployment(unittest.TestCase):
    """The segmentation model's deployment graphs and the export specs derived from them."""

    def setUp(self) -> None:
        """Build the segmentation model and one batch on the GPU."""
        self.model = build_seg3d_model().cuda().eval()
        self.batch = build_inputs(device=torch.device("cuda"))

    def test_seg3d_declaration_and_specs(self) -> None:
        """Test that the segmentation model declares the encoder and the seg3d head graphs.

        Input: the segmentation model's deployment declaration, its export specs and one
            serialized batch.
        Expected: the graphs are the encoder and the seg3d head, named as the runtime names
            them; every graph input is a serialized tensor or an encoder output; the encoder
            spec takes the encoder graph's inputs with the voxel count dynamic, and both specs'
            modules run on their args.
        Check: the graph names, the encoder's leading input names, the head's output names, the
            input names against the serialized tensors, the spec input names, the dynamic axis
            of ``voxels``, and the shapes of the level features and of the head's labels and
            probabilities.
        """
        graphs = {graph.name: graph for graph in self.model.deployment_declaration().graphs}
        self.assertEqual(set(graphs), {ENCODER_GRAPH, SEG3D_HEAD_GRAPH})
        encoder, head = graphs[ENCODER_GRAPH], graphs[SEG3D_HEAD_GRAPH]
        self.assertEqual(encoder.input_names[:5], ENCODER_EXPORT_INPUT_NAMES)
        self.assertFalse(any(name.endswith("_cluster") for name in encoder.input_names))
        self.assertEqual(head.output_names, ("pred_labels", "pred_probs"))
        tensors = self.model.serialize_points(self.batch)
        self.assertLessEqual(set(encoder.input_names), set(tensors))
        self.assertLessEqual(set(head.input_names), set(tensors) | set(encoder.output_names))

        specs = self.model.build_export_specs(self.batch)
        self.assertEqual(set(specs), set(graphs))
        encoder_spec, head_spec = specs[ENCODER_GRAPH], specs[SEG3D_HEAD_GRAPH]
        self.assertEqual(
            encoder_spec.input_names,
            list(encoder_graph_input_names(len(self.model.encoder.stride))),
        )
        self.assertEqual(encoder_spec.dynamic_axes["voxels"], {0: "num_voxels"})
        with torch.no_grad():
            level_feats = encoder_spec.module(*encoder_spec.args)
            pred_labels, pred_probs = head_spec.module(*head_spec.args)
        num_voxels = self.batch.voxels_data.voxels.shape[0]
        self.assertEqual(len(level_feats), len(self.model.encoder.stride) + 1)
        self.assertEqual(pred_labels.shape, (num_voxels,))
        self.assertEqual(pred_probs.shape[0], num_voxels)

    def test_seg3d_specs_export_onnx_with_the_declared_io_contract(self) -> None:
        """Test that each derived spec exports an ONNX with the graph's declared names and axes.

        Input: the segmentation model's specs, exported with the base config's ONNX settings.
        Expected: each ONNX carries the graph's input and output names and its dynamic axes,
            the seg3d head's outputs included.
        Check: the seg3d head's outputs among its declared axes, then
            assert_specs_export_onnx_with_the_declared_io().
        """
        graphs = {graph.name: graph for graph in self.model.deployment_graphs()}
        self.assertLessEqual(
            set(SEG3D_HEAD_OUTPUT_NAMES), set(graphs[SEG3D_HEAD_GRAPH].onnx_dynamic_axes)
        )
        assert_specs_export_onnx_with_the_declared_io(self.model, self.batch)


@REQUIRES_SPARSE_CUDA
class TestDet3dDeployment(unittest.TestCase):
    """The detection model's deployment graphs and their specs."""

    def setUp(self) -> None:
        """Build the detection model and one batch on the GPU."""
        self.model = build_trans_model().cuda().eval()
        self.batch = build_inputs(device=torch.device("cuda"))

    def test_det3d_declaration_and_specs(self) -> None:
        """Test that the detection model declares the encoder and the det3d head graphs.

        Input: the detection model's deployment declaration and its export specs.
        Expected: the graphs and the specs are the encoder and the det3d head; the head spec
            emits the configured export output names, one tensor each, the labels as int64 and
            the dense heatmap as one map per class.
        Check: the graph names, the spec names, the head spec's output names, and the count,
            the ``query_labels`` dtype and the ``dense_heatmap`` leading dims of a run.
        """
        graphs = {graph.name: graph for graph in self.model.deployment_declaration().graphs}
        self.assertEqual(set(graphs), {ENCODER_GRAPH, DET3D_HEAD_GRAPH})
        specs = self.model.build_export_specs(self.batch)
        self.assertEqual(set(specs), set(graphs))
        head_spec = specs[DET3D_HEAD_GRAPH]
        self.assertEqual(head_spec.output_names, list(self.model.bbox_head.export_output_names))
        with torch.no_grad():
            outputs = dict(zip(head_spec.output_names, head_spec.module(*head_spec.args)))
        self.assertEqual(len(outputs), len(head_spec.output_names))
        self.assertEqual(outputs["query_labels"].dtype, torch.int64)
        self.assertEqual(outputs["dense_heatmap"].shape[:2], (1, len(CLASS_NAMES)))

    def test_det3d_specs_export_onnx_with_the_declared_io_contract(self) -> None:
        """Test that the encoder and the det3d head specs export ONNX with their graphs' names.

        Input: the detection model's specs, exported with the base config's ONNX settings.
        Expected: each ONNX carries the graph's input and output names and its dynamic axes.
        Check: assert_specs_export_onnx_with_the_declared_io().
        """
        assert_specs_export_onnx_with_the_declared_io(self.model, self.batch)


@REQUIRES_SPARSE_CUDA
class TestDeploymentReproducesForward(unittest.TestCase):
    """The deployed PTv3 models against forward() on one batch."""

    def setUp(self) -> None:
        """Build one batch and the windows under which forward() attends like the export."""
        self.device = torch.device("cuda")
        self.batch = build_inputs(device=self.device)
        self.patch_sizes = whole_level_patch_sizes(self.batch)

    @staticmethod
    def _seg3d_tensors(outputs: ModelOutputs) -> dict[str, GraphTensor]:
        """Return forward()'s labels and probabilities under the seg3d head graph's names.

        Args:
            outputs: The outputs of ``forward()``.

        Returns:
            The label and the class probabilities of every voxel.
        """
        probs = torch.softmax(outputs.segmentation3d().logits, dim=1)
        return {"pred_labels": probs.argmax(dim=1), "pred_probs": probs}

    @staticmethod
    def _det3d_tensors(outputs: ModelOutputs, names: Sequence[str]) -> dict[str, GraphTensor]:
        """Return forward()'s detection head outputs under the det3d head graph's names.

        Args:
            outputs: The outputs of ``forward()``.
            names: The det3d head graph's output names.

        Returns:
            The head outputs by name.
        """
        head_outputs = outputs.detection3d().transfusion_head()
        return dict(zip(names, head_outputs.export_tensors(names)))

    def test_seg3d_deployment_reproduces_forward(self) -> None:
        """Test that the segmentation deployment reproduces forward().

        Input: a segmentation model whose window at every level is that level's voxel count,
            and one batch.
        Expected: the deployed labels equal forward()'s and the probabilities match to 1e-5.
        Check: assert_deployment_reproduces_forward().
        """
        model = build_seg3d_model(patch_sizes=self.patch_sizes).cuda().eval()
        assert_deployment_reproduces_forward(
            model, self.batch, self.device, self._seg3d_tensors, rtol=1e-5, atol=1e-5
        )

    def test_det3d_deployment_reproduces_forward(self) -> None:
        """Test that the detection deployment reproduces forward().

        Input: a detection model whose window at every level is that level's voxel count, and
            one batch.
        Expected: every head output of the deployment matches forward()'s to 1e-5.
        Check: assert_deployment_reproduces_forward().
        """
        model = build_trans_model(patch_sizes=self.patch_sizes).cuda().eval()
        graphs = {graph.name: graph for graph in model.deployment_graphs()}
        names = graphs[DET3D_HEAD_GRAPH].output_names
        assert_deployment_reproduces_forward(
            model,
            self.batch,
            self.device,
            partial(self._det3d_tensors, names=names),
            rtol=1e-5,
            atol=1e-5,
        )


def _graph_io(model) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return every declared graph's input and output names, by graph name."""
    return {
        graph.name: (tuple(graph.input_names), tuple(graph.output_names))
        for graph in model.deployment_graphs()
    }


class TestSeg3dRulebookHandoffDeclaration(unittest.TestCase):
    """The ``export_seg3d_rulebook_handoff`` option on the declared graphs."""

    def test_handoff_is_off_by_default_and_changes_only_the_rulebook_interface(self) -> None:
        """Test that the option is off by default and, once set, adds only the rulebooks.

        Input: the segmentation model (one decoder block on stage 0, whose attention reads no
            coordinates) with the default option, then with it set.
        Expected: off by default, no hand-off stage; once set, stage 0 is handed off: the
            encoder returns the stage-0 rulebook after the features, the seg3d head takes it
            after its other inputs and no longer takes ``grid_coord``, which only fed the
            rulebook; every other name is unchanged; every axis of a rulebook is dynamic under
            a name of its own (the plugin's layout is not PyTorch's).
        Check: the option, ``export_rulebook_stages()``, the graphs' input and output names and
            dynamic axes.
        """
        model = build_seg3d_model()
        self.assertFalse(model.export_seg3d_rulebook_handoff)
        self.assertEqual(model.export_rulebook_stages(), ())
        reference = _graph_io(model)
        model.export_seg3d_rulebook_handoff = True
        self.assertEqual(model.export_rulebook_stages(), (0,))
        graphs = {graph.name: graph for graph in model.deployment_graphs()}
        rulebooks = rulebook_names([0])
        self.assertEqual(
            rulebooks,
            ["rulebook_enc0_pairs", "rulebook_enc0_point_mask", "rulebook_enc0_mask_argsort"],
        )
        encoder_inputs, encoder_outputs = reference[ENCODER_GRAPH]
        self.assertEqual(graphs[ENCODER_GRAPH].input_names, encoder_inputs)
        self.assertEqual(graphs[ENCODER_GRAPH].output_names, (*encoder_outputs, *rulebooks))
        head_inputs, head_outputs = reference[SEG3D_HEAD_GRAPH]
        self.assertEqual(
            graphs[SEG3D_HEAD_GRAPH].input_names,
            (*(name for name in head_inputs if name != "grid_coord"), *rulebooks),
        )
        self.assertEqual(graphs[SEG3D_HEAD_GRAPH].output_names, head_outputs)
        for graph in (ENCODER_GRAPH, SEG3D_HEAD_GRAPH):
            axes = graphs[graph].onnx_dynamic_axes
            for name, rank in zip(rulebooks, (2, 2, 1), strict=True):
                self.assertEqual(
                    axes[name], {axis: f"{name}_dim_{axis}" for axis in range(rank)}, graph
                )
        self.assertNotIn("grid_coord", graphs[SEG3D_HEAD_GRAPH].onnx_dynamic_axes)

    def test_handoff_leaves_models_without_a_decoder_rulebook_unchanged(self) -> None:
        """Test that the option changes nothing where the seg3d head builds no rulebook.

        Input: the detection model (no seg3d head) and the LitePT segmentation model (no
            decoder blocks), with and without the option.
        Expected: no hand-off stage and the same graphs' names either way.
        Check: ``export_rulebook_stages()`` and the graphs' input and output names.
        """
        for model in (build_trans_model(), build_litept_seg3d_model()):
            reference = _graph_io(model)
            model.export_seg3d_rulebook_handoff = True
            self.assertEqual(model.export_rulebook_stages(), (), type(model).__name__)
            self.assertEqual(_graph_io(model), reference, type(model).__name__)

    def test_a_handed_off_stage_keeps_the_coordinates_its_blocks_read(self) -> None:
        """Test the input rule for a stage whose attention reads the voxel coordinates.

        Input: the seg3d head input rule for one decoder block stage handed off, with and
            without ``grid_coord`` (RoPE or RPE in its blocks keep it), and a coordinate-free
            stage that is not handed off.
        Expected: ``grid_coord`` stays unless the stage is coordinate-free; the rulebooks
            follow the other inputs; dropping the coordinates of a stage without a rulebook is
            refused.
        Check: the returned names and the ValueError.
        """
        base = seg3d_head_export_input_names(2, [1])
        self.assertEqual(
            seg3d_head_export_input_names(2, [1], rulebook_stages=[0]), base + rulebook_names([0])
        )
        self.assertEqual(
            seg3d_head_export_input_names(2, [1], [0], coordinate_free_stages=[0]),
            [name for name in base if name != "grid_coord"] + rulebook_names([0]),
        )
        with self.assertRaises(ValueError):
            seg3d_head_export_input_names(2, [1], coordinate_free_stages=[0])


@REQUIRES_SPARSE_CUDA
class TestSeg3dRulebookHandoff(unittest.TestCase):
    """The seg3d head consuming the encoder's rulebooks, against rebuilding them."""

    def setUp(self) -> None:
        """Build one batch and two segmentation models with the same weights, one handing off."""
        self.device = torch.device("cuda")
        self.batch = build_inputs(device=self.device)
        torch.manual_seed(0)
        self.reference = build_seg3d_model().cuda().eval()
        self.model = build_seg3d_model().cuda().eval()
        self.model.load_state_dict(self.reference.state_dict())
        self.model.export_seg3d_rulebook_handoff = True

    def test_head_on_the_encoders_rulebooks_matches_the_head_rebuilding_them(self) -> None:
        """Test that the handed-off rulebooks are the head's own and give the same outputs.

        Input: the export specs of both models on one batch.
        Expected: the encoder returns the same features, then the stage-0 rulebook as int32
            tensors bitwise equal to the rulebook the reference head builds for its stage-0
            convolution; the head fed them returns bitwise the reference head's outputs.
        Check: the trace args and outputs of the specs; the reference head's rulebook is read
            from the indice cache of its convolution's output.
        """
        reference_specs = self.reference.build_export_specs(self.batch)
        specs = self.model.build_export_specs(self.batch)
        stage_count = len(self.model.encoder.stride) + 1
        reference_head = reference_specs[SEG3D_HEAD_GRAPH].module
        conv = stage_submanifold_conv(reference_head.seg3d_head.dec._modules["dec0"])
        built = {}

        def keep_rulebook(_module, _inputs, output) -> None:
            rulebook = output.find_indice_pair(conv.indice_key)
            built["rulebook"] = (
                rulebook.pair_fwd,
                rulebook.pair_mask_fwd_splits,
                rulebook.mask_argsort_fwd_splits,
            )

        handle = conv.register_forward_hook(keep_rulebook)
        try:
            with torch.no_grad():
                reference_feats = reference_specs[ENCODER_GRAPH].module(
                    *reference_specs[ENCODER_GRAPH].args
                )
                encoder_outputs = specs[ENCODER_GRAPH].module(*specs[ENCODER_GRAPH].args)
                reference_outputs = reference_head(*reference_specs[SEG3D_HEAD_GRAPH].args)
                head_spec = specs[SEG3D_HEAD_GRAPH]
                outputs = head_spec.module(*head_spec.args)
        finally:
            handle.remove()

        self.assertEqual(len(encoder_outputs), stage_count + 3)
        for feat, reference_feat in zip(
            encoder_outputs[:stage_count], reference_feats, strict=True
        ):
            self.assertTrue(torch.equal(feat, reference_feat))
        handed = dict(zip(specs[ENCODER_GRAPH].output_names, encoder_outputs))
        for name, tensor in zip(rulebook_names([0]), built["rulebook"], strict=True):
            self.assertEqual(handed[name].dtype, torch.int32, name)
            self.assertTrue(torch.equal(handed[name], tensor), name)
            arg = head_spec.args[head_spec.input_names.index(name)]
            self.assertTrue(torch.equal(arg, handed[name]), name)
        self.assertEqual(head_spec.input_names[:stage_count], stage_feature_names(stage_count))
        for output, reference_output in zip(outputs, reference_outputs, strict=True):
            self.assertTrue(torch.equal(output, reference_output))

    def test_deployment_with_the_handoff_reproduces_forward(self) -> None:
        """Test that the deployment chaining the rulebooks reproduces forward().

        Input: a handing-off segmentation model whose window at every level is that level's
            voxel count, and one batch.
        Expected: the deployed labels equal forward()'s and the probabilities match to 1e-5.
        Check: assert_deployment_reproduces_forward().
        """
        model = build_seg3d_model(patch_sizes=whole_level_patch_sizes(self.batch)).cuda().eval()
        model.export_seg3d_rulebook_handoff = True
        assert_deployment_reproduces_forward(
            model,
            self.batch,
            self.device,
            TestDeploymentReproducesForward._seg3d_tensors,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_exported_head_reads_the_rulebooks_instead_of_building_them(self) -> None:
        """Test the exported ONNX of the hand-off against the default export.

        Input: both models' specs exported to ONNX with the base config's settings.
        Expected: the graphs carry their declared names and axes; the encoder graph is the
            default one plus three INT32 outputs, which are outputs 1, 2 and 3 of its stage-0
            ``GetIndicePairsImplicitGemm``; the hand-off head has no
            ``GetIndicePairsImplicitGemm`` (the default head has one) and its ``ImplicitGemm``
            reads the three inputs in slots 2, 3 and 4, as the encoder's does.
        Check: assert_specs_export_onnx_with_the_declared_io(), then the ONNX nodes and I/O.
        """
        assert_specs_export_onnx_with_the_declared_io(self.model, self.batch)
        onnx_cfg = OmegaConf.create(
            {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
        )
        exported = {}
        with tempfile.TemporaryDirectory() as directory:
            for tag, model in (("reference", self.reference), ("handoff", self.model)):
                for name, spec in model.build_export_specs(self.batch).items():
                    path = Path(directory) / f"{tag}_{name}.onnx"
                    export_to_onnx(spec, onnx_cfg, path)
                    exported[tag, name] = onnx.load(path).graph

        def op_count(graph: onnx.GraphProto, op_type: str) -> int:
            return sum(node.op_type == op_type for node in graph.node)

        rulebooks = rulebook_names([0])
        encoder = exported["handoff", ENCODER_GRAPH]
        reference_encoder = exported["reference", ENCODER_GRAPH]
        self.assertEqual(len(encoder.node), len(reference_encoder.node))
        self.assertEqual([o.name for o in encoder.output][-3:], rulebooks)
        for output in encoder.output[-3:]:
            self.assertEqual(output.type.tensor_type.elem_type, onnx.TensorProto.INT32)
        (encoder_gip,) = [
            node
            for node in encoder.node
            if node.op_type == "GetIndicePairsImplicitGemm" and "enc0" in node.name
        ]
        self.assertEqual(list(encoder_gip.output[1:4]), rulebooks)

        head = exported["handoff", SEG3D_HEAD_GRAPH]
        self.assertEqual(
            op_count(exported["reference", SEG3D_HEAD_GRAPH], "GetIndicePairsImplicitGemm"), 1
        )
        self.assertEqual(op_count(head, "GetIndicePairsImplicitGemm"), 0)
        for name in rulebooks:
            (value,) = [i for i in head.input if i.name == name]
            self.assertEqual(value.type.tensor_type.elem_type, onnx.TensorProto.INT32)
        gemms = [node for node in head.node if node.op_type == "ImplicitGemm"]
        self.assertTrue(gemms)
        for node in gemms:
            self.assertEqual(list(node.input[2:5]), rulebooks)
