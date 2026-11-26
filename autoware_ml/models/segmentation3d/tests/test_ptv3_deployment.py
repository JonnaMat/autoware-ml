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

from collections import Counter
from collections.abc import Sequence
from functools import partial
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import onnx
import onnxruntime
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
    argmax_by_comparison_fold,
    encoder_graph_input_names,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.deploy import ExportSpec, export_to_onnx

REQUIRES_SPARSE_CUDA = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(), "PTv3 export needs spconv and CUDA"
)
#: The export settings of the PTv3 base configs.
BASE_ONNX_CONFIG = {"opset_version": 17, "dynamo": False, "do_constant_folding": False}


def label_fold_op_counts(num_classes: int) -> Counter[str]:
    """Return the ONNX operators :func:`argmax_by_comparison_fold` exports to.

    Args:
        num_classes: Number of score columns.

    Returns:
        One ``Split``, a ``Greater`` per column after the first, a value and an index ``Where``
        per column after the first except the last column's value, and one ``Squeeze``.
    """
    return Counter(
        {"Split": 1, "Greater": num_classes - 1, "Where": 2 * num_classes - 3, "Squeeze": 1}
    )


def fold_test_scores(num_classes: int) -> dict[str, torch.Tensor]:
    """Return fp32 score matrices that stress the fold's tie and extreme-value handling.

    Args:
        num_classes: Number of score columns.

    Returns:
        Scores by case name: softmax probabilities, coarse grids with many ties, signed zeros,
        all-equal rows, fp16 extremes with infinities and subnormals, and every one-hot row
        and every pair of columns tied at the maximum.
    """
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(20000, num_classes, generator=generator) * 4
    grid = torch.randint(0, 3, (20000, num_classes), generator=generator).float()
    signs = torch.where(torch.rand(grid.shape, generator=generator) < 0.5, -1.0, 1.0)
    fp16 = torch.finfo(torch.float16)
    extremes = torch.tensor(
        [fp16.max, -fp16.max, fp16.tiny, 2.0**-24, -(2.0**-24), 0.0, -0.0, 1.0, 1.0 + 2**-10]
        + [float("inf"), float("-inf")]
    )
    pairs = torch.zeros(num_classes * num_classes, num_classes)
    for first in range(num_classes):
        for second in range(num_classes):
            pairs[first * num_classes + second, [first, second]] = 1.0
    return {
        "softmax": torch.softmax(logits, dim=1),
        "softmax rounded to fp16": torch.softmax(logits, dim=1).half().float(),
        "grid with ties": torch.randint(0, 4, (20000, num_classes), generator=generator) / 4.0,
        "signed zeros": grid * signs,
        "all-equal rows": torch.randn(100, 1, generator=generator).expand(-1, num_classes),
        "fp16 extremes": extremes[
            torch.randint(0, len(extremes), (20000, num_classes), generator=generator)
        ],
        "one-hot and tied pairs": torch.cat([torch.eye(num_classes), pairs]),
    }


class TestArgmaxByComparisonFold(unittest.TestCase):
    """The seg3d head's label fold against ``argmax`` and its exported graph."""

    def test_fold_equals_argmax_on_ties_and_extremes(self) -> None:
        """Test that the fold returns argmax's first maximal index, in fp32 and in fp16.

        Input: score matrices of 16, 3 and 2 columns with ties at the maximum, signed zeros,
            infinities and subnormals, in fp32 and cast to fp16.
        Expected: the fold's labels are int64 and equal ``argmax(dim=1)``, which returns the
            first maximal index.
        Check: torch.equal on the labels.
        """
        for num_classes in (16, 3, 2):
            for case, scores in fold_test_scores(num_classes).items():
                for dtype in (torch.float32, torch.float16):
                    with self.subTest(num_classes=num_classes, case=case, dtype=dtype):
                        typed = scores.to(dtype)
                        labels = argmax_by_comparison_fold(typed)
                        self.assertEqual(labels.dtype, torch.int64)
                        self.assertTrue(torch.equal(labels, typed.argmax(dim=1)))

    def test_fold_of_one_class_is_argmax(self) -> None:
        """Test that a single score column labels every row 0.

        Input: a ``(5, 1)`` score matrix.
        Expected: five zero labels of shape ``(5,)``.
        Check: torch.equal against zeros.
        """
        labels = argmax_by_comparison_fold(torch.randn(5, 1))
        self.assertTrue(torch.equal(labels, torch.zeros(5, dtype=torch.int64)))

    def test_fold_exports_without_argmax_and_runs_like_it(self) -> None:
        """Test that the fold exports as elementwise operators that compute ArgMax's labels.

        Input: the fold over 16 classes, exported with the PTv3 base configs' ONNX settings
            and a dynamic row count, run on onnxruntime's CPU provider.
        Expected: apart from constants the graph is the fold's operators, no ArgMax; on the
            stress scores the ONNX labels equal argmax.
        Check: the operator counts of the graph and the labels of every case.
        """
        num_classes = 16

        class Fold(torch.nn.Module):
            def forward(self, scores: torch.Tensor) -> torch.Tensor:
                return argmax_by_comparison_fold(scores)

        spec = ExportSpec(
            module=Fold(),
            args=(torch.randn(10, num_classes),),
            input_names=["scores"],
            output_names=["labels"],
            dynamic_axes={"scores": {0: "n"}, "labels": {0: "n"}},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold.onnx"
            export_to_onnx(spec, OmegaConf.create(BASE_ONNX_CONFIG), path)
            graph = onnx.load(path).graph
            session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ops = Counter(node.op_type for node in graph.node if node.op_type != "Constant")
        self.assertEqual(ops, label_fold_op_counts(num_classes))
        for case, scores in fold_test_scores(num_classes).items():
            with self.subTest(case=case):
                (labels,) = session.run(None, {"scores": scores.contiguous().numpy()})
                self.assertTrue(torch.equal(torch.from_numpy(labels), scores.argmax(dim=1)))

    def test_option_is_off_by_default_and_reaches_the_seg3d_head_graph(self) -> None:
        """Test that the segmentation model folds its exported labels only on request.

        Input: the segmentation model with the default settings and with
            ``export_seg3d_label_fold``.
        Expected: off by default; once set, the seg3d head graph's module folds and the
            graphs keep their names, inputs and outputs.
        Check: ``label_fold`` of the seg3d head graph's module and the graphs' I/O names.
        """
        default = build_seg3d_model()
        folded = build_seg3d_model(export_seg3d_label_fold=True)
        self.assertFalse(default.export_seg3d_label_fold)
        default_graphs = {graph.name: graph for graph in default.deployment_graphs()}
        folded_graphs = {graph.name: graph for graph in folded.deployment_graphs()}
        self.assertFalse(default_graphs[SEG3D_HEAD_GRAPH].module.label_fold)
        self.assertTrue(folded_graphs[SEG3D_HEAD_GRAPH].module.label_fold)
        self.assertEqual(set(default_graphs), set(folded_graphs))
        for name, graph in default_graphs.items():
            self.assertEqual(graph.input_names, folded_graphs[name].input_names, name)
            self.assertEqual(graph.output_names, folded_graphs[name].output_names, name)


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

    def test_seg3d_head_exports_the_label_fold_instead_of_argmax(self) -> None:
        """Test that the option replaces only the head's ArgMax and keeps the graph I/O.

        Input: the seg3d head spec of the default model and of one with
            ``export_seg3d_label_fold``, exported with the base configs' ONNX settings.
        Expected: the default head has one ArgMax; the folded head has none, the fold's
            operators in its place, the same other operators, and the same graph inputs and
            outputs (names, types and shapes).
        Check: the operator counts and the serialized graph inputs and outputs.
        """
        folded_model = build_seg3d_model(export_seg3d_label_fold=True).cuda().eval()
        num_classes = self.model.seg3d_head.classifier.out_features
        graphs = {}
        with tempfile.TemporaryDirectory() as directory:
            for name, model in (("default", self.model), ("folded", folded_model)):
                path = Path(directory) / f"{name}.onnx"
                spec = model.build_export_specs(self.batch)[SEG3D_HEAD_GRAPH]
                export_to_onnx(spec, OmegaConf.create(BASE_ONNX_CONFIG), path)
                graphs[name] = onnx.load(path).graph
        ops = {
            name: Counter(node.op_type for node in graph.node if node.op_type != "Constant")
            for name, graph in graphs.items()
        }
        self.assertEqual(ops["default"]["ArgMax"], 1)
        self.assertEqual(
            ops["folded"],
            ops["default"] - Counter({"ArgMax": 1}) + label_fold_op_counts(num_classes),
        )
        for field in ("input", "output"):
            self.assertEqual(
                [tensor.SerializeToString() for tensor in getattr(graphs["default"], field)],
                [tensor.SerializeToString() for tensor in getattr(graphs["folded"], field)],
                field,
            )


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

    def test_seg3d_deployment_with_label_fold_reproduces_forward(self) -> None:
        """Test that the segmentation deployment with the label fold reproduces forward().

        Input: a segmentation model with ``export_seg3d_label_fold`` whose window at every
            level is that level's voxel count, and one batch.
        Expected: the deployed labels equal forward()'s argmax and the probabilities match to
            1e-5.
        Check: assert_deployment_reproduces_forward().
        """
        model = build_seg3d_model(patch_sizes=self.patch_sizes, export_seg3d_label_fold=True)
        assert_deployment_reproduces_forward(
            model.cuda().eval(), self.batch, self.device, self._seg3d_tensors, rtol=1e-5, atol=1e-5
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
