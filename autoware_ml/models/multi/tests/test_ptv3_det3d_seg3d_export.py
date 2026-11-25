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

"""Export-contract tests for combined PTv3 segmentation+detection export."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
from omegaconf import OmegaConf

from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.deployment.graphs import GraphTensor, run_deployment_in_torch
from autoware_ml.deployment.tests.deployment_assertions import (
    assert_deployment_reproduces_forward,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.detection3d.tests.head_output_fixtures import build_transfusion_outputs
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_detection3d_gt_batch,
    build_point_cloud_batch,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_bev_neck,
    build_inputs,
    build_ptv3_encoder,
    build_seg3d_head,
    build_seg3d_model,
    build_trans_model,
    build_transfusion_head,
    whole_level_patch_sizes,
)
from autoware_ml.models.multi.ptv3_det3d_seg3d import PTv3Det3DSeg3DModel
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET3D_HEAD_GRAPH,
    ENCODER_GRAPH,
    PTv3BaseModel,
    SEG3D_HEAD_GRAPH,
    seg3d_head_export_input_names,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.checkpoints import apply_matching_weights
from autoware_ml.utils.deploy import export_to_onnx
from autoware_ml.utils.onnx_precision import OnnxPrecision, convert_onnx_precision

JOINT_EXPORT_OUTPUT_NAMES = [
    "pred_labels",
    "pred_probs",
    "dense_heatmap",
    "query_heatmap_score",
    "query_labels",
    "heatmap",
    "center",
    "height",
    "dim",
    "rot",
    "vel",
]


class _DummyBBoxHead:
    def decode_outputs(self, outputs: object) -> ModelPredictions:
        del outputs
        return ModelPredictions(
            detection3d_predictions=[
                Detection3DSamplePredictions(
                    bboxes_3d=torch.zeros((0, 9), dtype=torch.float32),
                    scores_3d=torch.zeros((0,), dtype=torch.float32),
                    labels_3d=torch.zeros((0,), dtype=torch.long),
                )
            ]
        )


def _save_segmentation3d_checkpoint(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "ptv3_segmentation3d.ckpt"
    torch.save({"state_dict": build_seg3d_model().state_dict()}, checkpoint_path)
    return checkpoint_path


def test_seg3d_head_export_input_names_follow_dec_depths_rule() -> None:
    """The split seg-head contract: base set for depths-0, per-block-stage metadata otherwise."""
    base_names = [
        "point_feat_0",
        "point_feat_1",
        "point_feat_2",
        "point_feat_3",
        "point_feat_4",
        "pooling_cluster_0",
        "pooling_cluster_1",
        "pooling_cluster_2",
        "pooling_cluster_3",
    ]
    assert seg3d_head_export_input_names(5, [0, 0, 0, 0]) == base_names
    assert seg3d_head_export_input_names(5, [0, 0, 1, 1]) == base_names + [
        "serialized_pooling_1_patch_order",
        "serialized_pooling_1_serialized_inverse",
        "serialized_pooling_1_grid_coord",
        "serialized_pooling_2_patch_order",
        "serialized_pooling_2_serialized_inverse",
        "serialized_pooling_2_grid_coord",
    ]
    assert seg3d_head_export_input_names(2, [1]) == [
        "point_feat_0",
        "point_feat_1",
        "pooling_cluster_0",
        "patch_order",
        "serialized_inverse",
        "grid_coord",
    ]
    with pytest.raises(ValueError, match="entries"):
        seg3d_head_export_input_names(5, [0, 0, 0])


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_seg3d_split_export_supports_decoder_blocks() -> None:
    """A blocks decoder (dec_depths (1,) at stage 0) exports and runs as a split seg head."""
    model = build_seg3d_model().cuda().eval()
    batch = build_inputs(device=torch.device("cuda"))

    specs = model.build_export_specs(batch)

    # The encoder-only encoder graph never consumes pooling clusters; the
    # tracer would prune them, so the declared interface must not list them.
    assert not any("_cluster" in name for name in specs["ptv3_encoder"].input_names)
    with torch.no_grad():
        encoder_outputs = specs["ptv3_encoder"].module(*specs["ptv3_encoder"].args)
    assert len(encoder_outputs) == 2

    spec = specs["ptv3_seg3d_head"]
    assert spec.input_names == [
        "point_feat_0",
        "point_feat_1",
        "pooling_cluster_0",
        "patch_order",
        "serialized_inverse",
        "grid_coord",
    ]
    assert spec.dynamic_axes["serialized_inverse"] == {1: "num_voxels"}
    assert spec.dynamic_axes["patch_order"] == {1: "padded_voxels"}
    assert spec.dynamic_axes["grid_coord"] == {0: "num_voxels"}

    with torch.no_grad():
        pred_labels, pred_probs = spec.module(*spec.args)

    num_voxels = batch.voxels_data.voxels.shape[0]
    assert pred_labels.shape == (num_voxels,)
    assert pred_probs.shape == (num_voxels, 3)
    assert pred_labels.dtype == torch.long


def test_ptv3_det3d_seg3d_eval_output_scatters_segmentation3d_to_original_points() -> None:
    model = SimpleNamespace(
        bbox_head=_DummyBBoxHead(),
        _detection3d_gt_batch=PTv3Det3DSeg3DModel._detection3d_gt_batch,
        _detection3d_frame_mask=PTv3Det3DSeg3DModel._detection3d_frame_mask,
        _mask_list=PTv3Det3DSeg3DModel._mask_list,
    )
    det3d_outputs = build_transfusion_outputs(
        {
            name: torch.zeros((1, channels, 8))
            for name, channels in (
                ("heatmap", 2),
                ("center", 2),
                ("height", 1),
                ("dim", 3),
                ("rot", 2),
            )
        }
    )
    outputs = ModelOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(
            center_head_outputs=None, transfusion_head_outputs=det3d_outputs
        ),
        segmentation3d_head_outputs=Segmentation3DHeadOutputs(
            logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        ),
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [1.0, 1.0, 0.0, 0.5, 0.0],
            [2.0, 2.0, 0.0, 0.5, 0.0],
            [3.0, 3.0, 0.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    segment = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    point_cloud = build_point_cloud_batch([points], timestamp_difference_dim=4)
    batch = build_batch_inputs(
        point_cloud=point_cloud,
        detection=build_detection3d_gt_batch(
            [torch.zeros((0, 9), dtype=torch.float32)], [torch.zeros((0,), dtype=torch.long)]
        ),
        segmentation=Segmentation3DGTBatch(
            gt_semantic_masks=segment, batch_indices=point_cloud.batch_indices
        ),
        voxels=VoxelsData(
            voxels=torch.zeros((2, 2, 5), dtype=torch.float32),
            coords=torch.zeros((2, 3), dtype=torch.int32),
            num_points=torch.full((2,), 2, dtype=torch.int32),
            batch_indices=torch.zeros(2, dtype=torch.int32),
            point_voxel_indices=torch.tensor([0, 1, 1, 0], dtype=torch.long),
            num_dropped_voxels=torch.tensor(0),
        ),
    )

    eval_out = PTv3Det3DSeg3DModel.build_eval_output(model, batch, outputs)

    assert eval_out.segmentation3d is not None and eval_out.detection3d is not None
    (frame,) = eval_out.segmentation3d.frames
    assert frame.pred.tolist() == [0, 1, 1, 0]
    assert torch.equal(frame.target, segment)
    assert torch.equal(frame.coord, points[:, :3])
    assert frame.scores is not None and frame.scores.shape == (4, 2)
    assert frame.gt_boxes is not None and frame.gt_boxes.shape == (0, 9)
    assert len(eval_out.detection3d.frames) == 1


def _save_detection3d_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    *,
    drift_encoder: bool = False,
) -> Path:
    state_dict = model.state_dict()
    if drift_encoder:
        state_dict = dict(state_dict)
        key = next(
            key
            for key, value in state_dict.items()
            if key.startswith("encoder.") and torch.is_tensor(value) and value.is_floating_point()
        )
        state_dict[key] = state_dict[key].clone()
        state_dict[key].flatten()[0].add_(1.0)
    torch.save(
        {
            "state_dict": state_dict,
            "autoware_ml_checkpoint_recipe": {
                "type": "ptv3_detection3d",
                "freeze_encoder": True,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _build_det3d_seg3d_model(
    patch_sizes: Sequence[int] = (4, 4), export_fp16_point_features: bool | None = None
) -> PTv3Det3DSeg3DModel:
    """Return a small joint PTv3 model over the test encoder.

    Args:
        patch_sizes: Attention window of every level, the input level first.
        export_fp16_point_features: Export the encoder-to-head features as fp16; ``None``
            leaves the model's default.

    Returns:
        The joint model on the CPU.
    """
    return PTv3Det3DSeg3DModel(
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        encoder=build_ptv3_encoder(patch_sizes=patch_sizes),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg3d_head(patch_sizes=patch_sizes),
        bev_neck=build_bev_neck(),
        bbox_head=build_transfusion_head(),
        export_output_names=JOINT_EXPORT_OUTPUT_NAMES,
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        **(
            {}
            if export_fp16_point_features is None
            else {"export_fp16_point_features": export_fp16_point_features}
        ),
    )


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_det3d_seg3d_deployment_is_the_seg3d_and_det3d_graphs_over_one_encoder(
    tmp_path: Path,
) -> None:
    """Test that the joint model's deployment is the seg3d and det3d graphs over one encoder.

    Input: the joint model loaded with the single-task models' weights, and one batch.
    Expected: three graphs named encoder, seg3d head, det3d head, the det head reading nothing
        the seg head emits; each head's outputs match the single-task model's own deployment.
    Check: the graph names and output names, the deployed outputs against the seg-only and
        det-only deployments, and the derived spec names.
    """
    segmentation3d_checkpoint = _save_segmentation3d_checkpoint(tmp_path)
    detection3d_model = build_trans_model(freeze_encoder=True)
    apply_matching_weights(detection3d_model, (segmentation3d_checkpoint,))
    detection3d_checkpoint = tmp_path / "ptv3_trans_detection3d.ckpt"
    _save_detection3d_checkpoint(detection3d_checkpoint, detection3d_model)
    model = _build_det3d_seg3d_model().cuda()
    apply_matching_weights(
        model,
        (detection3d_checkpoint, segmentation3d_checkpoint),
        map_location=torch.device("cuda"),
        device=torch.device("cuda"),
        set_eval=True,
    )
    seg_model = build_seg3d_model().cuda().eval()
    apply_matching_weights(
        seg_model, (segmentation3d_checkpoint,), map_location=torch.device("cuda")
    )
    detection3d_model = detection3d_model.cuda().eval()

    graphs = {graph.name: graph for graph in model.deployment_declaration().graphs}
    assert set(graphs) == {ENCODER_GRAPH, SEG3D_HEAD_GRAPH, DET3D_HEAD_GRAPH}
    seg_graph, det_graph = graphs[SEG3D_HEAD_GRAPH], graphs[DET3D_HEAD_GRAPH]
    assert seg_graph.output_names == ("pred_labels", "pred_probs")
    assert det_graph.output_names == tuple(JOINT_EXPORT_OUTPUT_NAMES[2:])
    assert not set(det_graph.input_names) & set(seg_graph.output_names)

    batch = build_inputs(device=torch.device("cuda"))
    cuda = torch.device("cuda")
    joint = run_deployment_in_torch(model.deployment_declaration(), batch, cuda)
    assert list(joint) == JOINT_EXPORT_OUTPUT_NAMES
    seg_only = run_deployment_in_torch(seg_model.deployment_declaration(), batch, cuda)
    det_only = run_deployment_in_torch(detection3d_model.deployment_declaration(), batch, cuda)
    assert torch.equal(joint["pred_labels"], seg_only["pred_labels"])
    torch.testing.assert_close(joint["pred_probs"], seg_only["pred_probs"])
    for name in JOINT_EXPORT_OUTPUT_NAMES[2:]:
        torch.testing.assert_close(joint[name], det_only[name], msg=name)

    specs = model.build_export_specs(batch)
    assert set(specs) == set(graphs)
    assert specs[DET3D_HEAD_GRAPH].output_names == JOINT_EXPORT_OUTPUT_NAMES[2:]


def _joint_tensors(outputs: ModelOutputs) -> dict[str, GraphTensor]:
    """Return forward()'s segmentation and detection outputs under the joint graphs' names.

    Args:
        outputs: The outputs of ``forward()``.

    Returns:
        The seg3d head graph's outputs, then the det3d head graph's, by name.
    """
    probs = torch.softmax(outputs.segmentation3d().logits, dim=1)
    det3d_names = JOINT_EXPORT_OUTPUT_NAMES[2:]
    head_outputs = outputs.detection3d().transfusion_head()
    tensors = {"pred_labels": probs.argmax(dim=1), "pred_probs": probs}
    tensors.update(zip(det3d_names, head_outputs.export_tensors(det3d_names)))
    return tensors


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_det3d_seg3d_deployment_reproduces_forward() -> None:
    """Test that the joint deployment reproduces forward() for both heads.

    Input: a joint model whose window at every level is that level's voxel count, and one
        batch.
    Expected: every segmentation and detection output of the deployment matches forward()'s
        to 1e-5.
    Check: assert_deployment_reproduces_forward().
    """
    cuda = torch.device("cuda")
    batch = build_inputs(device=cuda)
    model = _build_det3d_seg3d_model(whole_level_patch_sizes(batch)).cuda().eval()
    assert_deployment_reproduces_forward(model, batch, cuda, _joint_tensors, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_det3d_seg3d_detection3d_outputs_invariant_to_seg3d_head() -> None:
    """The core stage-4 guarantee: with a fixed encoder, changing the seg head
    must not change detection outputs (the det branch taps the encoder only)."""
    torch.manual_seed(0)
    model = _build_det3d_seg3d_model().cuda().eval()
    batch = build_inputs(device=torch.device("cuda"))

    with torch.no_grad():
        reference = model(batch)
        for parameter in model.seg3d_head.parameters():
            parameter.add_(1.0)
        perturbed = model(batch)

    names = ["dense_heatmap", "query_heatmap_score", "query_labels", "heatmap", "center"]
    names += ["height", "dim", "rot", "vel"]
    reference_tensors = reference.detection3d().transfusion_head().export_tensors(names)
    perturbed_tensors = perturbed.detection3d().transfusion_head().export_tensors(names)
    for name, value, perturbed_value in zip(names, reference_tensors, perturbed_tensors):
        assert torch.equal(value, perturbed_value), name
    assert not torch.equal(reference.segmentation3d().logits, perturbed.segmentation3d().logits)


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_det3d_seg3d_seg3d_logits_invariant_to_det3d_branch_pass() -> None:
    """The BEV neck must read the encoder chain non-destructively: seg logits
    are identical whether or not the detection branch ran first."""
    torch.manual_seed(0)
    model = _build_det3d_seg3d_model().cuda().eval()
    batch = build_inputs(device=torch.device("cuda"))

    with torch.no_grad():
        joint_logits = model(batch).segmentation3d().logits
        point = model.encode_batch(batch)
        seg3d_only_logits = model.seg3d_head(point)

    torch.testing.assert_close(joint_logits, seg3d_only_logits)


def _is_point_feature(name: str) -> bool:
    """Return whether a graph tensor is one of the per-stage encoder features."""
    return name.startswith("point_feat_")


def _cast_target(node: onnx.NodeProto) -> int:
    """Return the ``to`` dtype of a ``Cast`` node."""
    return next(attribute.i for attribute in node.attribute if attribute.name == "to")


@pytest.mark.parametrize(
    "build_model", [_build_det3d_seg3d_model, build_seg3d_model, build_trans_model]
)
def test_ptv3_export_fp16_point_features_is_off_by_default(
    build_model: Callable[[], PTv3BaseModel],
) -> None:
    """Test that every PTv3 model exports fp32 features unless it opts in.

    Input: the joint, the segmentation and the detection model with their default
        ``export_fp16_point_features``, then with it set.
    Expected: off by default (the fp16 interface changes the engines' I/O dtypes); once set,
        the declared encoder graph returns fp16 features.
    Check: the model attribute and the encoder graph module's ``fp16_point_features``.
    """
    model = build_model()
    assert model.export_fp16_point_features is False
    graphs = {graph.name: graph for graph in model.deployment_graphs()}
    assert graphs[ENCODER_GRAPH].module.fp16_point_features is False
    model.export_fp16_point_features = True
    graphs = {graph.name: graph for graph in model.deployment_graphs()}
    assert graphs[ENCODER_GRAPH].module.fp16_point_features is True


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_det3d_seg3d_fp16_point_features_match_fp16_rounded_fp32_features() -> None:
    """Test that the fp16 interface is the fp32 deployment with fp16-rounded features.

    Input: one joint model exported with and without ``export_fp16_point_features``, the same
        weights, one batch.
    Expected: the encoder returns exactly fp16(the fp32 features); each head fed the fp16
        features returns exactly what it returns on the fp16-rounded fp32 features; every other
        tensor is unchanged.
    Check: the trace args and outputs of the derived specs, bitwise.
    """
    cuda = torch.device("cuda")
    batch = build_inputs(device=cuda)
    torch.manual_seed(0)
    reference = _build_det3d_seg3d_model().cuda().eval()
    model = _build_det3d_seg3d_model(export_fp16_point_features=True).cuda().eval()
    model.load_state_dict(reference.state_dict())

    reference_specs = reference.build_export_specs(batch)
    specs = model.build_export_specs(batch)
    with torch.no_grad():
        reference_feats = reference_specs[ENCODER_GRAPH].module(
            *reference_specs[ENCODER_GRAPH].args
        )
        feats = specs[ENCODER_GRAPH].module(*specs[ENCODER_GRAPH].args)
        assert all(feat.dtype == torch.float16 for feat in feats)
        for feat, reference_feat in zip(feats, reference_feats, strict=True):
            assert reference_feat.dtype == torch.float32
            assert torch.equal(feat, reference_feat.half())

        for graph in (SEG3D_HEAD_GRAPH, DET3D_HEAD_GRAPH):
            spec, reference_spec = specs[graph], reference_specs[graph]
            assert spec.input_names == reference_spec.input_names
            rounded_args = []
            for name, arg, reference_arg in zip(
                spec.input_names, spec.args, reference_spec.args, strict=True
            ):
                if _is_point_feature(name):
                    assert arg.dtype == torch.float16, (graph, name)
                    assert torch.equal(arg, reference_arg.half()), (graph, name)
                    rounded_args.append(reference_arg.half().float())
                else:
                    assert torch.equal(arg, reference_arg), (graph, name)
                    rounded_args.append(reference_arg)
            outputs = spec.module(*spec.args)
            reference_outputs = reference_spec.module(*rounded_args)
            for name, output, reference_output in zip(
                spec.output_names, outputs, reference_outputs, strict=True
            ):
                assert torch.equal(output, reference_output), (graph, name)


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_det3d_seg3d_fp16_point_features_onnx_interface(tmp_path: Path) -> None:
    """Test that the exported graphs declare the point features fp16 and nothing else changes.

    Input: one joint model exported to ONNX with and without ``export_fp16_point_features``;
        the det3d head also converted to fp16 as the shipped configs do.
    Expected: the encoder's ``point_feat_*`` outputs and the heads' ``point_feat_*`` inputs are
        FLOAT16, every other input and output keeps its name, dtype and shape; the fp16 det3d
        head reads its fp16 inputs directly (no cast left on them), the fp32 seg3d head casts
        them to fp32 once; without the option no head casts its fp32 features.
    Check: the ONNX value infos and the consumers of the ``point_feat_*`` inputs.
    """
    cuda = torch.device("cuda")
    batch = build_inputs(device=cuda)
    onnx_cfg = OmegaConf.create(
        {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
    )

    def export(model: PTv3Det3DSeg3DModel, directory: Path) -> dict[str, onnx.ModelProto]:
        """Export every graph; the det3d head also before its fp16 conversion (``*_fp32``)."""
        directory.mkdir()
        models = {}
        for name, spec in model.build_export_specs(batch).items():
            path = directory / f"{name}.onnx"
            export_to_onnx(spec, onnx_cfg, path)
            if name == DET3D_HEAD_GRAPH:
                models[f"{name}_fp32"] = onnx.load(path.as_posix())
                path = convert_onnx_precision(path, OnnxPrecision.FP16)
            models[name] = onnx.load(path.as_posix())
        return models

    def io(model: onnx.ModelProto) -> list[tuple[str, int, list[int | str]]]:
        # The exporter names an axis it cannot infer after the op producing the tensor
        # (``<Op><tensor>_dim_<axis>``; the cast changes ``<Op>``), so only the declared
        # dynamic axes and the static sizes are compared by value.
        def dim(value: onnx.ValueInfoProto, axis: int, size: onnx.TensorShapeProto.Dimension):
            if size.dim_param.endswith(f"{value.name}_dim_{axis}"):
                return "?"
            return size.dim_param or size.dim_value

        return [
            (
                value.name,
                value.type.tensor_type.elem_type,
                [dim(value, i, d) for i, d in enumerate(value.type.tensor_type.shape.dim)],
            )
            for value in [*model.graph.input, *model.graph.output]
        ]

    torch.manual_seed(0)
    reference_model = _build_det3d_seg3d_model().cuda().eval()
    fp16_model = _build_det3d_seg3d_model(export_fp16_point_features=True).cuda().eval()
    fp16_model.load_state_dict(reference_model.state_dict())
    reference = export(reference_model, tmp_path / "fp32")
    exported = export(fp16_model, tmp_path / "fp16")

    for graph in (ENCODER_GRAPH, SEG3D_HEAD_GRAPH, DET3D_HEAD_GRAPH):
        expected = [
            (name, onnx.TensorProto.FLOAT16 if _is_point_feature(name) else dtype, shape)
            for name, dtype, shape in io(reference[graph])
        ]
        assert io(exported[graph]) == expected, graph
        point_feats = [name for name, _, _ in expected if _is_point_feature(name)]
        assert point_feats, graph
        for name, dtype, _ in io(reference[graph]):
            if _is_point_feature(name):
                assert dtype == onnx.TensorProto.FLOAT, (graph, name)

    def consumers(model: onnx.ModelProto, name: str) -> list[onnx.NodeProto]:
        return [node for node in model.graph.node if name in node.input]

    # The fp32 interface keeps its graphs: no cast is traced on an fp32 feature (checked on the
    # det3d head before its fp16 conversion, which adds casts to fp16 of its own).
    for graph in (SEG3D_HEAD_GRAPH, f"{DET3D_HEAD_GRAPH}_fp32"):
        for value in reference[graph].graph.input:
            if _is_point_feature(value.name):
                nodes = consumers(reference[graph], value.name)
                assert nodes, (graph, value.name)
                assert all(node.op_type != "Cast" for node in nodes), (graph, value.name)

    det3d_inputs = [i.name for i in exported[DET3D_HEAD_GRAPH].graph.input]
    for name in filter(_is_point_feature, det3d_inputs):
        nodes = consumers(exported[DET3D_HEAD_GRAPH], name)
        assert nodes and all(node.op_type != "Cast" for node in nodes), name
    seg3d_inputs = [i.name for i in exported[SEG3D_HEAD_GRAPH].graph.input]
    for name in filter(_is_point_feature, seg3d_inputs):
        (node,) = consumers(exported[SEG3D_HEAD_GRAPH], name)
        assert node.op_type == "Cast", name
        assert _cast_target(node) == onnx.TensorProto.FLOAT, name
