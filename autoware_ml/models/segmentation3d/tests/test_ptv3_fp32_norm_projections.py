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

"""The projections that feed a BatchNorm stay in fp32 under autocast and fold it on export."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import onnx
import torch
from torch import nn

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    FullPrecisionLinear,
    SerializedUnpooling,
)


class TestFullPrecisionLinear(unittest.TestCase):
    """Unit tests for the linear layer that computes in fp32 under autocast."""

    @unittest.skipUnless(torch.cuda.is_available(), "autocast on CUDA")
    def test_projection_survives_inputs_that_overflow_fp16(self) -> None:
        """
        Input: a 64 -> 32 projection with weights scaled by 40 and 1000 features scaled by
        3000, run under fp16 autocast both as FullPrecisionLinear and as an nn.Linear with
        the same weights.
        Expected: the products exceed the fp16 range (65,504), so the plain layer overflows,
        while FullPrecisionLinear computes in fp32 and stays finite.
        Check: the plain output has non-finite values; the FullPrecisionLinear output is fp32,
        finite and matches the fp32 projection of the same input.
        """
        torch.manual_seed(0)
        layer = FullPrecisionLinear(64, 32).cuda()
        with torch.no_grad():
            layer.weight.mul_(40.0)
        linear = nn.Linear(64, 32).cuda()
        linear.load_state_dict(layer.state_dict())
        feat = torch.randn(1000, 64, device="cuda") * 3000.0  # products exceed 65,504
        with torch.autocast("cuda", dtype=torch.float16):
            naive = linear(feat)
            guarded = layer(feat)
        self.assertFalse(
            torch.isfinite(naive).all(), "the test input must overflow the plain fp16 path"
        )
        self.assertEqual(guarded.dtype, torch.float32)
        self.assertTrue(torch.isfinite(guarded).all())
        torch.testing.assert_close(guarded, linear(feat.float()), rtol=1e-4, atol=1e-2)

    def test_layer_is_a_plain_linear_outside_autocast(self) -> None:
        """
        Input: an 8 -> 4 FullPrecisionLinear and an nn.Linear loaded with its state dict,
        applied to five random features without autocast.
        Expected: outside autocast FullPrecisionLinear is a plain linear layer with the same
        parameters, so checkpoints load unchanged.
        Check: both outputs match, and the state dict holds only ``weight`` and ``bias``.
        """
        torch.manual_seed(1)
        layer = FullPrecisionLinear(8, 4)
        plain = nn.Linear(8, 4)
        plain.load_state_dict(layer.state_dict())
        x = torch.randn(5, 8)
        torch.testing.assert_close(layer(x), plain(x))
        self.assertEqual(set(layer.state_dict()), {"weight", "bias"})


class TestSerializedUnpooling(unittest.TestCase):
    """Unit tests for the unpooling projections."""

    def test_unpooling_state_dict_layout_is_unchanged(self) -> None:
        """
        Input: a SerializedUnpooling with 16 input, 8 skip and 12 output channels.
        Expected: the projections are FullPrecisionLinear inside the same PointSequential, so
        the parameter names of existing checkpoints do not change.
        Check: the state dict has ``proj.0.weight``, ``proj_skip.0.weight`` and the
        BatchNorm running means at ``proj.1`` and ``proj_skip.1``.
        """
        module = SerializedUnpooling(16, 8, 12)
        keys = set(module.state_dict())
        self.assertIn("proj.0.weight", keys)
        self.assertIn("proj_skip.0.weight", keys)
        self.assertIn("proj.1.running_mean", keys)
        self.assertIn("proj_skip.1.running_mean", keys)


def _unpooling_with_trained_batch_norms() -> SerializedUnpooling:
    """Return an eval-mode unpooling whose batch norms hold non-trivial statistics."""
    torch.manual_seed(2)
    module = SerializedUnpooling(16, 8, 12)
    for norm in (module.proj.get_submodule("1"), module.proj_skip.get_submodule("1")):
        nn.init.uniform_(norm.weight, 0.5, 1.5)
        nn.init.uniform_(norm.bias, -0.5, 0.5)
        nn.init.uniform_(norm.running_mean, -0.5, 0.5)
        nn.init.uniform_(norm.running_var, 0.1, 2.0)
    return module.eval()


class TestSerializedUnpoolingBatchNormFold(unittest.TestCase):
    """The export folds the unpooling batch norms into the linear layers before them."""

    def test_folded_projections_match_the_batch_norm_projections(self) -> None:
        """
        Input: an eval-mode SerializedUnpooling (16 input, 8 skip, 12 output channels) with
        non-trivial batch-norm statistics and affines, and a copy of it with the batch norms
        folded, both applied to 64 random features per projection in fp32.
        Expected: the fold is an exact rewrite of the eval-mode batch norm, so each projection
        computes the same output up to fp32 rounding, and only the copy's parameters change.
        Check: the outputs of ``proj`` and ``proj_skip`` match within 1e-5, the folded
        projections hold no BatchNorm1d, and the original's state dict is unchanged.
        """
        module = _unpooling_with_trained_batch_norms()
        state = {key: value.clone() for key, value in module.state_dict().items()}
        folded = deepcopy(module)
        folded.fold_batch_norm()

        for name, channels in (("proj", 16), ("proj_skip", 8)):
            feat = torch.randn(64, channels) * 3.0
            with torch.no_grad():
                reference = getattr(module, name)(feat)
                result = getattr(folded, name)(feat)
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5, msg=name)
            self.assertFalse(
                any(isinstance(child, nn.BatchNorm1d) for child in getattr(folded, name).modules()),
                name,
            )
        for key, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_folded_projection_exports_without_batch_normalization(self) -> None:
        """
        Input: the ``proj`` projection of an unpooling, exported to ONNX (TorchScript exporter,
        opset 17, no constant folding, as the deploy configs do) before and after the fold.
        Expected: the fold removes the batch norm from the graph, so the projection exports as
        a linear layer followed by the GELU.
        Check: the unfolded graph has one BatchNormalization node and the folded one has none,
        and both keep the GELU's Erf.
        """
        module = _unpooling_with_trained_batch_norms()
        counts = {}
        with tempfile.TemporaryDirectory() as directory:
            for fold in (False, True):
                if fold:
                    module.fold_batch_norm()
                path = Path(directory) / f"proj_{fold}.onnx"
                torch.onnx.export(
                    module.proj,
                    (torch.randn(5, 16),),
                    path,
                    input_names=["feat"],
                    output_names=["out"],
                    opset_version=17,
                    dynamo=False,
                    do_constant_folding=False,
                )
                ops = [node.op_type for node in onnx.load(path).graph.node]
                counts[fold] = (ops.count("BatchNormalization"), ops.count("Erf"))
        self.assertEqual(counts[False], (1, 1))
        self.assertEqual(counts[True], (0, 1))


if __name__ == "__main__":
    unittest.main()
