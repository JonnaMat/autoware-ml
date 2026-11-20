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

"""Convert YOLOX weights into a checkpoint the ``--weights`` option of the CLI loads.

Usage::

    python -m autoware_ml.scripts.tools.yolox.convert_weights yolox_s.pth yolox_s_coco.ckpt
    python -m autoware_ml.scripts.tools.yolox.convert_weights detector.onnx detector.ckpt

A ``.onnx`` source is an export of the YOLOX reference implementation with folded batch
normalizations, any other file a checkpoint of it. Weights trained for another number of
classes load with ``skip_mismatched_weights=true``, which leaves the class outputs at their
initialization.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from autoware_ml.models.detection2d.yolox_weights import (
    convert_folded_onnx,
    convert_reference_checkpoint,
)

logger = logging.getLogger(__name__)


def convert(source: Path, destination: Path, with_ema: bool = True) -> None:
    """Convert one weights file.

    Args:
        source: ONNX file or checkpoint of the YOLOX reference implementation.
        destination: Checkpoint to write, holding the detector state dict under ``state_dict``.
        with_ema: Whether to also fill the averaged copy of the network.
    """
    if source.suffix == ".onnx":
        state_dict = convert_folded_onnx(source, with_ema=with_ema)
    else:
        checkpoint = torch.load(str(source), map_location="cpu", weights_only=True)
        state_dict = convert_reference_checkpoint(checkpoint, with_ema=with_ema)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, str(destination))
    logger.info("Wrote %d tensors to %s", len(state_dict), destination)


def main() -> None:
    """Parse the command line and convert the weights."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="ONNX file or YOLOX checkpoint to convert.")
    parser.add_argument("destination", type=Path, help="Checkpoint to write.")
    parser.add_argument(
        "--without-ema",
        action="store_true",
        help="Leave the averaged copy of the network out of the checkpoint.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert(args.source, args.destination, with_ema=not args.without_ema)


if __name__ == "__main__":
    main()
