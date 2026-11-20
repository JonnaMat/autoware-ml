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

"""Score an ONNX file of the YOLOX detector contract on the test split of a config.

The script composes a YOLOX task config for its database, its test pipeline, its decoding
settings and its metric suites, and swaps the network for the ONNX file, so a deployed graph
is scored exactly as ``autoware-ml test`` scores a checkpoint of that config.

Usage::

    python -m autoware_ml.scripts.tools.yolox.evaluate_onnx \\
        --config-name tasks/detection2d/yolox/t4dataset/yolox_s_960x960_tlr \\
        +onnx_path=/path/to/detector.onnx +metrics_path=/path/to/metrics.json

Further keys, all appended with ``+``:

* ``onnx_class_indices=[1,2]``: class output of the graph of every class of the config, for a
  graph with other class outputs than the config, such as a leading background entry. The
  first outputs by default, shifted by one when the config exports a background class.
* ``onnx_providers=[CPUExecutionProvider]``: ONNX Runtime execution providers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.model_builder import build_data_preprocessor
from autoware_ml.models.detection2d.yolox_onnx import ONNXYOLOXDetector
from autoware_ml.utils.runtime import configure_torch_runtime, get_config_path, set_seed

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


def build_onnx_detector(cfg: DictConfig) -> ONNXYOLOXDetector:
    """Build the ONNX backed detector of a composed YOLOX config.

    Args:
        cfg: Composed config with the ``onnx_path`` key added.

    Returns:
        The detector reading the channel order, the decoding settings and the metric suites
        of the config.
    """
    model_cfg = cfg.model
    num_classes = int(model_cfg.head.num_classes)
    class_indices = cfg.get("onnx_class_indices")
    if class_indices is None:
        offset = 1 if model_cfg.get("export_background_class", False) else 0
        class_indices = [offset + index for index in range(num_classes)]
    providers = cfg.get("onnx_providers")
    return ONNXYOLOXDetector(
        onnx_path=cfg.onnx_path,
        num_classes=num_classes,
        class_indices=list(class_indices),
        strides=list(model_cfg.head.strides),
        providers=list(providers) if providers is not None else None,
        log_dict_configs=hydra.utils.instantiate(model_cfg.log_dict_configs),
        data_preprocessor=build_data_preprocessor(cfg),
        metrics=hydra.utils.instantiate(cfg.metrics),
        # The graph reads the order it was exported for, not necessarily that of the backbone.
        input_channel_order=model_cfg.get("export_input_channel_order")
        or model_cfg.input_channel_order,
        score_threshold=model_cfg.score_threshold,
        nms_iou_threshold=model_cfg.nms_iou_threshold,
        class_agnostic_nms=model_cfg.class_agnostic_nms,
        max_detections=model_cfg.max_detections,
    )


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Evaluate the ONNX file on the test split and write the metrics.

    Args:
        cfg: Composed Hydra configuration.
    """
    if "onnx_path" not in cfg:
        raise ValueError("Pass the graph to score with +onnx_path=<file>.")
    configure_torch_runtime()
    set_seed(cfg)

    datamodule = build_datamodule(cfg)
    model = build_onnx_detector(cfg)
    trainer = L.Trainer(
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_test_batches=cfg.trainer.get("limit_test_batches", 1.0),
    )
    metrics = trainer.test(model, datamodule=datamodule)[0]
    report = {
        "onnx_path": str(cfg.onnx_path),
        "config_name": hydra.core.hydra_config.HydraConfig.get().job.config_name,
        "class_names": list(cfg.taxonomy.detection2d.class_names),
        "class_indices": model.class_indices,
        "postprocessing": {
            "score_threshold": model.score_threshold,
            "nms_iou_threshold": model.nms_iou_threshold,
            "class_agnostic_nms": model.class_agnostic_nms,
            "max_detections": model.max_detections,
        },
        "metric_params": OmegaConf.to_container(cfg.metric_params, resolve=True),
        "metrics": {name: float(value) for name, value in sorted(metrics.items())},
    }
    for name, value in report["metrics"].items():
        logger.info("%s: %.4f", name, value)
    metrics_path = cfg.get("metrics_path")
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Wrote the metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
