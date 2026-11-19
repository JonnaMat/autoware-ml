"""Metric lifecycle for models.

``MetricEvalMixin`` is mixed into ``BaseModel`` and drives the validation and
test metric lifecycle for a list of :class:`~autoware_ml.metrics.base.MetricSuite`
objects. A model only implements ``build_eval_output``. The mixin resets each
suite at epoch start, calls ``update`` per batch, and ``result`` at epoch end,
logging under ``{split}/{prefix}/{key}``.

Each suite is cloned per stage and registered as a submodule, so Lightning moves
its state to the right device. torchmetrics owns the cross-GPU sync, which runs
inside ``result`` at epoch end.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.metrics.base import EvalStage, MetricSuite
from autoware_ml.metrics.report import MetricReport


class MetricEvalMixin:
    """Owns the metric suites and the validation/test epoch lifecycle."""

    def __init__(
        self, *args: Any, metrics: Sequence[MetricSuite] | None = None, **kwargs: Any
    ) -> None:
        """Clone the metric suites per stage and register them as submodules.

        Args:
            metrics: Suites attached from config. Empty means only losses are
                logged.
            *args: Positional arguments forwarded to the next base.
            **kwargs: Keyword arguments forwarded to the next base.
        """
        super().__init__(*args, **kwargs)
        prototypes = list(metrics) if metrics else []
        for metric in prototypes:
            if not metric.prefix:
                raise ValueError(
                    f"{type(metric).__name__} has an empty prefix, every suite attached "
                    "to a model needs one so its log keys are namespaced."
                )
        prefixes = [metric.prefix for metric in prototypes]
        duplicates = sorted({prefix for prefix in prefixes if prefixes.count(prefix) > 1})
        if duplicates:
            raise ValueError(
                f"Two suites share the prefix(es) {duplicates}, their keys would merge "
                "into one namespace. Set a distinct prefix per suite."
            )
        # A suite is cloned only for stages it actually reports at, so a heavy
        # test-only suite (and its deep-copied providers) never exists at val.
        self._metrics_by_stage = nn.ModuleDict(
            {
                stage.value: nn.ModuleList(
                    [
                        self._stage_clone(metric, stage)
                        for metric in prototypes
                        if metric.runs_at(stage)
                    ]
                )
                for stage in (EvalStage.VAL, EvalStage.TEST)
            }
        )

    @staticmethod
    def _stage_clone(metric: MetricSuite, stage: EvalStage) -> MetricSuite:
        clone = metric.clone()
        clone.bind_stage(stage)
        return clone

    def build_eval_output(self, batch: ModelBatchInputs, outputs: Any) -> dict[str, Any]:
        """Map raw forward outputs and the batch to the flat dict metrics read.

        Override in a model that attaches metrics. The default produces nothing,
        which is correct for a model with no metrics.

        Args:
            batch: Model inputs as fed to the model.
            outputs: Raw forward outputs.

        Returns:
            Flat dict the attached metric suites read.
        """
        return {}

    def _stage_metrics(self, stage: EvalStage) -> nn.ModuleList:
        return self._metrics_by_stage[stage.value]

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metric state for a fresh epoch."""
        for metric in self._stage_metrics(EvalStage.VAL):
            metric.reset()

    def on_test_epoch_start(self) -> None:
        """Reset the test metric state for a fresh epoch."""
        for metric in self._stage_metrics(EvalStage.TEST):
            metric.reset()

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one validation batch into every metric.

        Args:
            outputs: Raw forward outputs of the batch.
            batch: Collated batch as fed to the model.
            batch_idx: Batch index within the epoch.
            dataloader_idx: Dataloader index, part of the Lightning signature.
        """
        self._update_metrics(EvalStage.VAL, outputs, batch, batch_idx)

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one test batch into every metric.

        Args:
            outputs: Raw forward outputs of the batch.
            batch: Collated batch as fed to the model.
            batch_idx: Batch index within the epoch.
            dataloader_idx: Dataloader index, part of the Lightning signature.
        """
        self._update_metrics(EvalStage.TEST, outputs, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        """Combine, compute, and log the validation metrics."""
        self._log_metrics(EvalStage.VAL)

    def on_test_epoch_end(self) -> None:
        """Combine, compute, and log the test metrics."""
        self._log_metrics(EvalStage.TEST)

    def _update_metrics(self, stage: EvalStage, outputs: Any, batch: Any, batch_idx: int) -> None:
        metrics = self._stage_metrics(stage)
        if not len(metrics):
            return
        raw_outputs = (
            outputs["model_outputs"]
            if isinstance(outputs, Mapping) and "model_outputs" in outputs
            else outputs
        )
        eval_out = self.build_eval_output(batch, raw_outputs)
        if batch_idx == 0:
            self._check_required_keys(list(metrics), eval_out)
        dump_dir = os.environ.get("AML_EVAL_DUMP_DIR")
        if dump_dir and stage is EvalStage.TEST:
            self._dump_eval_inputs(Path(dump_dir), list(metrics), eval_out, batch_idx)
        for metric in metrics:
            metric.update(eval_out)

    def _dump_eval_inputs(
        self, dump_dir: Path, metrics: list, eval_out: Mapping[str, Any], batch_idx: int
    ) -> None:
        """Opt-in (``AML_EVAL_DUMP_DIR``): save what the ``det3d`` and ``seg3d`` suites read.

        One file per test batch, for offline per-scene slicing with the same suites: the raw
        detection inputs per frame (predictions, unfiltered GT boxes/labels/point counts,
        padding flag) and, per frame, the ``seg3d`` confusion matrix of every range bucket
        (``[R+1, C, C]``, bucket 0 whole scene) built as the suite's ``update`` does. The
        scene token (``<db>/<uuid>/<version>``) is stored per frame. Metrics are unaffected.
        """
        record: dict[str, Any] = {"batch_idx": batch_idx}
        suites = {getattr(metric, "prefix", None): metric for metric in metrics}

        def cpu(value: Any) -> Any:
            return value.detach().cpu() if isinstance(value, torch.Tensor) else value

        if "det3d" in suites and "predictions" in eval_out:
            num_points = eval_out.get("gt_num_points")
            record["det3d"] = {
                "predictions": [
                    {key: cpu(value) for key, value in prediction.items()}
                    for prediction in eval_out["predictions"]
                ],
                "gt_boxes": [cpu(boxes) for boxes in eval_out["gt_boxes"]],
                "gt_labels": [cpu(labels) for labels in eval_out["gt_labels"]],
                "gt_num_points": [cpu(n) for n in num_points] if num_points is not None else None,
                "padding_mask": list(eval_out.get("padding_mask") or []),
                "scene_token": list(eval_out.get("scene_token") or []),
            }
        seg = suites.get("seg3d")
        if seg is not None and "seg3d_frames" in eval_out:
            frames = []
            for frame in eval_out["seg3d_frames"]:
                predictions = frame["pred"].detach().reshape(-1)
                targets = frame["target"].detach().reshape(-1)
                coord = frame["coord"].detach().to(dtype=torch.float32)
                valid = (
                    (targets != seg.ignore_index)
                    & (targets >= 0)
                    & (targets < seg.num_classes)
                    & (predictions >= 0)
                    & (predictions < seg.num_classes)
                )
                targets = targets[valid].long()
                predictions = predictions[valid].long()
                coord = coord[valid]
                confusion = torch.zeros(
                    len(seg.ranges) + 1, seg.num_classes, seg.num_classes, dtype=torch.long
                )
                confusion[0] = seg._counts(predictions, targets).cpu()
                distance = torch.linalg.vector_norm(coord[:, :2], dim=1)
                for index, metric_range in enumerate(seg.ranges):
                    in_range = distance >= metric_range.min_distance
                    if metric_range.max_distance is not None:
                        in_range &= distance < metric_range.max_distance
                    confusion[index + 1] = seg._counts(
                        predictions[in_range], targets[in_range]
                    ).cpu()
                frames.append({"confusion": confusion, "scene_token": frame.get("scene_token")})
            record["seg3d"] = frames
        dump_dir.mkdir(parents=True, exist_ok=True)
        torch.save(record, dump_dir / f"batch_{batch_idx:05d}.pt")

    def _check_required_keys(self, metrics: list, eval_out: Mapping[str, Any]) -> None:
        for metric in metrics:
            missing = [key for key in metric.required_keys() if key not in eval_out]
            if missing:
                raise ValueError(
                    f"Metric {type(metric).__name__!r} needs {missing}, not produced by "
                    f"{type(self).__name__}.build_eval_output."
                )

    def _log_metrics(self, stage: EvalStage) -> None:
        report = MetricReport()
        for metric in self._stage_metrics(stage):
            report.merge(metric.result(stage), prefix=metric.prefix)
        if not len(report):
            return
        # Values are already global and identical on every rank after sync, so no sync_dist.
        self.log_dict(report.to_log_dict(stage.value), on_step=False, on_epoch=True, logger=True)
