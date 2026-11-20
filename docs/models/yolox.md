---
icon: lucide/scan-search
---

# YOLOX

YOLOX is an anchor-free single-stage 2D object detector with a decoupled head and the SimOTA
label assignment. Autoware-ML trains it on single camera images annotated with 2D boxes and
exports it with the output layout the Autoware `tensorrt_yolox` runtime decodes.

## Summary

| Property     | Value                                                                    |
| ------------ | ------------------------------------------------------------------------ |
| Task         | 2D object detection                                                      |
| Modality     | Camera                                                                   |
| Input        | One BGR image in `[0, 255]`, letterboxed to the network input            |
| Output       | Per prior: raw box regression, objectness and class scores               |
| Architecture | CSPDarknet backbone, PAFPN neck, decoupled YOLOX head (YOLOX-S defaults) |
| Datasets     | T4Dataset scenes annotated in 2D                                         |

## Available Configurations

Experiment configs live under `autoware_ml/configs/tasks/detection2d/yolox/` and are passed to
the CLI as the path relative to `tasks/`.

| Config Name                                              | Backbone   | Purpose                                             |
| -------------------------------------------------------- | ---------- | --------------------------------------------------- |
| `detection2d/yolox/t4dataset/yolox_s_960x960_tlr`        | CSPDarknet | Whole-image traffic light detector, YOLOX-S         |
| `detection2d/yolox/t4dataset/resnet18_960x960_tlr`       | ResNet-18  | The same detector with a ResNet-18 backbone         |
| `detection2d/yolox/t4dataset/cspnext_s_960x960_tlr`      | CSPNeXt-S  | The same detector with a CSPNeXt-S backbone         |
| `detection2d/yolox/t4dataset/cspnext_s_coco_960x960_tlr` | CSPNeXt-S  | CSPNeXt-S with the input convention of COCO weights |

All detect two classes, `traffic_light` and `pedestrian_traffic_light`, at 960x960. They read
the scenario lists `db_tlr_v1.yaml` to `db_tlr_v7.yaml` below `t4dataset/detection2d/` of
`AUTOWARE_ML_SCENARIOS_PATH` and the scenes below `t4dataset/` of `AUTOWARE_ML_DATA_PATH`, see
[Configuration](../user-guide/configuration.md#external-configs-and-scenario-lists). The lists
name no validation scenes, so the runs validate on the test scenes
(`datamodule.splitter.split_fallbacks`).

The config tree separates what two detectors share from what tells them apart:

| Config                            | Holds                                                               |
| --------------------------------- | ------------------------------------------------------------------- |
| `base.yaml`                       | Detector, neck, head, optimizer, schedule, averaged weights, export |
| `backbone_csp_darknet_s.yaml`     | CSPDarknet of YOLOX-S and its input convention                      |
| `backbone_resnet18.yaml`          | ResNet-18 of torchvision and its input normalization                |
| `backbone_cspnext_s*.yaml`        | CSPNeXt-S of RTMDet and the input convention of its weights         |
| `t4dataset/base_960x960.yaml`     | Input size, training and test pipelines, metric suites              |
| `t4dataset/base_960x960_tlr.yaml` | Traffic light database, taxonomy and validation schedule            |

A backbone returns its feature maps at strides 8, 16 and 32 and the block tells the neck their
channels, so a leaf with another backbone composes another block and keeps everything else.

CSPNeXt-S has two published weights of MMDetection (Apache-2.0), downloaded into the torch hub
cache by name. `cspnext_s_960x960_tlr` with `model.backbone.weights=cspnext_s_imagenet_600e`
starts from the ImageNet classifier, which reads RGB. `cspnext_s_coco_960x960_tlr` with
`model.backbone.weights=rtmdet_s_coco_300e` starts from the backbone of the RTMDet-S detector
trained on COCO, which reads BGR; its neck and head are not used.

## Training

The configs default to the recipe of the YOLOX reference implementation for YOLOX-S:

- SGD with Nesterov momentum 0.9 and weight decay 5e-4, not applied to biases and
  normalization weights
- learning rate `base_lr_per_image * global_batch_size`, 0.01 for 64 images; set
  `global_batch_size` to `batch_size` times the number of GPUs
- 300 epochs: 5 of quadratic warmup, a cosine down to 5 % of the rate, and the last 15 at
  that minimum (`trainer.max_epochs`, `warmup_epochs`, `no_aug_epochs`)
- mosaic, random affine, mixup, HSV shift and horizontal flip; the first three stop for the
  last `no_aug_epochs`, when the L1 box loss joins the loss
- an exponential moving average of the weights, which validation, test and export use
- mixed precision (`trainer.precision: 16-mixed`), the losses and the assignment in fp32

```bash
autoware-ml train --config-name detection2d/yolox/t4dataset/yolox_s_960x960_tlr
```

To start from the COCO weights of the reference implementation, convert its checkpoint once
and pass it as weights. The class outputs are trained for 80 classes, so they are skipped and
keep their initialization:

```bash
python -m autoware_ml.scripts.tools.yolox.convert_weights yolox_s.pth yolox_s_coco.ckpt

autoware-ml train \
    --config-name detection2d/yolox/t4dataset/yolox_s_960x960_tlr \
    --weights yolox_s_coco.ckpt --skip-mismatched-weights
```

`convert_weights --parts` keeps a part of the weights only. `--parts backbone` writes a
checkpoint that initializes the backbone and leaves the neck and the head at their own
initialization; `--parts neck head` writes one that initializes the neck and the head below
another backbone, which keeps its own weights:

```bash
python -m autoware_ml.scripts.tools.yolox.convert_weights yolox_s.pth yolox_s_coco_backbone.ckpt \
    --parts backbone
python -m autoware_ml.scripts.tools.yolox.convert_weights yolox_s.pth yolox_s_coco_neck_head.ckpt \
    --parts neck head

autoware-ml train \
    --config-name detection2d/yolox/t4dataset/resnet18_960x960_tlr \
    model.backbone.weights=IMAGENET1K_V1 \
    --weights yolox_s_coco_neck_head.ckpt --skip-mismatched-weights
```

For a short schedule, set the validation cadence along with the epochs:

```bash
autoware-ml train \
    --config-name detection2d/yolox/t4dataset/yolox_s_960x960_tlr \
    trainer.max_epochs=60 no_aug_epochs=10 warmup_epochs=3 \
    "callbacks.dynamic_validation.epoch_intervals=[[0,5],[50,1]]"
```

## Evaluation

```bash
autoware-ml test \
    --config-name detection2d/yolox/t4dataset/yolox_s_960x960_tlr \
    --weights mlruns/yolox/T4Dataset-tlr-v1.0.0/t4dataset/yolox_s_960x960_tlr/<run_id>/artifacts/checkpoints/best.ckpt
```

The metric suite follows the COCO protocol on boxes mapped back to the pixels of the original
image, and slices by box size instead of by the COCO area ranges. The size of a box is the
square root of its area in original pixels, and the bins are a config value
(`metric_params.detection2d.size_bins`):

| Key under `<stage>/det2d/`                   | Value                                                                  |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| `ap`, `ap50`, `ap75`                         | Average precision over IoU 0.50:0.95, at 0.50 and at 0.75, class mean  |
| `ap_<class>`, `ap50_<class>`, `ap75_<class>` | The same per class                                                     |
| `ap_size_<bin>`, `ap50_size_<bin>`           | Average precision of the boxes of a size bin                           |
| `ap_size_<bin>_<class>`                      | The same per class                                                     |
| `ap_small`                                   | `ap_size_<bin>` of the bin `metric_params.detection2d.small_bin` names |
| `recall`, `precision`, `recall_size_<bin>`   | At the operating score threshold (0.35) and IoU 0.50                   |
| `num_gt`, `num_gt_size_<bin>`                | Ground truth boxes behind the values                                   |

The best checkpoint is selected by `val/det2d/ap_small`.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection2d/yolox/t4dataset/yolox_s_960x960_tlr \
    --weights mlruns/yolox/T4Dataset-tlr-v1.0.0/t4dataset/yolox_s_960x960_tlr/<run_id>/artifacts/checkpoints/best.ckpt
```

The exported graph has the static input `images` of shape `(1, 3, height, width)` and the
output `output` of shape `(1, num_priors, 5 + num_classes)`. A row holds the box regression
`(dx, dy, log_w, log_h)` relative to the grid cell and the stride of its prior, the objectness
and the class scores, both after a sigmoid. The priors are ordered from the stride 8 level to
the stride 32 level, row by row. The runtime letterboxes the BGR image (aspect ratio kept, top
left aligned, padded with 114, no normalization), decodes the boxes and suppresses overlaps.

`model.export_background_class: true` adds a leading class score that is always zero, for a
runtime whose label list starts with a background entry. The traffic light configs set it.

The graph reads the channel order `model.export_input_channel_order` names, BGR by default,
whatever order the backbone reads: for a backbone trained on RGB images the graph reverses
the channels itself, and its normalization is part of the graph as well, so every backbone
is deployed behind the same input. `null` exports the order of the backbone.

An exported graph, or any ONNX file of the same contract, is scored on the test split with
the pipeline, the decoding and the metric suite of a config:

```bash
python -m autoware_ml.scripts.tools.yolox.evaluate_onnx \
    --config-name tasks/detection2d/yolox/t4dataset/yolox_s_960x960_tlr \
    +onnx_path=/path/to/detector.onnx +metrics_path=/path/to/metrics.json
```

`+onnx_class_indices=[1,2]` names the class output of the graph of every class of the config
when the two differ. An ONNX file of the reference implementation also converts into weights
with `convert_weights`, which rebuilds the folded batch normalizations as identities.

## Data Pipeline

Records come from `T4Dataset2D`, the T4 database of camera images with 2D boxes: one record
per annotated camera image, see [T4Dataset](../databases/t4dataset.md). The pipelines work on
the image resized to fit the network input:

| Transform              | Training | Test | Effect                                                      |
| ---------------------- | -------- | ---- | ----------------------------------------------------------- |
| `LoadImageAndResize`   | yes      | yes  | Reads the image and resizes it to fit, aspect ratio kept    |
| `Mosaic`               | yes      |      | Tiles four images around a random center                    |
| `RandomAffine`         | yes      |      | Rotates, scales, shears and shifts onto the input size      |
| `MixUp`                | yes      |      | Blends in a further image                                   |
| `HSVAugmentation`      | yes      |      | Shifts hue, saturation and value                            |
| `RandomHorizontalFlip` | yes      |      | Mirrors the image                                           |
| `Letterbox`            | yes      | yes  | Pads to the input size at the bottom and the right with 114 |
| `FilterSmallBoxes2D`   | yes      |      | Drops boxes whose shorter side fell below `min_box_side`    |

`Mosaic`, `RandomAffine` and `MixUp` read the strong augmentation switch the
`StrongAugmentationSchedule` callback turns off for the last epochs.

## Implementation

| Path                                                    | Description                                        |
| ------------------------------------------------------- | -------------------------------------------------- |
| `autoware_ml/models/detection2d/yolox.py`               | Detector and export wrapper                        |
| `autoware_ml/models/detection2d/heads/yolox.py`         | Head, losses and decoding                          |
| `autoware_ml/models/detection2d/task_modules/simota.py` | SimOTA label assignment                            |
| `autoware_ml/models/common/backbones/csp_darknet.py`    | CSPDarknet backbone                                |
| `autoware_ml/models/common/backbones/cspnext.py`        | CSPNeXt backbone                                   |
| `autoware_ml/models/common/necks/yolox_pafpn.py`        | PAFPN neck                                         |
| `autoware_ml/models/common/ema.py`                      | Exponential moving average of the weights          |
| `autoware_ml/models/detection2d/yolox_weights.py`       | Conversion of reference checkpoints and ONNX files |
| `autoware_ml/models/detection2d/yolox_onnx.py`          | ONNX backed detector for scoring exported graphs   |
| `autoware_ml/databases/t4dataset/t4dataset_2d.py`       | T4 database of camera images with 2D boxes         |
| `autoware_ml/datamodule/t4dataset/detection2d.py`       | Single camera 2D detection dataset                 |
| `autoware_ml/transforms/detection2d/`                   | Loading, augmentation and letterbox transforms     |
| `autoware_ml/metrics/detection2d/`                      | COCO style evaluation and metric suite             |
| `autoware_ml/callbacks/strong_augmentation_schedule.py` | Schedule ending the strong augmentation            |
| `autoware_ml/utils/schedulers/yolox_warm_cosine.py`     | Warmup, cosine and constant tail learning rate     |
| `autoware_ml/scripts/tools/yolox/`                      | Weight conversion and ONNX evaluation tools        |
| `autoware_ml/configs/tasks/detection2d/yolox/`          | Experiment configurations                          |

## Acknowledgment

The backbone, the neck, the head, the label assignment, the augmentations and the training
schedule are ported from the YOLOX reference implementation.

<!-- cspell:ignore Zheng Songtao Feng Zeming -->
- Repository: <https://github.com/Megvii-BaseDetection/YOLOX>
- License: Apache 2.0
- Paper: Ge, Z., Liu, S., Wang, F., Li, Z., and Sun, J. "YOLOX: Exceeding YOLO Series in 2021." arXiv:2107.08430, 2021.
