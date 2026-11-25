perf(ptv3): optionally pass the encoder point features to the heads as fp16

## Summary

`model.export_fp16_point_features` (default `false`) exports the per-stage features `point_feat_*` between the PTv3 / LitePT encoder graph and its heads as fp16 instead of fp32:

- `ptv3_encoder`: `PTv3EncoderExportModule` returns every `point_feat_*` as `Cast(to=FLOAT16)` of the fp32 stage feature, declared `FLOAT16`. The fp32 feature still feeds the next stage, so the encoder computes exactly as before.
- `ptv3_seg3d_head`: the `point_feat_*` inputs are declared `FLOAT16` and cast to fp32 once at the input (`as_float32`); the rest of the graph is unchanged.
- `ptv3_det3d_head`: the `point_feat_*` inputs are declared `FLOAT16`. The fp16 conversion (`deploy.onnx.modules.ptv3_det3d_head.precision: fp16`) leaves fp16 inputs as they are, and the new `_drop_identity_input_casts` in `autoware_ml/utils/onnx_precision.py` removes the fp16 -> fp16 cast left on them, so its consumers read the input directly. The converter's "already fp16" check is skipped for a model with fp16 graph I/O.

The fp16 det3d head casts its fp32 inputs to fp16 first, and a seg3d head engine built with the TensorRT FP16 flag (weakly typed, as the measurements below) does the same. With the option the encoder engine writes the fp16 tensor it has instead of an fp32 copy, the heads read half the bytes, and the det3d head drops its input conversion. The outputs are the fp32 interface's outputs on fp16-rounded features: for those engines what they computed already. A seg3d engine kept in fp32 (the strongly typed TensorRT stage of this repository builds the fp32 seg3d ONNX that way) now sees fp16-rounded features.

Input and output names, ranks and sizes are unchanged; only the dtype of `point_feat_*` changes (and the exporter-generated name of the encoder outputs' channel axis, see the notes). With the option off, the three exported graphs are byte-identical to main's.

End-to-end GPU time (encoder + seg3d head + det3d head), v0.3.0 weights, main vs this option, `trtexec --fp16`, three interleaved build rounds per arm. Each round's time is the sum of the three engines' mean GPU compute times, each engine benched on its own inputs; the table gives the median over the rounds, the change is taken from the unrounded medians, and the spread is max - min over main's three rounds:

| GPU | Model | Input size | main | this option | change | main spread |
| --- | --- | --- | --- | --- | --- | --- |
| DRIVE Thor | PTv3 | max | 60.41 ms | 58.89 ms | -1.52 ms (-2.5 %) | 0.05 ms |
| DRIVE Thor | PTv3 | opt | 31.54 ms | 30.72 ms | -0.83 ms (-2.6 %) | 0.06 ms |
| DRIVE Thor | PTv3 | min | 7.66 ms | 7.66 ms | -0.01 ms | 0.07 ms |
| DRIVE Thor | LitePT | max | 50.90 ms | 49.50 ms | -1.40 ms (-2.8 %) | 0.25 ms |
| DRIVE Thor | LitePT | opt | 25.67 ms | 24.85 ms | -0.83 ms (-3.2 %) | 0.21 ms |
| DRIVE Thor | LitePT | min | 6.71 ms | 6.70 ms | -0.00 ms | 0.07 ms |
| RTX 4000 SFF Ada | PTv3 | max | 70.11 ms | 69.02 ms | -1.09 ms (-1.6 %) | 0.99 ms |
| RTX 4000 SFF Ada | PTv3 | opt | 38.21 ms | 37.70 ms | -0.51 ms (-1.3 %) | 0.09 ms |
| RTX 4000 SFF Ada | PTv3 | min | 10.00 ms | 10.10 ms | +0.10 ms | 0.13 ms |
| RTX 4000 SFF Ada | LitePT | max | 49.96 ms | 48.89 ms | -1.07 ms (-2.1 %) | 0.32 ms |
| RTX 4000 SFF Ada | LitePT | opt | 27.74 ms | 27.12 ms | -0.62 ms (-2.2 %) | 0.04 ms |
| RTX 4000 SFF Ada | LitePT | min | 9.27 ms | 9.39 ms | +0.13 ms | 0.03 ms |

Per engine at max input (median per engine over the rounds, so the three do not add up exactly to the end-to-end change):

| GPU | Model | encoder | seg3d head | det3d head |
| --- | --- | --- | --- | --- |
| DRIVE Thor | PTv3 | -0.34 ms | -0.82 ms | -0.30 ms |
| DRIVE Thor | LitePT | -0.23 ms | -0.80 ms | -0.32 ms |
| RTX 4000 SFF Ada | PTv3 | -0.08 ms | -0.68 ms | -0.29 ms |
| RTX 4000 SFF Ada | LitePT | -0.03 ms | -0.81 ms | -0.24 ms |

The saving grows with the point count. Per model and GPU:

| GPU | PTv3 | LitePT |
| --- | --- | --- |
| DRIVE Thor | faster at max and opt, unchanged at min | faster at max and opt, unchanged at min |
| RTX 4000 SFF Ada | faster at opt (max within its spread), min within its spread | faster at max and opt, slower at min |

On the RTX 4000 the encoder's output cast costs more than it saves at the smallest input: encoder at min PTv3 1.73 -> 1.87 ms, LitePT 1.11 -> 1.19 ms. The heads change little there (PTv3 -0.03 ms together, LitePT +0.05 ms in the det3d head). This is a trade-off of that GPU at small point counts, which is why the interface is an export option: its default can be chosen per platform and model.

- DRIVE Thor (DRIVE OS 7.0.5, TensorRT 10.14) and RTX 4000 SFF Ada (TensorRT 10.8), each arm on the same device as its main arm, with the same TensorRT, plugins and inputs; the heads of this option were fed the fp16 rounding of main's inputs. Engine outputs stay within the build-to-build variation of main's engines (after fp16 rounding).
- The engines were benched one at a time, not chained; the end-to-end time is the sum of the three engines' times, not a measurement of the chained pipeline.

## Migration

The option changes the engine interface. A runtime that runs the three engines, such as the Autoware PTv3 node, has to bind `point_feat_*` as fp16 buffers (2 bytes per value) on the encoder outputs and on both heads' inputs, and pass them from the encoder to the heads unchanged.

- Engines exported with and without the option cannot be mixed. A buffer of the wrong dtype bound by raw device pointer gives wrong results without an error, so the runtime should read each `point_feat_*` dtype from the engine at load time (`ICudaEngine::getTensorDataType`) and refuse a mismatch.
- Input files used to benchmark the heads on their own have to be fp16 for `point_feat_*`.
- With the option off (the default) nothing changes.

## Related Issues

- None.

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/contributing/pull-request-guidelines/).

## Additional Notes

- Off by default: `onnx_precision` promises fp32 graph I/O so inference code works with every precision without interface changes, and the option breaks that for these tensors. It is set per config (`model.export_fp16_point_features`, added to the PTv3 segmentation, detection and multi-task base configs).
- The device numbers above come from ONNX made by applying the same change to main's exports as a graph rewrite. The exports of this branch were checked against those graphs on the CPU (onnxruntime, three input sizes, bf16 attention casts set to fp32 on both sides, graphs with plugins compared up to the plugins): the same I/O names, dtypes and shapes, and bitwise-identical outputs for all six graphs of PTv3 and LitePT. The one difference: the exporter names the encoder outputs' channel axis after the op producing them, `Castpoint_feat_k_dim_1` instead of `Addpoint_feat_k_dim_1`.
- The LitePT v0.3.0 weights use the QK-norm attention option, which is not on main yet; the LitePT check ran with it applied on top of this branch.
- Tests: the fp16 encoder outputs are exactly `fp16` of the fp32 ones, each head fed them gives bitwise the same outputs as the fp32 head fed the fp16-rounded features, the exported ONNX declare `FLOAT16` only on `point_feat_*`, and the fp32 graphs gain no casts. PTv3 tests (`segmentation3d`, the PTv3 and LitePT `detection3d` tests, `multi`, `deployment`, `utils`): 570 passed; 2 BEVFusion config tests fail the same way on main in the test container (the `bev_pool` extension is not built there).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
