# perf(litept): optionally export norm1 as row statistics with the affine folded into qkv

## Summary

`model.export_norm1_row_stats` (default `false`) makes the exported LitePT encoder and segmentation head compute the `norm1` LayerNorm of their attention blocks without a `LayerNormalization` node.

In every attention block of the LitePT encoder (the four blocks of stages 3 and 4) and of the LitePT segmentation head (the decoder blocks `dec3/block0` and `dec2/block0`), `norm1` normalises the block input, the attention gathers the normalised rows into its windows (a row `Gather` over `patch_order`), and `qkv` projects them. With the option, `Block.fold_norm1_into_qkv` (`encoders/ptv3.py`) rewrites this for the export copies of the encoder and the head:

- The LayerNorm's centring and affine are folded into the `qkv` weights, `W' = W diag(gamma) (I - 11^T / C)` and `b' = b + W beta` (computed in float64, stored in float32).
- `norm1` becomes `Norm1RowStatistics`, which only computes the reciprocal standard deviation of each row, `rstd = 1 / sqrt(var(x) + eps)`: seven nodes, `ReduceMean`, `Sub`, `Mul`, `ReduceMean`, `Add`, `Sqrt`, `Div`, in the scope `.../norm1/norm1.0/stats_fp32/`.
- The attention gathers the un-normalised rows and their `rstd` and multiplies them before `qkv`.

The result is the same in real arithmetic. Per block, one `LayerNormalization` fewer and 9 nodes more: `ptv3_encoder` 654 -> 686 nodes and `ptv3_seg3d_head` 350 -> 366 nodes for LitePT; graph inputs and outputs unchanged. `ptv3_det3d_head`: unchanged.

The change comes in two commits: the encoder, then the segmentation head (`PTv3Seg3DHeadExportModule` takes the same option).

**The statistic layers must be built in fp32, 28 in the encoder and 14 in the segmentation head**, with the rest of the engines in fp16; the accuracy below holds only for such builds. With `trtexec`, for each of the two engines:

```bash
trtexec --onnx=ptv3_encoder.onnx --fp16 --precisionConstraints=obey \
  --layerPrecisions=<name>:fp32,... --layerOutputTypes=<name>:fp32,...
```

The layer names are the node names; `norm1_row_statistic_node_names` (`ptv3_base.py`) returns them for either graph (every node whose name contains `/stats_fp32/`). autoware-ml's own TensorRT export builds a strongly typed network from the fp32 ONNX, so it needs nothing extra; the requirement is for fp16 builds such as the Autoware node's.

LitePT only: `PTv3BaseModel` refuses the option for any encoder other than `LitePTEncoder` (see Additional Notes).

GPU time on DRIVE Thor (TensorRT 10.14), LitePT, `trtexec --fp16` with the statistic layers pinned to fp32, LayerNorm vs row statistics, paired differences over 5 rounds at the max / opt / min input sizes (spread: range of the paired differences). The encoder and the segmentation head were measured in separate sessions:

| Measurement                         | max                    | opt                    | min                    |
| ----------------------------------- | ---------------------- | ---------------------- | ---------------------- |
| LitePT encoder                      | -0.65 to -0.75 ms      | -0.17 to -0.20 ms      | 0.00 to -0.02 ms       |
| LitePT end to end (encoder), median | -0.66 ms (spread 0.09) | -0.14 ms (spread 0.08) | +0.01 ms (spread 0.10) |
| LitePT segmentation head, median    | -0.38 ms (spread 0.03) | -0.08 ms (spread 0.07) | 0.00 ms                |

Encoder: faster in every round at every size; the heads were within ±0.05 ms; end to end at max 46.16 -> 45.49 ms. Segmentation head alone: 12.70 / 5.01 / 0.35 ms with LayerNorm at max / opt / min.

Accuracy at the largest input (202,000 points), 5 builds per arm:

- Encoder outputs against a PyTorch fp32 reference of the same encoder, row statistics / LayerNorm error ratio: 0.99 (mean abs) and 0.98 (99.9th percentile) over all outputs; 1.00 for stages 0-2 (unchanged), 0.96-0.97 for stages 3 and 4. Segmentation labels, from the device encoder outputs chained into the fp32 segmentation head, against the all-fp32 chain: median 13 of 202,000 points differ with either arm (LayerNorm 10-17 per round, row statistics 11-15).
- Segmentation head outputs against onnxruntime fp32 of the original head on the same inputs: differing labels median 156 (max 162) with row statistics, 152 (max 157) with LayerNorm; class probabilities mean abs error median 1.01e-4 vs 9.95e-5 (LayerNorm max 1.01e-4), max abs error median 0.035 with both. This is at the upper edge of the LayerNorm builds' range, not clearly better or worse.

## Related Issues

- None.

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/contributing/pull-request-guidelines/).

## Additional Notes

- Why LitePT only: the same rewrite of PTv3's ten encoder attention blocks was 1.9 ms faster at max on the same device, but its encoder error grew (mean abs 1.16-1.22x per output, 99.9th percentile up to 2.24x) and the chained segmentation labels got worse (median 54 -> 70 differing points). PTv3's `norm1` rows have a large mean relative to their spread (`|mean| * rstd` up to 0.68 at the largest input, LitePT encoder 0.017, LitePT segmentation head 0.074), and the fp16 `qkv` GEMM with the folded centring has to cancel it.
- Why the pins: the statistics in fp16 lose accuracy; this branch was only measured with them in fp32. A builder that cannot pin layers by name should not use the option.
- The device numbers come from ONNX made by applying the same change to the exported graphs as a graph rewrite; the engines also had other build and export changes, the same in both arms. This branch's export (v0.3.0 LitePT weights, CPU) was checked against that rewrite:
  - The rewrite applied to main's encoder / segmentation head and this branch's encoder / segmentation head are the same graphs with node and tensor names ignored (686 / 366 nodes; `W'` and `b'` bitwise equal), and the branch's 28 + 14 statistic nodes map one to one onto the rewrite's.
  - Their outputs are bitwise equal at three input sizes (16,000 to 202,000 points; onnxruntime, sparse-convolution plugins emulated for the encoder). Against main: encoder stages 0-2 bitwise, stages 3 and 4 within 6e-6 (relative 1.1e-6); segmentation head class probabilities within 2.5e-6, no label changes; fp32 rounding.
  - With per-head QK-norm in the LitePT attention (which the weights were trained with; not on main), the branch's encoder outputs are bitwise equal to the fp32 outputs of the encoder ONNX that was timed, at all three sizes. The branch's segmentation head is within 4.5e-6 (class probabilities, no label changes) of the head ONNX that was timed; that ONNX had other export changes, and the branch's head without the option differs from its source head by as much (up to 5.4e-6).
  - The statistic node names differ from the rewrite's (`.../stats_fp32/ReduceMean_1` here, `.../stats_fp32/var` there), and the folded blocks of each graph share one `eps` and one `1` initializer, where the rewrite had one per block: the exporter replaces equal initializers by `Identity` nodes, which sharing avoids.
- With the option off (the default, or set to `false`), the PTv3 and LitePT exports (encoder, segmentation head, detection head) are identical to main's apart from `export_date` and `producer_version` (the exporting commit). With it on, the LitePT detection head is too, and the LitePT encoder is the same as with the encoder commit alone.
- Tests: the row statistics normalise like `layer_norm`; the folded `qkv` of the scaled, gathered rows equals `qkv` of the gathered LayerNorm output (float64, with repeated rows as in padded windows); the fold refuses post-norm and non-attention blocks and reaches exactly the attention blocks of the encoder and the segmentation head; the option is off by default, reaches only the export copies of the encoder and the segmentation head and is refused for the PTv3 encoder; the folded encoder and segmentation head graphs give the unfolded ones' outputs; each export has the expected operators, 7 statistic nodes per block under `/stats_fp32/` with initializer constants, and unchanged graph inputs and outputs. PTv3 tests (`segmentation3d`, `detection3d/tests`, `multi`, `deployment`, `utils`): 649 passed (642 on main, 7 new); 2 BEVFusion config tests fail the same way on main in the test container (the `bev_pool` extension is not built there).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
