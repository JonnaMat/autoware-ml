# perf(litept): project before the pooling gather at the first three stages

## Summary

Since #288, `SerializedPooling` gathers the pooling rows before the projection (`proj(feat[indices])`). That is the faster order when the op that produces `feat` is the attention MLP residual, as at every PTv3 stage: TensorRT fuses the row gather into that kernel cheaply.

LitePT's first three stages are convolution-only, so the input of `enc1/down`, `enc2/down` and `enc3/down` ends in the cpe LayerNorm plus the block residual. There, the fused gather makes TensorRT split the LayerNorm into two kernels with an fp32 `[N, C]` tensor and an fp32 `[N, 1]` tensor between them. Projecting first (`proj(feat)[indices]`) keeps the LayerNorm in one kernel with only fp16 I/O, and the gather becomes a standalone fp16 gather of the projected rows.

`SerializedPooling` gains `project_first` (default `False`). `PointTransformerV3Encoder` sets it on a stage's downsampler when the previous stage has no attention (`enc_attn[stage - 1]` is false). With LitePT's `enc_attn: [false, false, false, true, true]`, that is `enc1..enc3/down`; `enc4/down` follows an attention stage and keeps gathering first, and PTv3 (attention at every stage) is unchanged. No config change is needed.

The two orders compute the same rows: `proj` is a row-wise affine map and `indices` is a permutation of all `N` rows, so `proj(x[indices]) == proj(x)[indices]` and the `Gemm` still runs on `N` rows. Training, the eager forward and the checkpoints are unchanged.

LitePT GPU compute time of encoder + seg3d head + det3d head (`trtexec --fp16`, v0.3.0 checkpoint; each engine's mean compute time, summed per round), median over 5 interleaved rounds, each with its own build:

| GPU | Input size | main | this PR | change per round |
| --- | --- | --- | --- | --- |
| DRIVE Thor | max | 47.09 ms | 46.53 ms | -0.49 ms (spread 0.13) |
| DRIVE Thor | opt | 22.86 ms | 22.76 ms | -0.19 ms (spread 0.16) |
| DRIVE Thor | min | 5.44 ms | 5.43 ms | +0.00 ms (spread 0.08) |
| RTX 4000 SFF Ada | max | 47.23 ms | 46.71 ms | -0.34 ms (spread 0.52) |
| RTX 4000 SFF Ada | opt | 25.70 ms | 25.56 ms | -0.15 ms (spread 0.19) |
| RTX 4000 SFF Ada | min | 8.28 ms | 8.26 ms | -0.03 ms (spread 0.05) |

LitePT encoder alone, change per round: DRIVE Thor -0.51 / -0.22 / -0.02 ms, RTX 4000 SFF Ada -0.44 / -0.16 / -0.02 ms (max / opt / min).

- DRIVE Thor: TensorRT 10.14.1. RTX 4000 SFF Ada: TensorRT 10.8.0. Both arms ran on the same device with the same plugins, inputs and weights, with the rounds interleaved.
- "Change per round" is the median over the rounds of this PR minus main within the same round, which removes the drift between rounds; that is why it differs from the difference of the two medians. The spread is the range of the per-round changes. At max and opt the change is negative in every round on both GPUs (DRIVE Thor max -0.47 to -0.60 ms, opt -0.08 to -0.24 ms; RTX 4000 max -0.18 to -0.70 ms, opt -0.09 to -0.28 ms). At min it is within noise (-0.06 to +0.02 ms).
- The seg3d and det3d head engines are unchanged; their per-round change is at most 0.03 ms. The gain is in the encoder and grows with the point count.
- In the DRIVE Thor layer profile at max input, the three LayerNorm windows lose 1.39 ms (two kernels → one per stage), the three new standalone gathers add 0.93 ms, and the `Gemm`s on the ungathered input are 0.07 ms faster: -0.52 ms per-layer time over the three windows.

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/contributing/pull-request-guidelines/).

## Additional Notes

- No checkpoint change. The encoder ONNX keeps its inputs, outputs and node count; in `enc1..enc3/down` the `Gather` now consumes the `proj` `Gemm` output instead of feeding it. `enc4/down`, the seg3d and det3d heads, and every PTv3 graph are unchanged.
- Checked on an export from this branch and from main, with a LitePT encoder with the default stage layout (5 stages, `enc_attn` false/false/false/true/true), random weights and a random 20,000-point cloud (CUDA, fp32): main's ONNX has `Gather → Gemm` at all four downsamplers; this branch has `Gemm → Gather` at `enc1..enc3/down` and `Gather → Gemm` at `enc4/down`, 969 nodes on both sides, same graph inputs and outputs. The export module's `point_feat_0..4` are bit-identical between the two.
- The timings were measured on the v0.3.0 encoder ONNX with the same reordering applied at the graph level (same node names, attributes and initializers; only the `Gather`/`Gemm` wiring of the three stages differs), not on an export from this branch. In onnxruntime on CPU, fp32, that rewrite reproduces the original encoder's outputs exactly.
- Device outputs: on the RTX 4000, every output of this PR's engines matches a main build bit for bit. On DRIVE Thor, the encoder outputs differ from main by at most as much as two main builds differ from each other (max abs 0.020 vs 0.020); over all outputs the largest float difference is 0.031 vs 0.020 between two main builds, and the det3d integer outputs (labels, indices) differ between builds by a similar amount in both arms (up to 234 elements against main vs 189 between two main builds).
- Tests: a new test checks that `project_first` gives bit-identical pooled features to the gather-first order, and one checks that `project_first` is set only after a convolution-only stage (default LitePT: `[True, True, True, False]`; PTv3: all `False`). `segmentation3d`, `detection3d` and `utils/config` tests pass (683), except the BEVFusion tests that need the `bev_pool`/`diff_iou_rotated` CUDA extensions, which were not built in this environment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
