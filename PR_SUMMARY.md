# perf(ptv3): apply the seg3d classifier before the finest unpooling gather on export

## Summary

The seg3d head ends with the finest decoder unpooling (`dec0/up`) and the classifier. The unpooling gathers the projected coarse features through the pooling cluster and adds them to the projected skip features, and the classifier maps the sum from the decoder width (64) to the classes (16):

$$y = \big(s + c[\pi]\big) W^\top + b$$

with $s$ the projected skip features of the finest level, $c$ the projected features of the coarser level and $\pi$ the pooling cluster. A linear map distributes over the sum and commutes with a row gather, so

$$y = \big(s W^\top + b\big) + \big(c W^\top\big)[\pi]$$

with the bias added once. The export now computes the right-hand side: the gather moves 16 channels instead of 64, and the 64-channel sum at the finest level is never formed. The coarse level gets one extra 64 → 16 product.

`prepare_for_export` of `PTv3Seg3DDecoderHead` sets `defer_fusion` on the finest `SerializedUnpooling` when the finest decoder stage has no blocks (`dec_depths[0] == 0`, as in the PTv3 and LitePT configs), since only then is the classifier the next layer. The unpooling then returns the projected skip features with the projected coarse features and the cluster alongside, and the head fuses them after the classifier. Only the export copy changes; training, the eager forward and the checkpoint are unchanged, and a head with blocks on the finest stage exports as before.

Seg3d head GPU compute time (`trtexec --fp16`, mean), the v0.3.0 PTv3 and LitePT checkpoints, mean over 5 interleaved rounds, each with its own build:

| Model | Input size | main | this PR | change |
| --- | --- | --- | --- | --- |
| PTv3 | max | 11.935 ms | 11.608 ms | -0.33 ms (-2.7 %) |
| PTv3 | opt | 5.545 ms | 5.329 ms | -0.22 ms (-3.9 %) |
| PTv3 | min | 0.4610 ms | 0.4270 ms | -0.034 ms (-7.4 %) |
| LitePT | max | 14.687 ms | 14.417 ms | -0.27 ms (-1.8 %) |
| LitePT | opt | 6.128 ms | 5.928 ms | -0.20 ms (-3.3 %) |
| LitePT | min | 0.3726 ms | 0.3344 ms | -0.038 ms (-10.3 %) |

- NVIDIA DRIVE Thor, TensorRT 10.14, both arms on the same devkit with the same TensorRT, plugins and inputs. Each round runs both arms back to back; the per-round difference spreads over the rounds by 0.06 / 0.09 / 0.03 ms (PTv3 max / opt / min) and 0.03 / 0.08 / 0.02 ms (LitePT).
- Timed on DRIVE Thor only, and on a graph-level rewrite of the seg3d head ONNX that gives bit-identical outputs to this PR's export (see the notes below); an engine built from this PR's export has not been timed.

## Related Issues

- Touches the same region as the decoder unpooling batch-norm fold (`perf(ptv3): fold the decoder unpooling batch norms into the projections on export`, a separate change not yet opened as a PR): both change `SerializedUnpooling` and the `dec0/up` part of the seg3d head graph. They are independent (both based on main) and compose algebraically (the classifier reads the GELU outputs, which the fold leaves in place), but expect a small textual conflict in `SerializedUnpooling`, and the two have not been timed together.

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/contributing/pull-request-guidelines/).

## Additional Notes

- No checkpoint change. The seg3d head ONNX has the same inputs and outputs and the same, bit-identical initializers. Against main it loses the `dec0/up` `Gather` and `Add` and the classifier `Gemm` on the sum, and gains the classifier `Gemm` on the skip features (with the bias, same name), a bias-free product of the coarse features with the classifier weight (`Transpose` of the weight initializer + `MatMul`), the 16-channel `Gather` and the `Add` (PTv3 202 → 204 nodes, LitePT 358 → 360); every other node is unchanged. The encoder and det3d head ONNX are identical to main's apart from the export metadata.
- The rewrite is exact in real arithmetic, but the outputs are not bit-identical to main's: the fp32 sums run in a different order. In onnxruntime on CPU (fp32, 202,000 / 128,000 / 16,000 voxels) the probabilities match main's within 1.7e-6 with no label change, for LitePT on the whole head and for PTv3 with a Python emulation of its sparse-convolution plugins. On the device, against an fp32 reference of the seg3d head on the same input (PyTorch for PTv3, onnxruntime for LitePT; DRIVE Thor output dumps at max input, 202,000 voxels, median over the 5 rounds), the engine is as close as main's:

  | Model | | main | this PR |
  | --- | --- | --- | --- |
  | PTv3 | labels differing from the fp32 argmax | 1142 | 1189 |
  | PTv3 | mean $\|\Delta\|$ of the probabilities | 5.72e-4 | 5.85e-4 |
  | PTv3 | max $\|\Delta\|$ of the probabilities | 0.226 | 0.242 |
  | LitePT | labels differing from the fp32 argmax | 158 | 158 |
  | LitePT | mean $\|\Delta\|$ of the probabilities | 1.07e-4 | 1.09e-4 |
  | LitePT | max $\|\Delta\|$ of the probabilities | 0.037 | 0.038 |

  For PTv3, main's own count of differing labels varies from round to round up to 1360 (median 1142), so the difference is within its round-to-round range.
- The timings were measured on seg3d head ONNX rewritten at the graph level. In onnxruntime on CPU it gives bit-identical probabilities and labels to this PR's export on the three inputs above; it differs only in node names and in writing the bias-free product as one `Gemm` (`transB=1`, no bias) instead of `Transpose` of the weight + `MatMul`.
- Tests: the deferred unpooling reproduces the fused features bitwise; the seg3d head export module matches the eager head in fp32 within 1e-5 (on the CPU without decoder blocks, and on CUDA with a block on the coarser stage); only a blockless finest stage enables the rewrite; the exported ONNX gathers a `num_classes`-channel tensor through `pooling_cluster_0` instead of the decoder-width one. PTv3 tests (`segmentation3d`, the PTv3 and LitePT `detection3d` tests, `multi`, `deployment`): 276 passed at this branch head.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
