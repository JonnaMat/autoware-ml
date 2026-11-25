# perf(ptv3): fold the decoder unpooling batch norms into the projections on export

## Summary

`SerializedUnpooling.fold_batch_norm` in `autoware_ml/models/segmentation3d/encoders/ptv3.py` folds the batch norm of both decoder up projections (`proj` and `proj_skip`, each linear layer → batch norm → GELU) into the linear layer before it. In eval mode the batch norm is an affine map per output channel, so with $s = \gamma / \sqrt{\sigma^2 + \epsilon}$

$$\mathrm{BN}\big(\mathrm{Linear}(W, b)(x)\big) = \mathrm{Linear}\big(s W,\ s (b - \mu) + \beta\big)(x)$$

with $s$ scaling the rows of $W$ (computed in float64, then cast back to the parameter dtype). Each projection then runs as the linear layer followed by its GELU. `prepare_point_module_for_export` calls it on every `SerializedUnpooling`, which only the seg3d head has (in the segmentation and the multi-task export): its 4 up blocks lose their 8 `BatchNormalization` nodes. Only the export copy changes; the model and its checkpoint keep their parameters, and the encoder and det3d head exports are unchanged.

With the batch norm between them, TensorRT runs each GELU as its own pointwise kernel after the gemm. Once the linear layer feeds the GELU directly, the GELU fuses into the gemm: in the LitePT seg3d head engine (TensorRT 10.8, layer info) all 8 GELUs sit inside a gemm layer, where without the fold they ran as 4 standalone pointwise layers.

Seg3d head GPU compute time (`trtexec --fp16`, mean), the v0.3.0 PTv3 and LitePT checkpoints, mean over 5 interleaved rounds, each with its own build:

| Model | Input size | main | this PR | change |
| --- | --- | --- | --- | --- |
| PTv3 | max | 11.93 ms | 11.06 ms | -0.86 ms (-7.2 %) |
| PTv3 | opt | 5.59 ms | 5.12 ms | -0.47 ms (-8.5 %) |
| PTv3 | min | 0.454 ms | 0.439 ms | -0.015 ms (-3.2 %) |
| LitePT | max | 14.69 ms | 13.82 ms | -0.87 ms (-5.9 %) |
| LitePT | opt | 6.13 ms | 5.78 ms | -0.34 ms (-5.6 %) |
| LitePT | min | 0.374 ms | 0.348 ms | -0.026 ms (-6.9 %) |

- NVIDIA DRIVE Thor, TensorRT 10.14, both arms on the same devkit with the same TensorRT, plugins and inputs. Taken round by round (each round runs both arms back to back), the difference at max spreads by 0.03 ms (PTv3) and 0.10 ms (LitePT) over the rounds.
- Timed on DRIVE Thor only. The fusion above was read from the layer info of a TensorRT 10.8 build on an RTX 5080, which was not timed.

## Related Issues

- Follows #300, which folds the positional-encoding linear layer the same way.

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/contributing/pull-request-guidelines/).

## Additional Notes

- No checkpoint change. The seg3d head ONNX has the same inputs and outputs, and its nodes are main's minus the 8 `BatchNormalization` nodes (PTv3 202 → 194, LitePT 358 → 350); the other initializers are bit-identical. The encoder and det3d head ONNX are identical to main's apart from the export metadata.
- The fold is an exact rewrite in real arithmetic, but the engine output is not bit-identical to main's: the scaled weights are rounded to fp16 once, and the GELU runs in the gemm epilogue. Against an fp32 reference of the seg3d head on the same input (PyTorch for PTv3, onnxruntime for LitePT; DRIVE Thor output dumps at max input, 202,000 voxels, median over the 5 rounds), the folded engine is as close as main's:

  | Model | | main | this PR |
  | --- | --- | --- | --- |
  | PTv3 | labels differing from the fp32 argmax | 1223 | 1182 |
  | PTv3 | mean $\|\Delta\|$ of the probabilities | 5.85e-4 | 5.50e-4 |
  | PTv3 | max $\|\Delta\|$ of the probabilities | 0.273 | 0.286 |
  | LitePT | labels differing from the fp32 argmax | 159 | 155 |
  | LitePT | mean $\|\Delta\|$ of the probabilities | 1.08e-4 | 9.75e-5 |
  | LitePT | max $\|\Delta\|$ of the probabilities | 0.035 | 0.039 |

- The timings were measured on seg3d head ONNX folded at the graph level, which has the same nodes as this PR's export and folded weights within float32 rounding of it (largest relative difference 2e-7 in the weights); in onnxruntime on CPU the two give the same probabilities within 7e-6 for LitePT, and the same tensors within 5e-6 for PTv3 up to its first sparse convolution (which onnxruntime cannot run).
- Tests: the folded projections match the batch-norm projections in fp32 within 1e-5 and export without `BatchNormalization`; the seg3d head export module matches the unfolded one within 1e-5 and its ONNX has no `BatchNormalization` node. PTv3 tests (`segmentation3d`, the PTv3 and LitePT `detection3d` tests, `multi`, `deployment`): 274 passed at this branch head.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
