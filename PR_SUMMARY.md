# perf(ptv3): optionally export the seg3d label as a comparison fold instead of ArgMax

## Summary

`model.export_seg3d_label_fold` (default `false`) makes the exported seg3d head compute `pred_labels` without an `ArgMax` node.

The head ends in `pred_probs = Softmax(logits)` and `pred_labels = ArgMax(pred_probs, axis=1)` over the 16 classes. TensorRT 10.8 on Ada GPUs compiles that `ArgMax` to a separate top-k kernel (`__myl_Top`), which takes 0.74-0.76 ms at the largest input. With the option, `argmax_by_comparison_fold` (`ptv3_base.py`) computes the label as a running maximum over the class columns instead:

- `pred_probs` is split into its 16 columns; for each column after the first, a strict `Greater` against the running maximum and two `Where` update the maximum and its index (the last column only the index), and a `Squeeze` returns the `[N]` int64 labels. In the graph: 1 `Split`, 15 `Greater`, 29 `Where`, 1 `Squeeze` and 18 `Constant` in place of 1 `ArgMax`.
- The strict comparison keeps the first of several equal maxima, as `ArgMax` with `select_last_index=0` does, so the labels are identical for every NaN-free input, ties included.
- `ptv3_seg3d_head`: graph inputs, outputs and initializers unchanged; every other node unchanged. `ptv3_encoder` and `ptv3_det3d_head`: unchanged.
- PTv3 and LitePT both use it (LitePT inherits the option from the PTv3 configs).

TensorRT 10.8 fuses the fold into one elementwise kernel (0.02 ms at the largest input), and the engine has one layer fewer.

GPU time of the seg3d head on an Ada GPU (TensorRT 10.8), `trtexec --fp16`, ArgMax vs fold, paired differences over 5 rounds, at the max / opt / min input sizes:

| Measurement       | max      | opt      | min      |
| ----------------- | -------- | -------- | -------- |
| PTv3 seg3d head   | -0.49 ms | -0.20 ms | -0.03 ms |
| LitePT seg3d head | -0.52 ms | -0.20 ms | -0.04 ms |

Every round was faster at every size for both heads, and `pred_labels` matched the ArgMax engine's exactly in every round.

## Related Issues

- None.

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/contributing/pull-request-guidelines/).

## Additional Notes

- Off by default: on DRIVE Thor (TensorRT 10.14) the `ArgMax` already compiles to a cheap kernel (about 0.06 ms at the largest input), so no gain is expected there; this was not measured. Set it per config for Ada deployments.
- The device numbers come from ONNX made by applying the same change to the exported heads as a graph rewrite. The heads timed there also had the fp16 point-feature interface (a separate change); the fold only reads `pred_probs`, so it does not depend on it. This branch's exports (v0.3.0 PTv3 and LitePT weights) were checked against that rewrite on the CPU:
  - The rewrite applied to main's seg3d head and this branch's export have the same fold: the same operators in the same order with the same wiring and index constants. Differences: the `Split` has an explicit `split` input of 16 ones (the rewrite relies on the equal-split default), the index constants are `Constant` nodes instead of initializers, and the new nodes have different names. In the LitePT head 5 unscoped `Constant` nodes of the rest of the graph get different node names (`Constant_155` becomes `Constant_173`, and so on); their outputs and values are unchanged.
  - Main's head (ArgMax), the rewrite and this branch's head give bitwise-identical `pred_labels` and `pred_probs` at three input sizes (16,000 to 202,000 points; onnxruntime, sparse-convolution plugins emulated), for both models.
  - The fold sub-graph of this branch and the rewrite's give the same labels as `ArgMax` in fp32 and fp16 on random scores, scores with many ties, signed zeros, infinities and fp16 subnormals, and the fp16 probabilities from 10 device runs at the largest input.
  - Built with TensorRT 10.8 on a Blackwell desktop GPU (build only, no timing), this branch's heads have the fused `Slice…Greater…Select` kernel in place of `__myl_Top`, as the rewrite's had.
- With the option off (the default, or set to `false`), the PTv3 and LitePT exports (encoder, seg3d head, det3d head) are identical to main's apart from `export_date`. With it on, the encoder and det3d head are too.
- Tests: the fold equals `argmax` on ties and extreme values in fp32 and fp16 (16, 3 and 2 classes, and one class); it exports with the base configs' ONNX settings to the fold operators without `ArgMax` and runs on onnxruntime with `ArgMax`'s labels; the option is off by default and reaches the seg3d head graph (segmentation and joint model); the exported head with the option has the fold instead of the `ArgMax` and the same graph inputs and outputs; the deployment with the option reproduces `forward()` (segmentation and joint model). PTv3 tests (`segmentation3d`, the PTv3 and LitePT `detection3d` tests, `multi`, `deployment`, `utils`): 569 passed; 2 BEVFusion config tests fail the same way on main in the test container (the `bev_pool` extension is not built there).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
