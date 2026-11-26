# perf(ptv3): hand the encoder's stage-2/3 rulebooks to the seg3d head on export

## Summary

`model.export_seg3d_rulebook_handoff` (default `false`) lets the exported PTv3 seg3d head reuse the submanifold-convolution rulebooks the encoder has already built, instead of building them a second time.

The first block of a decoder stage with blocks starts with a submanifold convolution (`cpe`), which builds its rulebook (`autoware::GetIndicePairsImplicitGemm`) from the stage's voxel coordinates. The encoder stage of the same level has built the same rulebook already: same voxels in the same order, same sparse shape, same convolution attributes. In the PTv3 configs (`dec_depths: [0, 0, 1, 1]`) this is the case for stages 2 and 3, so the seg3d head graph rebuilds two rulebooks the encoder graph holds. With the option:

- `ptv3_encoder`: outputs 1, 2 and 3 of the stage-2 and stage-3 `GetIndicePairsImplicitGemm` (the ones `ImplicitGemm` reads) become six extra graph outputs `rulebook_enc{2,3}_{pairs,point_mask,mask_argsort}` (`INT32`), after the `point_feat_*` outputs. No node changes; the encoder computes exactly as before.
- `ptv3_seg3d_head`: the six tensors are inputs, after the other inputs. The head's two `GetIndicePairsImplicitGemm` nodes and the coordinate preparation that only fed them (18 nodes) are gone (202 → 182 nodes), and with them the inputs `serialized_pooling_1_grid_coord` and `serialized_pooling_2_grid_coord`. The two `ImplicitGemm` nodes read the new inputs in the slots of the old rulebook.
- `ptv3_det3d_head`: unchanged.

In PyTorch the head's export module puts the handed-in rulebook into the indice cache of the stage's sparse tensor under its convolution's key, so spconv reuses it and the exporter traces no rulebook build. Which stages are handed off is derived from the model (`seg3d_rulebook_handoff_stages`: decoder stages whose first submanifold convolution matches the encoder's of the same level), and a stage keeps its `grid_coord` input if its attention reads coordinates (RoPE or relative position encoding). The LitePT decoder has no convolution, so its export does not change with the option.

GPU time on DRIVE Thor (TensorRT 10.14.1), `trtexec --fp16`, main vs this option, paired differences over 5 rounds, at the max / opt / min input sizes:

| Measurement | max | opt | min |
| --- | --- | --- | --- |
| PTv3 seg3d head | -0.54 ms | -0.42 ms | -0.12 ms |
| PTv3 encoder | +0.07 ms (noise) | | |
| PTv3 encoder + seg3d head | -0.49 ms | -0.39 ms | -0.12 ms |
| PTv3 end to end, on top of the other kept export optimizations | -0.51 ms | -0.33 ms | -0.11 ms |

The saving is the rulebook build and its coordinate preparation in the head, so it grows with the point count but remains at the smallest input. The rulebooks the encoder passes are bitwise identical to the ones the head builds itself, checked on the device at three input sizes and in every timed round, so the head computes the same outputs.

## Migration

The option changes the engine interface. A runtime that runs the encoder and the seg3d head, such as the Autoware PTv3 node, has to pass the six `INT32` tensors `rulebook_enc{2,3}_{pairs,point_mask,mask_argsort}` from the encoder engine's outputs to the seg3d head engine's inputs unchanged (device buffers, no copy or conversion needed).

- The tensors are in the sparse-convolution plugin's internal rulebook layout, not in a layout of the model: the plugin pads the pairs (for example 28 × 68,096 for 68,000 voxels, where the PyTorch export traces 27 × 68,000). Every axis of them is therefore declared dynamic in both graphs; the runtime takes each size from the encoder engine's output shape and sets it as the head's input shape.
- A change of that layout in the plugin, or another sparse-convolution backend (for example spconv's plugins), produces different tensors and needs its own variant of this interface. The option is per config, so each deployment target can choose.
- Engines exported with and without the option cannot be mixed.
- Input files used to benchmark the seg3d head on its own need the six rulebook tensors, dumped from the encoder engine.
- With the option off (the default) nothing changes: the three exported graphs are identical to main's.

## Related Issues

- None.

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/contributing/pull-request-guidelines/).

## Additional Notes

- Off by default because it changes the graph I/O. It is set per config (`model.export_seg3d_rulebook_handoff`, added to the PTv3 segmentation and multi-task base configs; the LitePT configs inherit it and are unaffected).
- Related: tier4/t4sparse#49 (and the stack after it) makes `torch.onnx.export` write t4sparse's own `SubMRulebook` and `SubMConv` nodes instead of the `GetIndicePairsImplicitGemm` / `ImplicitGemm` pair. If that lands in the PTv3 export, the handed-off tensors become `SubMRulebook` outputs and this interface has to follow; the tensors passed are listed in one table (`RULEBOOK_HANDOFF_FIELDS` in `ptv3_base.py`), the rest of the change is independent of which ones they are.
- Touches the same code as the fp16 point-feature interface (`perf(ptv3): optionally pass the encoder point features to the heads as fp16`, a separate change): both change `PTv3EncoderExportModule` and the `PTv3BaseModel` constructor, so expect a small textual conflict. They are independent of each other; the last row of the table was measured with the hand-off added to the other kept export optimizations.
- The device numbers come from ONNX made by applying the same change to main's exports as a graph rewrite. This branch's exports (v0.3.0 PTv3 weights) were checked against that rewrite on the CPU: the same node removals (the two `GetIndicePairsImplicitGemm` and the same 18 preparation nodes, no node added), the encoder byte-identical to main's apart from the six renamed outputs, the same input and output names, dtypes and dimension names, and bitwise-identical head outputs to main's head at three input sizes (onnxruntime, sparse-convolution plugins emulated). The one difference: the exporter now declares the class axis of `pred_probs` as `16` instead of a symbolic name. In PyTorch, with the v0.3.0 weights on the same three inputs (250 to 68,000 voxels at the handed-off stages), the encoder's six tensors equal the rulebooks the head builds itself and the head's outputs are bitwise identical with and without the option.
- With the option off, the PTv3 and LitePT exports (encoder, seg3d head, det3d head) are identical to main's apart from the export date; with it on, the LitePT exports are too.
- Tests: off by default and only the rulebook names change once set (segmentation and joint model; the det3d head, the detection model and LitePT unchanged); a stage keeps the coordinates its attention reads; the handed-off rulebooks are bitwise the head's own and the head's outputs bitwise unchanged; the deployment with the option reproduces `forward()` (segmentation and joint model); the exported head has no `GetIndicePairsImplicitGemm` and its `ImplicitGemm` reads the encoder's `INT32` outputs. PTv3 tests (`segmentation3d`, the PTv3 and LitePT `detection3d` tests, `multi`, `deployment`, `utils`): 569 passed; 2 BEVFusion config tests fail the same way on main in the test container (the `bev_pool` extension is not built there).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
