# Architecture

```text
AttentionRequest -> dispatch_shape -> CoreKey(window, grid, variant) -> ArtifactRegistry -> CoreInfo
  -> SingleModelArtifact | StagedHeadMergeArtifact -> NcnnModelCache -> executor
  -> baseline single-model | hybrid four-head -> HostHeadMerge -> output projection -> reverse/crop -> BTHWC
```

`MaskMode`: ArtifactReference | PaddingValidity. `RopeMode`: ArtifactReference | DynamicMMRoPE3D. Cache keys use param/bin paths.

Phase 12 `HybridDiTBlockRuntime` (staged DiT block): `DiTBlockInput(vid, txt, emb[B,6*C], vid_shapes, txt_lengths)` -> branch RMSNorm -> AdaSingle modulation -> qkv -> per-head q/k norm -> MM-RoPE3D -> window packing (per segment `[vid_win; txt]`) -> per-head varlen attention (ncnn `head{h}` at fixed `q_max=128`) -> window reverse + txt mean merge -> attention projection/gate/residual -> mlp_norm -> AdaSingle -> SwiGLU (gate/up/act/out) -> gate/residual -> `DiTBlockOutput`. Stable stages run in ncnn (Linear as dynamic Gemm, RMSNorm no-affine dim 512, q/k norm affine dim 128, rope_apply, heads, swiglu_act); dynamic windowing/packing/merge/modulation/residuals run on the host. Registry metadata `execution: "staged_dit_block"` + stage list; last-layer variant is vid_only for mlp_norm/ada/mlp/swiglu and shares the base attention artifacts.
