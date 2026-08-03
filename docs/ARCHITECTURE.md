# Architecture

```text
AttentionRequest -> dispatch_shape -> CoreKey(window, grid, variant) -> ArtifactRegistry -> CoreInfo
  -> SingleModelArtifact | StagedHeadMergeArtifact -> NcnnModelCache -> executor
  -> baseline single-model | hybrid four-head -> HostHeadMerge -> output projection -> reverse/crop -> BTHWC
```

`MaskMode`: ArtifactReference | PaddingValidity. `RopeMode`: ArtifactReference | DynamicMMRoPE3D. Cache keys use param/bin paths.

Phase 12 `HybridDiTBlockRuntime` (staged DiT block): `DiTBlockInput(vid, txt, emb[B,6*C], vid_shapes, txt_lengths)` -> branch RMSNorm -> AdaSingle modulation -> qkv -> per-head q/k norm -> MM-RoPE3D -> window packing (per segment `[vid_win; txt]`) -> per-head varlen attention (ncnn `head{h}` at fixed `q_max=128`) -> window reverse + txt mean merge -> attention projection/gate/residual -> mlp_norm -> AdaSingle -> SwiGLU (gate/up/act/out) -> gate/residual -> `DiTBlockOutput`. Stable stages run in ncnn (Linear as dynamic Gemm, RMSNorm no-affine dim 512, q/k norm affine dim 128, rope_apply, heads, swiglu_act); dynamic windowing/packing/merge/modulation/residuals run on the host. Registry metadata `execution: "staged_dit_block"` + stage list; last-layer variant is vid_only for mlp_norm/ada/mlp/swiglu and shares the base attention artifacts.

Phase 13 real-width staged DiT block (READY): the SAME `HybridDiTBlockRuntime` /
`ArtifactRegistry` / `NcnnModelCache` at real 3B width C=2560/H=20/HD=128
(variant `realwidth_seedvr2_dit_block_c2560_h20_hd128_random`). Metadata-driven:
no runtime copy. 19 stage models (18 base + one weight-free `head` shared by 20
attention heads); qk-norm stages export 0-byte ncnn bins because the real
all-ones RMSNorm affine weights are folded by pnnx. Heavy artifacts are
git-ignored (`.phase13_local_artifacts/`); the committed inventory manifest and
weight-mapping contract describe them. Last-layer reuses the same 19 models
(vid_only control flow).

Phase 14 real-weight single-block hybrid visual milestone (READY): keeps the
official PyTorch SeedVR2 pipeline and replaces exactly one real
`NaMMSRTransformerBlock` (middle block `floor(32/2)` = 16) with the
`HybridDiTBlockRuntime` via a host adapter
(`tests/seedvr2_real_weight_hybrid/real_block_adapter.cpp`, modes
capture/pytorch/ncnn; `ncnn_adapter_call_count>0`, `original_block_call_count==0`,
no silent fallback). Real-weight stage artifacts live in git-ignored
`.phase14_local_artifacts/`; visual outputs in `.phase14_visual_outputs/`.

Phase 15 real-weight multi-block streaming hybrid (READY): a streaming adapter
(`multiblock_streaming_adapter`) replaces 1/2/4 contiguous real blocks
(`[16]` / `[16,17]` / `[15..18]`) with the ncnn `HybridDiTBlockRuntime` under a
strict per-block lifecycle `LOAD -> EXECUTE -> UNLOAD`: after each block the
runtime is destroyed, the model cache cleared to 0, resident bytes 0, and the
artifact files rename-probed (no leaked handles). `max_resident_weighted_blocks=1`
for every streaming run; resident mode is diagnostic only.

Phase 16 full SeedVR2 DiT streaming (READY): the streaming adapter
(`full_dit_streaming_adapter`, range 0..31) replaces the WHOLE 32-block DiT body
with ncnn, still one block resident at a time
(`max_resident_weighted_blocks=1`). Blocks 0..9 use the split 29-key layout
(separate `.vid.` / `.txt.` weights); blocks 10..31 use the shared 15-key
`.all.` layout; the last layer 31 runs at most once as the final block. The
adapter writes per-block outputs so the ncnn chain result can be replayed
through the official pipeline (visual harness `run_phase16_visual.py`,
`ChainHybrid`); a single strict adapter call replaces the whole chain with
`original_block_call_count=0` and no silent fallback. One gotcha: the official
`vid_out_ada` head modulation reuses the shared cache key `emb_repeat_0_vid`
(cache keys carry only idx + branch_tag, not layer/mode), so the visual harness
rebuilds that weight-independent entry exactly before the head executes.
