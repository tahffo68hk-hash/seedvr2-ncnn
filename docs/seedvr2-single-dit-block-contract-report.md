# Phase 11: real SeedVR2 single DiT block interface / shape / execution contract

## Status

`Phase 11 = PARTIAL`. The real SeedVR2 DiT block (`NaMMSRTransformerBlock`) was
located and its full field-level interface, tensor layout, RoPE slicing,
modulation, attention, MLP, and residual order were extracted with source +
AST evidence, and the real upstream window partition math was executed. The
random-weight forward probe of the real block is **BLOCKED** because the
non-model Python deps it imports (`einops`, `diffusers`, `flash_attn`, `apex`,
`rotary_embedding_torch`) are not installed in the repo `.conda-cuda` env, and
the task forbids installing them. No hand-written substitute was used for the
real block probe. Phase 12 (synthetic single DiT block staged fidelity) is not
started.

## SeedVR2 commit and config source

- SeedVR2 repo: `D:\AIProjects\third_party\SeedVR2`, HEAD
  `e4de8c24441a67e1b7df56abea10645059bb1185`, clean (matches Phase 10.1).
- Config: `configs_3b/main.yaml` `dit.model` -> `NaDiT` (module
  `models.dit_v2.nadit`).
- `vid_in_channels=33`, `vid_out_channels=16`, `vid_dim=2560`, `txt_in_dim=5120`,
  `txt_dim=2560`, `emb_dim=15360` (`6*vid_dim`), `heads=20`, `head_dim=128`,
  `expand_ratio=4`, `norm=fusedrms`, `norm_eps=1e-5`, `ada=single`,
  `qk_bias=False`, `qk_norm=fusedrms`, `patch_size=[1,2,2]`, `num_layers=32`,
  `mm_layers=10`, `mlp_type=swiglu`, `block_type=mmdit_sr`, `window=(4,3,3)`,
  alternating `720pwin_by_size_bysize` / `720pswin_by_size_bysize`,
  `rope_type=mmrope3d`, `rope_dim=128`.

## Real block and call chain

- Block class: `NaMMSRTransformerBlock`
  (`models/dit_v2/nablocks/mmsr_block.py`; registry
  `nablocks/__init__.py: nadit_blocks = {"mmdit_sr": NaMMSRTransformerBlock}`).
  This is the only block type in `dit_v2`; the config selects it for all 32
  layers.
- Call chain (AST-verified):
  `VideoDiffusionTrainer.sample` (projects/video_diffusion_sr/infer.py)
  -> `NaDiT.forward(vid, txt, vid_shape, txt_shape, timestep)`
  -> `NaPatchIn` + text projection + `TimeEmbedding`
  -> 32x `NaMMSRTransformerBlock(vid, txt, vid_shape, txt_shape, emb, cache)`
  -> optional `vid_out_norm/ada` + `NaPatchOut` -> `NaDiTOutput(vid_sample)`.
- Upper-level tensor construction: `vid = cat([noisy x_t, cond], -1)` `[L_vid, 33]`;
  `vid_shape = [b,3]`; `txt = text encoder embeds` `[L_txt, 5120]`;
  `txt_shape = [b,1]` (per-sample text length `l`);
  `timestep = [b]`.
- Block output consumed by: the next block (and, for the last layer,
  `vid_out_norm/ada` + `NaPatchOut`); `txt` continues to the next block and is
  bypassed in the last layer's MLP/ada (`vid_only=True`).

## forward signature (AST-verified)

`forward(self, vid, txt, vid_shape, txt_shape, emb, cache)` returning
`(vid_out, txt_out, vid_shape, txt_shape)`.
`NaDiT.forward(self, vid, txt, vid_shape, txt_shape, timestep, disable_cache=False)`.

## Inputs / outputs (field-level)

| name | role | dtype/rank | logical shape | static | dynamic | producer | consumer |
|---|---|---:|---|---|---|---|---|
| vid | video hidden states | f32 / 2 | [L_vid, 2560] | 2560 | L_vid | NaPatchIn (patch 1,2,2, T-major W-fastest) | attn_norm, mlp_norm, proj_qkv |
| txt | text hidden states | f32 / 2 | [L_txt, 2560] | 2560 | L_txt=l | text encoder -> txt_in proj | attn_norm, mlp_norm, proj_qkv |
| vid_shape | video shape metadata | i64 / 2 | [b, 3] (T,H,W post-patch) | 3 | b,T,H,W | na.flatten + NaPatchIn | window partition, vid_len, RoPE |
| txt_shape | text length metadata | i64 / 2 | [b, 1] | 1 | b, l | text encoder flatten | txt_len, RoPE get_freqs |
| emb | conditioning embedding | f32 / 2 | [b, 15360] | 15360 | b | TimeEmbedding(timestep) | AdaSingle |
| cache | memoized inference cache | object / 0 | Cache | - | - | NaDiT.forward | all components |
| attention mask | NO explicit tensor | - | implicit | - | - | cu_seqlens via concat_idx | FlashAttentionVarlen |
| text/video validity | NO explicit tensor | - | implicit | - | - | dynamic window partition | attention segments |

Outputs: `vid_out [L_vid, 2560]`, `txt_out [L_txt, 2560]`, shapes passed through.

## Token layout

- Video flatten is **T-major, W-fastest** (patch rearrange
  `(T t) (H h) (W w) c -> T H W (t h w c)`), matching the Phase 10.1 oracle.
- Text tokens are a 1-D sequence `[L_txt, C]`; not window-partitioned.
- Per window the attention segment is **`[vid_win tokens; txt tokens]`** (vid
  first, then txt); text q/k are **repeated per window** and the text attention
  outputs are **coalesced by mean** over the repeats.
- Batch is implicit: token dim is dim 0 of the flattened `[L, C]` tensors;
  `vid_shape [b,3]` / `txt_shape [b,1]` carry per-sample structure.

## RoPE offset semantics (`vid_freqs[l : l+f]`)

- `NaMMRotaryEmbedding3d(dim=128)` -> per-axis `RotaryEmbedding(dim=128//3=42,
  freqs_for="lang", theta=10000)`; effective rotary dim 126, identity tail 2
  (Phase 10.1 confirmed, unchanged).
- `get_freqs(vid_shape, txt_shape)` per batch item:
  - `vid_freq = vid_freqs[l : l+f, :h, :w]` from the precomputed
    `get_axial_freqs(1024, 128, 128)` grid — **`l` = text token count
    (txt_shape[i,0]); `f` = post-patch video frame count (vid_shape[i,0], the
    T dim, NOT total video tokens)**; the video T-axis positions are offset by
    the text length in the shared 1024-frame table.
  - `txt_freq = txt_freqs[:l].repeat(1, 3)` from `get_axial_freqs(1024)` — the
    first `l` 1-D freqs replicated across the 3 axis blocks.
- RoPE is applied to q/k per head; first 126 dims rotated, tail 2 identity.
- No block-index dependence; frequencies depend only on shapes and text length.
- Window-local coordinates: each window uses its own `(f_w,h_w,w_w)` with the
  same `l` offset; batch items do not share frequencies.

## Modulation (AdaSingle)

- `emb` `[b, 15360]` is rearranged `[b, d=2560, l=2, g=3]`; per layer (attn,
  mlp) 3 groups: shift / scale / gate, plus per-layer learned params
  (`shift ~N(0,1/sqrt(d))`, `scale ~ +1`, `gate`).
- mode "in": `hid * (scaleA + scaleB) + (shiftA + shiftB)`.
- mode "out": `hid * (gateA + gateB)` (linear gate, no activation).
- Sites: after attn_norm (in), after attention (out), after mlp_norm (in),
  after MLP (out); vid/txt modulated separately with per-token emb repetition.

## Attention (NaSwinAttention)

- `proj_qkv` Linear -> `3*heads*head_dim = 7680`; `proj_out` Linear 2560 ->
  2560; Q/K RMSNorm (affine) per head dim; qk_bias=False.
- Window partition (4,3,3) on vid qkv via `window_idx`; txt repeated per window.
- Window-local MM-RoPE3D on vid q/k (window_shape) and txt q/k.
- `FlashAttentionVarlen` (`flash_attn.flash_attn_varlen_func`) with
  `cu_seqlens_q == cu_seqlens_k` per window segment `[vid; txt]` -> varlen
  **block-diagonal** masking; no explicit mask, no causal mask, no
  cross-window attention.
- Attention math runs in **bfloat16** (inputs cast `.bfloat16()`); output cast
  back to fp32.
- After attention: `window_reverse`, head/seq scatter, `proj_out`.

## MLP / FFN (SwiGLUMLP)

- `proj_in_gate` / `proj_in` Linear(2560 -> 6912, bias=False);
  `proj_out` Linear(6912 -> 2560, bias=False); hidden = round-up of
  `2*2560*4/3 = 6826.7` to multiple of 256 = 6912.
- Activation: `silu(gate(x)) * x` (gated); no dropout, no bias, no conv.

## Residual order

1. `attn_norm(vid,txt)` -> ada(attn, in) -> attention -> ada(attn, out) ->
   `+vid/txt` (residual 1).
2. `mlp_norm(vid_attn,txt_attn)` -> ada(mlp, in) -> MLP -> ada(mlp, out) ->
   `+vid_attn/txt_attn` (residual 2).
3. Pre-norm style; gate is applied AFTER attention/MLP and BEFORE the residual
   add.

## Executable probes (real upstream code)

- `window.py` executed directly (pure math): profile A `(1,4,4)` -> 1 window
  (both methods); profile B `(5,32,32)` -> 12 / 27 windows; profile C
  asymmetric `(7,45,64)` -> 36 / 64 windows; all cells covered, all runs
  deterministic; `get_window_op("bogus")` -> ValueError.
- Negative contract tests: 16/16 PASS (dims, mlp rounding, `l+f<=1024`, AST
  NotImplementedError paths, patch flatten order, window determinism).
- Config parse: 32 layers, 20 heads, head_dim 128, hidden 2560, MLP 4x,
  emb 15360, window (4,3,3), mmrope3d 128.

## Dependency gap (Phase 11 blocker)

Missing in `.conda-cuda` (torch only): `einops`, `diffusers`, `flash_attn`,
`apex`, `rotary_embedding_torch` (pins in SeedVR2 `environment.yml`:
`flash-attn==2.5.9.post1`, `apex==0.1`, `diffusers==0.33.1`, `einops==0.7.0`,
`rotary-embedding-torch==0.5.3`; `requirements.txt`: `diffusers==0.29.1`,
`einops==0.7.0`, `rotary-embedding-torch==0.5.3`). The real random-weight
forward probe (`probe_real_block.py`) therefore exits BLOCKED (exit 3) with
`probe_result.json`; per task rules nothing was installed. The source/AST/
window contract is complete; only the executable forward probe is missing.

## ncnn mapping risk (Phase 12+)

`flash_attn_varlen_func` (CUDA-only, cu_seqlens block-diagonal, bf16);
sequence-parallel index permutations (reducible to static index maps);
dynamic by-size window partition; apex fused RMS/LN (covered by existing
fixtures); diffusers sinusoidal embedding (must match
`flip_sin_to_cos=False, downscale_freq_shift=0`); per-window `[vid;txt]`
concat + txt repeat/mean coalesce; per-token modulation broadcast; SwiGLU
(two GEMMs); `vid_freqs[l:l+f]` T-offset; shared vs separate vid/txt weights
by layer index (layers 0..9 separate, 10..31 shared, last layer vid_only);
bf16-vs-fp32 parity tolerance.

## Regressions

Phase 10.1 clean build `build/phase10_1-real-mmrope-release` re-run:
`dynamic_rope3d_tests`, `real_mmrope_contract_validation`,
`synthetic_hd128_validation`, `dynamic_multi_shape_validation`,
`runtime_validation` - see final report for commands and exit codes.

## JSON report

`reports/seedvr2_single_dit_block_contract.json` and
`tests/seedvr2_dit_block_contract/seedvr2_block_contract.json`: Python-stdlib
parseable, no NaN/Infinity, numeric fields are numbers, source evidence
complete, `real_seedvr2_weights=false`, `ncnn_block_complete=false`,
`overall_status=PARTIAL` with the single blocker recorded.
