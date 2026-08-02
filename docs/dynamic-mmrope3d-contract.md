# Dynamic MMRoPE3D Contract (Phase 9.3)

Status: **READY for the fixed RMS host-compatibility RoPE contract only.** Runtime routing and artifact metadata modes remain TODO.

## Source trace and coordinate scope

The real SeedVR2 path is `NaSwinAttention.forward` in
`D:/AIProjects/third_party/SeedVR2/models/dit_v2/nablocks/attention/mmattn.py`,
which applies the window transform before calling `NaMMRotaryEmbedding3d` in
`models/dit_v2/rope.py`. `NaMMRotaryEmbedding3d.get_freqs` gets axial frequencies
and slices each window shape from origin (`[:f, :h, :w]`). Therefore this path
uses **window-local** T/H/W coordinates, not global video offsets.

The production 3B config (`configs_3b/main.yaml`) selects `rope_type: mmrope3d`
and `rope_dim: 128`; it uses theta 10000 in `MMRotaryEmbeddingBase`. The current
small RMS fixture is a separately explicit host compatibility contract from
`tests/seedvr2_attention_fidelity/rope3d_reference.py`, not a claim that the
8-channel fixture equals the real 128-channel SeedVR2 deployment contract.

## Fixture-compatible layout and math

`Rope3DConfig` explicitly carries `theta`, real-channel `axis_dims`, and
`axis_scales`. For the RMS fixture these are theta 10000,
`axis_dims=[2,2,2]`, and `axis_scales=[1, 1/100, 1/10000]`.

`rope_dim=6` means three interleaved complex pairs: pair 0=T, pair 1=H, pair
2=W. The fourth pair of `head_dim=8` is unrotated (`cos=1`, `sin=0`). For pair
`p`, `inv_freq=exp(-log(theta)*p/(rope_dim/2))`; angle is local coordinate times
axis scale times inv_freq. The implementation deliberately uses FP32 frequency,
angle, `cos`, and `sin`, matching the PyTorch fixture path.

Tokens are W-fastest: `token=((lt*Wh)+lh)*Ww+lw`. Flat output is
`flat_index=token*pairs+pair`, logical `[tokens,pairs]`; the later ncnn helper
must construct independent `ncnn::Mat(pairs,tokens)` values without changing this
order. Batch/window/head are not baked into these local cos/sin values and are
consumer-broadcast dimensions.

Padding does not alter RoPE: all padded local positions receive ordinary RoPE
values. Validity/mask remains a separate contract that excludes invalid keys.

## Verification

`dynamic_rope3d_tests` checks origin, isolated W/H/T motion, W-fastest order,
`[3,4,5]` (60 tokens), `[1,2,3]`, output clearing after an invalid axis contract,
and Golden Parity against the fixed RMS artifact.

For `[4,4,4]`, `head_dim=8`, `rope_dim=6`, Golden Parity is:

* cos max absolute error: `5.96046e-08`
* sin max absolute error: `0`

No custom ncnn layer is required; this is host-side tensor generation.

## Phase 9 status

* DynamicAttentionContract: READY
* Dynamic padding validity/mask: READY
* Dynamic 3D RoPE: READY (fixed RMS host-compatible contract)
* Metadata auxiliary modes: TODO
* Runtime dynamic routing: TODO
* Multi-shape ncnn parity: TODO
# Phase 9.4 validation addendum

Dynamic A RoPE has logical shape `[64,4]` and is passed as
`ncnn::Mat(4,64)`. The contract test verifies dimensions, count 256, flat
token/pair order, independent storage, and rejection of count/shape mismatch.
