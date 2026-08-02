# Phase 11: real SeedVR2 single DiT block contract

Source/AST contract extraction for the real SeedVR2 DiT block
(`NaMMSRTransformerBlock`, `models/dit_v2/nablocks/mmsr_block.py`, SeedVR2
commit `e4de8c2`). No weights, no downloads, no `third_party/SeedVR2`
modification.

## Files

- `extract_seedvr2_block_contract.py` - config parse + AST contract + executable
  window-partition probes + negative tests + dependency-gap + freshness.
  Writes `seedvr2_block_contract.json` and
  `reports/seedvr2_single_dit_block_contract.json`.
- `seedvr2_block_contract.json` - generated machine-readable contract (no
  NaN/Infinity; numeric fields are numbers).
- `probe_real_block.py` - real random-weight forward probe of the real block.
  Exits 3 (BLOCKED) with `probe_result.json` when non-model deps are missing.
- `README.md` - this file.

## Run

```powershell
D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe tests\seedvr2_dit_block_contract\extract_seedvr2_block_contract.py
```

## Dependency gap (blocker for the executable forward probe)

The real block imports `einops`, `diffusers`, `flash_attn`, `apex` and
`rotary_embedding_torch` (pins in SeedVR2 `environment.yml`:
`flash-attn==2.5.9.post1`, `apex==0.1`, `diffusers==0.33.1`, `einops==0.7.0`,
`rotary-embedding-torch==0.5.3`). None are installed in `.conda-cuda` (torch
only), and the task forbids installing them, so the random-weight forward
probe stays BLOCKED; the source/AST/window contract is complete.
`overall_status` is PARTIAL until the probe can run.
