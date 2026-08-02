# Next task prompt

Implement **Phase 10: synthetic head_dim=128 / rope_dim=128 stress validation** only after reading `CODEX_HANDOFF.md` and its linked files. Build a shape-only synthetic fixture, not real SeedVR2 weights: heads=4, head_dim=128, channels=512, rope_dim=128, window=[4,4,4], `pnnx fp16=0`, CPU FP32 parity, dynamic mask/RoPE, single- and multi-window validation, memory/latency reporting, and baseline/fixed/dynamic regressions. Do not download weights or modify third_party sources. Alternative only: a synthetic complete DiT block fixture.
