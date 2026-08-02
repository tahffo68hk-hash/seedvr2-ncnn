# Next task prompt

Resolve the real SeedVR2 DiT block contract, only after reading `CODEX_HANDOFF.md` and its linked files, and only if `rotary-embedding-torch` (or equivalent) is available locally. Phase 10 established a synthetic `head_dim=128 / rope_dim=128` contract (state PARTIAL); do not re-run it.

Task: read `D:\AIProjects\third_party\SeedVR2` read-only and extract the interface/shape contract of a single real DiT block: QKV projection shapes, head count and head_dim from config, the true MM-RoPE3D T/H/W axis pair split and frequency rule from `NaMMRotaryEmbedding3d`/`rotary-embedding-torch`, padding-validity semantics, and the output projection/merge contract. Report the concrete contract with file/symbol references. Do not download weights and do not start implementation. If the true RoPE split still cannot be determined, keep Phase 10 PARTIAL and do not claim real-SeedVR2 RoPE parity.
