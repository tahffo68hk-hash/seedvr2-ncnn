# Phase 18C.1 decoder resume

Block-32 checkpoint inspection is available without deleting the existing gate files. The UI presents Decoder Only as a guarded mode and reports checkpoint presence/byte sizes. The native decoder-only tensor bridge is intentionally not claimed complete yet: a Block-32 internal DiT tensor must pass the Phase 17 post-head/scheduler endpoint before it can be treated as a VAE latent. Until that bridge is implemented and verified on Pad, Decoder Only remains PARTIAL and the full demo remains BLOCKED.
