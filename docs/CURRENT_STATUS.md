# Current status

This branch is the Phase 9.5 adaptive-window attention prototype. It is not full SeedVR2. The desktop CPU FP32 errors below were measured on this tree. No tablet ARM run is recorded here.

Phases 7.5-B, 8, 8.1, 9.1, 9.2, 9.3, 9.4, and 9.5 are READY for their documented fixture scope. Recent real validation: Dynamic A `[1,4,8,8,32]` error `1.49012e-7`; B `[1,4,9,10,32]` `1.49012e-7`; C `[1,5,9,10,32]` `2.08616e-7`; all validity/mask parity exact, no NaN/Inf. Cache grows 5→10→15; A→B→C→A is deterministic. Stability: multi-shape×10, runtime×10, dispatcher×20 PASS.

No current blocker. TODO: synthetic head_dim=128, complete DiT, real weights, further shapes, Vulkan, VAE. Recommended next: Phase 10 synthetic head_dim=128.
