# Auxiliary Input Metadata Contract

Staged hybrid artifacts declare `auxiliary_inputs` explicitly. `MaskMode` is
either `artifact_reference` (safe relative binary path) or `padding_validity`
(finite negative `invalid_key_value`). `RopeMode` is either
`artifact_reference` (safe cos/sin paths) or `dynamic_mmrope3d` (even rope_dim,
positive theta, three even axis dimensions summing to rope_dim, and three
positive finite scales).

The parser accepts strict JSON integers, decimals, and scientific notation; it
rejects leading plus/zero, incomplete fractions/exponents, and non-finite values.
Exponent signs may be either `+` or `-`. Integer fields require integer-token
syntax (`4.0` is rejected). Auxiliary objects are strict: unknown fields and
cross-mode fields are rejected with their `auxiliary_inputs.*` field path.
Fixed LayerNorm and RMS artifacts explicitly use artifact references. The
Dynamic RMS profile uses padding validity and dynamic MMRoPE3D. The historical
fixed RMS `valid_tokens=60` mask is an artifact override, not natural padding.

## Phase 9.4 final reference validation

Reference paths must be non-empty, safe relative paths to regular files beneath
the artifact directory. Absolute drive/UNC paths and any `..` component are
rejected. The final matrix covers 11 mask-reference and 10 RoPE-reference
negative manifests, including missing/type/empty/cross-mode fields, traversal,
absolute or UNC paths, directories, and missing files. Failures include the
`metadata.json` path, the relevant `auxiliary_inputs.*` field where available,
and JSON character offset for malformed syntax. A failed scan is atomic: the
previous registry remains available for baseline, fixed LayerNorm/RMSNorm, and
Dynamic A lookups.
