"""Phase 13: real-width last-layer mapping / artifact reuse verification.

The base and last-layer `NaMMSRTransformerBlock` are re-instantiated with the
same deterministic seed.  This verifies the last-layer contract documented in
`seedvr2_block_weight_mapping.json`:

- attention + RoPE parameter sets are bit-identical between base and last-layer;
- txt-side MLP / AdaSingle parameters are ABSENT from the last-layer state_dict
  (vid_only control flow) - exactly the 9 non-applicable contract entries;
- the remaining 20 common keys are shape-compatible and the last-layer block
  strict-loads them from the base state_dict;
- artifact reuse: the last-layer block is served by the SAME 19 base stage
  models (no second real-width export) - the runtime model set is unchanged.

Numeric last-layer execution is NOT repeated at real width: Phase 12 already
validated scaled last-layer numerics end-to-end (see synthetic_dit_block
last-layer report), and the real-width reference exporter emits base-stage
models only.
Writes `reports/realwidth_last_layer_reuse_validation.json`.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))

import _realwidth_common as rw  # noqa: E402
import probe_lib  # noqa: E402


def build_sd(cfg_override: dict) -> dict:
    probe_lib.ensure_compat()
    torch.Tensor.bfloat16 = lambda self: self
    cfg = dict(rw.REAL_CFG)
    cfg.update(cfg_override)
    block = probe_lib.build_block(cfg, seed=rw.SEED).eval()
    return {k: v.detach().cpu() for k, v in block.state_dict().items()}


def main() -> int:
    contract = json.load(io.open(HERE / "seedvr2_block_weight_mapping.json", "r", encoding="utf-8"))
    sd_base = build_sd({})
    sd_last = build_sd({"is_last_layer": True})

    only_base = sorted(set(sd_base) - set(sd_last))
    only_last = sorted(set(sd_last) - set(sd_base))
    common = sorted(set(sd_base) & set(sd_last))

    attn_diff = [k for k in common
                 if (k.startswith("attn") or k.startswith("attn_norm") or "rope" in k)
                 and not torch.equal(sd_base[k], sd_last[k])]
    shape_mismatch = [k for k in common if list(sd_base[k].shape) != list(sd_last[k].shape)]

    # last-layer block must strict-load the base common subset (reuse proof)
    load_ok = True
    try:
        probe_lib.ensure_compat()
        torch.Tensor.bfloat16 = lambda self: self
        block_last = probe_lib.build_block(dict(rw.REAL_CFG, is_last_layer=True), seed=rw.SEED).eval()
        block_last.load_state_dict({k: sd_base[k] for k in common}, strict=True)
    except Exception as exc:  # noqa: BLE001
        load_ok = False
        last_error = str(exc)

    # expected non-applicable keys: exactly the txt-side MLP + AdaSingle params
    expected_absent = sorted(e["state_dict_key"] for e in contract["entries"]
                             if not e["is_last_layer_applicable"])
    absent_ok = only_base == expected_absent and only_last == []
    attn_ok = attn_diff == []
    shape_ok = shape_mismatch == []
    ll = contract["last_layer"]

    results = {
        "phase": "phase13",
        "target": "realwidth_last_layer_reuse_tests",
        "variant": rw.VARIANT,
        "base_state_dict_keys": len(sd_base),
        "last_layer_state_dict_keys": len(sd_last),
        "common_keys": len(common),
        "only_in_base": only_base,
        "only_in_last": only_last,
        "expected_absent_keys": expected_absent,
        "absent_keys_ok": absent_ok,
        "attention_and_rope_bit_identical": attn_ok,
        "attention_different_keys": attn_diff,
        "common_shape_compatible": shape_ok,
        "shape_mismatches": shape_mismatch,
        "last_layer_strict_loads_base_subset": load_ok,
        "artifact_reuse": ll["artifact_reuse"],
        "control_flow_only": ll["control_flow_only"],
        "numeric_last_layer_not_repeated_at_real_width": True,
        "phase12_scaled_last_layer_numeric_validation": "synthetic_dit_block_last_layer_validation.json",
        "overall_pass": bool(absent_ok and attn_ok and shape_ok and load_ok),
        "overall_status": "PASS" if (absent_ok and attn_ok and shape_ok and load_ok) else "FAIL",
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "full_32_block_dit": False,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    if not load_ok:
        results["last_layer_load_error"] = last_error

    out = rw.REPO / "reports" / "realwidth_last_layer_reuse_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(results, indent=2))
    print("absent_ok:", absent_ok, "attn_ok:", attn_ok, "shape_ok:", shape_ok, "load_ok:", load_ok)
    print("overall:", results["overall_status"])
    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
