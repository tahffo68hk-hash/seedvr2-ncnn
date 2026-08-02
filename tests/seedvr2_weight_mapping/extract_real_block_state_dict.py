"""Phase 13: enumerate the REAL real-width block state_dict (inventory).

Instantiates the real `NaMMSRTransformerBlock` at C=2560/H=20/HD=128 with
deterministic random weights (seed 20260803, FP32 CPU eval/no_grad) and
records every parameter from the real object graph (`named_parameters` +
`named_modules`), not hand-written strings.  Writes the raw inventory to the
git-ignored local artifacts dir; the committed mapping contract is derived
from it by `generate_weight_mapping_contract.py`.
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


def module_path_of(param_name: str) -> str:
    parts = param_name.split(".")
    return ".".join(parts[:-1])


def semantic_role(mod_path: str, pname: str) -> str:
    if mod_path.startswith("attn.proj_qkv."):
        return "qkv_projection"
    if mod_path.startswith("attn.proj_out."):
        return "attention_output_projection"
    if mod_path.startswith("attn.norm_q."):
        return "q_rmsnorm"
    if mod_path.startswith("attn.norm_k."):
        return "k_rmsnorm"
    if mod_path.startswith("mlp."):
        leaf = mod_path.rsplit(".", 1)[-1]
        if leaf == "proj_in_gate":
            return "swiglu_gate_projection"
        if leaf == "proj_in":
            return "swiglu_up_projection"
        if leaf == "proj_out":
            return "swiglu_output_projection"
    if mod_path.startswith("ada."):
        return "ada_single_parameter"
    if pname == "attn.rope.rope.freqs":
        return "rope_freqs_base_buffer"
    return "unknown:" + mod_path + "." + pname


def branch_of(mod_path: str) -> str:
    # branches are the next-to-last path component for MMModule-owned params
    parts = mod_path.split(".")
    for i, part in enumerate(parts):
        if part in ("vid", "txt"):
            return part
    return "shared"


def main() -> int:
    probe_lib.ensure_compat()
    torch.Tensor.bfloat16 = lambda self: self  # FP32 attention mode (documented)
    block = probe_lib.build_block(dict(rw.REAL_CFG), seed=rw.SEED).eval()

    mod_map = {path: mod for path, mod in block.named_modules()}
    entries = []
    total_numel = 0

    def record(kind, pname, p):
        nonlocal total_numel
        mod_path = module_path_of(pname)
        mod = mod_map.get(mod_path)
        cls = type(mod).__name__ if mod is not None else "?"
        numel = p.numel()
        total_numel += numel
        entries.append({
            "state_dict_key": pname,
            "source_module": mod_path,
            "source_module_class": cls,
            "semantic_role": semantic_role(mod_path, pname),
            "branch": branch_of(mod_path),
            "shape": list(p.shape),
            "numel": int(numel),
            "dtype": str(p.dtype),
            "requires_grad": bool(p.requires_grad) if isinstance(p, torch.nn.Parameter) else False,
            "kind": kind,
        })

    for pname, p in block.named_parameters():
        kind = "weight" if pname.endswith(".weight") else "bias" if pname.endswith(".bias") else "parameter"
        record(kind, pname, p)

    for bname, b in block.named_buffers():
        record("buffer", bname, b)

    # authoritative key set = state_dict (non-persistent buffers such as
    # attn.rope.rope.dummy are not part of a checkpoint and are excluded)
    sd = dict(block.state_dict())
    entries = [e for e in entries if e["state_dict_key"] in sd]
    names = {e["state_dict_key"] for e in entries}
    total_numel = sum(e["numel"] for e in entries)  # exact (excludes non-persistent buffers)
    assert len(sd) == len(entries), (len(sd), len(entries))
    assert names == set(sd.keys()), (sorted(names - set(sd.keys())), sorted(set(sd.keys()) - names))

    inventory = {
        "phase": "phase13",
        "variant": rw.VARIANT,
        "seed": rw.SEED,
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "config": dict(rw.REAL_CFG),
        "parameter_count": len(entries),
        "numel": int(total_numel),
        "theoretical_fp32_bytes": int(total_numel * 4),
        "entries": entries,
    }
    rw.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    io.open(rw.INVENTORY, "w", encoding="utf-8").write(json.dumps(inventory, indent=2))
    print("inventory entries:", len(entries))
    print("numel:", total_numel)
    print("fp32 bytes:", total_numel * 4)
    print("saved:", rw.INVENTORY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
