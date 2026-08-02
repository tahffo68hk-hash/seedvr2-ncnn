"""Phase 11.1 dependency-compat tests.

Verifies that the test-only CPU compatibility implementations
(compat/flash_attn_cpu_compat.py, compat/apex_norm_cpu_compat.py) are
mathematically equivalent to explicit reference implementations
(max_abs_error < 1e-6), that injection through sys.modules works, and that
importing the real SeedVR2 block WITHOUT the compat layer fails safely
(subprocess, no crash).
"""
from __future__ import annotations

import subprocess
import sys

import torch

from probe_lib import SEEDVR2_ROOT, ensure_compat


def ref_rmsnorm(x, weight, eps):
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    return y.to(x.dtype)


def ref_varlen(q, k, v, cu_q, cu_k, scale, causal=False):
    out = torch.empty_like(q)
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    for i in range(len(cu_q) - 1):
        qs = q_f[cu_q[i]:cu_q[i + 1]]
        ks = k_f[cu_k[i]:cu_k[i + 1]]
        vs = v_f[cu_k[i]:cu_k[i + 1]]
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            q_len, k_len = qs.shape[0], ks.shape[0]
            mask = (torch.arange(k_len)[None, :]
                    > torch.arange(q_len)[:, None])
            scores = scores.masked_fill(mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[cu_q[i]:cu_q[i + 1]] = torch.einsum("hqk,khd->qhd", probs, vs).to(q.dtype)
    return out


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def run_rmsnorm_tests():
    from compat.apex_norm_cpu_compat import FusedRMSNorm
    results = []
    cases = [
        dict(shape=(7, 5, 128), affine=True, eps=1e-5, seed=1),
        dict(shape=(3, 64), affine=True, eps=1e-6, seed=2),
        dict(shape=(2, 9, 2560), affine=False, eps=1e-5, seed=3),
        dict(shape=(1, 1, 128), affine=True, eps=1e-5, seed=4),
        dict(shape=(4, 8, 32), affine=False, eps=1e-4, seed=5),
    ]
    for c in cases:
        torch.manual_seed(c["seed"])
        x = torch.randn(*c["shape"]) * 3.0 + 1.0
        m = FusedRMSNorm(c["shape"][-1], eps=c["eps"], elementwise_affine=c["affine"])
        weight = m.weight if c["affine"] else None
        with torch.no_grad():
            got = m(x)
        exp = ref_rmsnorm(x, weight, c["eps"])
        err = max_abs(got, exp)
        results.append({"case": c, "max_abs_error": err, "pass": err < 1e-6,
                        "weight_present": m.weight is not None})
    # weight correctness: after setting a distinct weight, compare against ref
    torch.manual_seed(9)
    m = FusedRMSNorm(16, eps=1e-5, elementwise_affine=True)
    with torch.no_grad():
        m.weight.copy_(torch.randn(16))
    x = torch.randn(5, 16)
    err = max_abs(m(x), ref_rmsnorm(x, m.weight, 1e-5))
    results.append({"case": "affine_weight_random", "max_abs_error": err, "pass": err < 1e-6})
    return results


def run_varlen_tests():
    from compat.flash_attn_cpu_compat import flash_attn_varlen_func
    results = []
    torch.manual_seed(11)
    cases = [
        dict(name="single_segment", cu_q=[0, 8], cu_k=[0, 8], heads=2, dim=16, scale=None),
        dict(name="multi_segment", cu_q=[0, 3, 9, 17], cu_k=[0, 3, 9, 17], heads=3, dim=32, scale=None),
        dict(name="uneven_lengths", cu_q=[0, 2, 7, 12], cu_k=[0, 2, 7, 12], heads=4, dim=64, scale=0.25),
        dict(name="equal_cu", cu_q=[0, 5, 11], cu_k=[0, 5, 11], heads=2, dim=16, scale=None),
        dict(name="causal_true", cu_q=[0, 6, 10], cu_k=[0, 6, 10], heads=2, dim=16, scale=None),
        dict(name="bf16_dtype", cu_q=[0, 4, 9], cu_k=[0, 4, 9], heads=2, dim=16, scale=None, dtype=torch.bfloat16),
    ]
    for c in cases:
        n = c["cu_q"][-1]
        dtype = c.get("dtype", torch.float32)
        q = torch.randn(n, c["heads"], c["dim"], dtype=dtype)
        k = torch.randn(n, c["heads"], c["dim"], dtype=dtype)
        v = torch.randn(n, c["heads"], c["dim"], dtype=dtype)
        cu_q = torch.tensor(c["cu_q"], dtype=torch.int32)
        cu_k = torch.tensor(c["cu_k"], dtype=torch.int32)
        head_dim = c["dim"]
        scale = c["scale"] if c["scale"] is not None else head_dim ** -0.5
        got = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                     max_seqlen_q=n, max_seqlen_k=n,
                                     softmax_scale=scale,
                                     causal=(c["name"] == "causal_true"))
        exp = ref_varlen(q, k, v, c["cu_q"], c["cu_k"], scale, causal=(c["name"] == "causal_true"))
        err = max_abs(got, exp)
        results.append({"name": c["name"], "max_abs_error": err, "pass": err < 1e-6,
                        "dtype": str(dtype)})

    # segment isolation: first segment data must not leak into the second
    cu = [0, 4, 8]
    q = torch.randn(8, 1, 8)
    k = q.clone(); v = q.clone()
    # zero out the second segment entirely
    q[4:] = 0.0; k[4:] = 0.0; v[4:] = 0.0
    got = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
                                 cu_seqlens_k=torch.tensor(cu, dtype=torch.int32),
                                 max_seqlen_q=8, max_seqlen_k=8)
    exp = ref_varlen(q, k, v, cu, cu, 8 ** -0.5)
    err = max_abs(got, exp)
    results.append({"name": "segment_isolation", "max_abs_error": err,
                    "pass": err < 1e-6 and float(got[4:].abs().max().item()) == 0.0})
    return results


def run_injection_tests():
    results = []
    info = ensure_compat()
    import importlib
    fa = importlib.import_module("flash_attn")
    results.append({"name": "flash_attn_importable", "pass": hasattr(fa, "flash_attn_varlen_func")})
    from apex.normalization import FusedRMSNorm  # noqa: F401
    results.append({"name": "apex_normalization_importable", "pass": True})
    for mod in ("einops", "diffusers", "rotary_embedding_torch", "torch"):
        spec = importlib.util.find_spec(mod)
        results.append({"name": f"real_module_{mod}", "pass": spec is not None})
    results.append({"name": "injection_info", "pass": info["injected"].get("flash_attn") == "compat"
                    and info["injected"].get("apex") == "compat"})
    return results, info


def run_missing_compat_subprocess():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {SEEDVR2_ROOT!r})\n"
        "import torch\n"
        "try:\n"
        "    import models.dit_v2.nablocks.mmsr_block\n"
        "    print('UNEXPECTED: imported without compat')\n"
        "    sys.exit(2)\n"
        "except ModuleNotFoundError as e:\n"
        "    print('EXPECTED:', e.name)\n"
        "    sys.exit(0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    out = (proc.stdout + proc.stderr).lower()
    ok = proc.returncode == 0 and "expected" in out and "flash_attn" in out
    return {"pass": ok, "returncode": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-3:]}


def run_all():
    results = []
    rms = run_rmsnorm_tests()
    results.append({"group": "rmsnorm_parity", "pass": all(r["pass"] for r in rms), "cases": rms})
    varlen = run_varlen_tests()
    results.append({"group": "varlen_attention_parity", "pass": all(r["pass"] for r in varlen), "cases": varlen})
    inj, info = run_injection_tests()
    results.append({"group": "injection", "pass": all(r["pass"] for r in inj), "cases": inj})
    sub = run_missing_compat_subprocess()
    results.append({"group": "missing_compat_safe_fail", "pass": sub["pass"], "detail": sub})
    return {"results": results, "all_pass": all(r["pass"] for r in results), "injection_info": info}


if __name__ == "__main__":
    report = run_all()
    print("test_dependency_compat all_pass =", report["all_pass"])
    for r in report["results"]:
        print(" ", r["group"], "->", r["pass"])
    sys.exit(0 if report["all_pass"] else 1)