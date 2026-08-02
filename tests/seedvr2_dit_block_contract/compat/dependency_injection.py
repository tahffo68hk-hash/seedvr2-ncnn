"""Dependency injection for the real SeedVR2 import chain under CPU.

Phase 11.1 isolated probe environment (conda env ``seedvr2-block-probe``)
provides the REAL non-model Python packages pinned by SeedVR2
(environment.yml / requirements.txt): torch, einops==0.7.0,
diffusers==0.33.1 (requirements.txt: diffusers==0.29.1),
rotary-embedding-torch==0.5.3.

The two fused-CUDA-kernel libraries that cannot load in a Windows CPU-only
box are replaced in the TEST HARNESS ONLY with math-equivalent CPU
implementations:

  * flash_attn  -> compat.flash_attn_cpu_compat (varlen block-diagonal attention)
  * apex        -> compat.apex_norm_cpu_compat (RMSNorm / LayerNorm)

Injection is performed through ``sys.modules`` BEFORE any SeedVR2 module is
imported, so no SeedVR2 source file is modified and the real block class and
real forward source execute unmodified.
"""
from __future__ import annotations

import importlib.util
import sys
import types

SEEDVR2_ROOT = r"D:\AIProjects\third_party\SeedVR2"

_REAL_MODULES = ["einops", "diffusers", "rotary_embedding_torch", "torch"]


def _real_import_ok(name: str) -> bool:
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except Exception:
        return False


def install_compat() -> dict:
    """Install CPU-compat modules into sys.modules.

    If a REAL flash_attn / apex import succeeds, the real module is kept;
    otherwise the math-equivalent CPU compat module is injected.  Returns a
    machine-readable dict describing what was injected.
    """
    result = {"seedvr2_root": SEEDVR2_ROOT, "real_modules": {}, "injected": {}}

    for name in _REAL_MODULES:
        result["real_modules"][name] = _real_import_ok(name)

    from . import apex_norm_cpu_compat
    from . import flash_attn_cpu_compat

    if _real_import_ok("flash_attn"):
        result["injected"]["flash_attn"] = "real"
    else:
        fa_mod = types.ModuleType("flash_attn")
        fa_mod.flash_attn_varlen_func = flash_attn_cpu_compat.flash_attn_varlen_func
        fa_mod.__doc__ = flash_attn_cpu_compat.__doc__
        sys.modules["flash_attn"] = fa_mod
        result["injected"]["flash_attn"] = "compat"
        result["injected"]["flash_attn.compat_version"] = flash_attn_cpu_compat.CPU_COMPAT_VERSION

    if _real_import_ok("apex"):
        result["injected"]["apex"] = "real"
    else:
        apex_pkg = types.ModuleType("apex")
        apex_norm = types.ModuleType("apex.normalization")
        apex_norm.FusedRMSNorm = apex_norm_cpu_compat.FusedRMSNorm
        apex_norm.FusedLayerNorm = apex_norm_cpu_compat.FusedLayerNorm
        apex_norm.__doc__ = apex_norm_cpu_compat.__doc__
        apex_pkg.normalization = apex_norm
        sys.modules["apex"] = apex_pkg
        sys.modules["apex.normalization"] = apex_norm
        result["injected"]["apex"] = "compat"
        result["injected"]["apex.compat_version"] = apex_norm_cpu_compat.CPU_COMPAT_VERSION

    return result


def install_probe_paths() -> None:
    """Prepend the real SeedVR2 root so ``models``/``common`` resolve to it."""
    if SEEDVR2_ROOT not in sys.path:
        sys.path.insert(0, SEEDVR2_ROOT)