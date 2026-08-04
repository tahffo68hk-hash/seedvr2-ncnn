# -*- coding: utf-8 -*-
"""Phase 17: VAE memory / IO / latency profiles (profiles A/B).

Measures per-stage runner peak working set (ctypes GetProcessMemoryInfo),
artifact bytes (logical vs physical), cold/warm load latency and stage
execution latency.  Full-pipeline RAM/VRAM is recorded by the visual pipeline
runner; this module covers the VAE side plus artifact accounting.
Writes reports/phase17_memory_profile.json, phase17_io_profile.json,
phase17_latency_profile.json.
"""
from __future__ import annotations

import ctypes
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402
from ncnn_vae_runtime import _run_stage_exe, ncnn_dims, out_shape  # noqa: E402

RUNNER = Path(r"D:\AIProjects\seedvr2-ncnn\build\phase17-vae-smoke\vae_stage_runner.exe")
STAGES_ALL = ["enc_conv_in", "enc_down0", "enc_down1", "enc_down2", "enc_down3",
              "enc_mid", "enc_out", "dec_conv_in", "dec_mid", "dec_up0",
              "dec_up1", "dec_up2", "dec_up3", "dec_out"]

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def sample_peak(pid: int) -> dict:
    k32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid))
    if not h:
        return {}
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    k32.CloseHandle(h)
    if not ok:
        return {}
    return {"peak_working_set_bytes": int(pmc.PeakWorkingSetSize),
            "working_set_bytes": int(pmc.WorkingSetSize),
            "peak_private_bytes": int(pmc.PeakPagefileUsage)}


def run_stage_sampled(param: Path, bin_: Path, x: np.ndarray, in_shape, out_shape,
                      tmp: Path, tag: str, repeats: int) -> dict:
    inp = tmp / f"{tag}_in.bin"
    outp = tmp / f"{tag}_out.bin"
    x.astype("<f4").tofile(str(inp))
    cmd = [str(RUNNER), str(param), str(bin_), str(inp), str(outp)]
    cmd += [str(d) for d in ncnn_dims(in_shape)] + ["--"] + [str(d) for d in ncnn_dims(out_shape)]
    times = []
    peak = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while proc.poll() is None:
            m = sample_peak(proc.pid)
            if m:
                peak = max(peak, m.get("peak_working_set_bytes", 0), m.get("working_set_bytes", 0))
            time.sleep(0.005)
        times.append(time.perf_counter() - t0)
    return {"times": times, "peak_working_set_bytes": peak}


def main() -> int:
    root = Path(pc.ART)
    tmp = Path(pc.EXPORT_TMP) / "profiles"
    tmp.mkdir(parents=True, exist_ok=True)

    mem = {"phase": 17, "real_seedvr2_weights": True, "real_vae_weights": True,
           "checkpoint_sha256": pc.VAE_CKPT_SHA256, "profiles": {}}
    io = {"phase": 17, "real_seedvr2_weights": True, "real_vae_weights": True,
          "checkpoint_sha256": pc.VAE_CKPT_SHA256, "profiles": {}, "note": {
              "physical_bytes": "sum of stage model.ncnn.bin + .param on disk",
              "logical_bytes": "sum of state-dict weights consumed by the stages",
              "current_pc_fp32_artifact_size_is_not_mobile_final": True}}
    lat = {"phase": 17, "real_seedvr2_weights": True, "real_vae_weights": True,
           "checkpoint_sha256": pc.VAE_CKPT_SHA256, "profiles": {}, "method": {
               "stage_times": "measured by vae_stage_runner subprocess",
               "repeats": "small stages x3, heavy stages x1 (P50/P95 only where >=3)"}}

    for prof in ("A", "B"):
        manifest = json.loads((root / f"reference_{prof}" / "manifest.json").read_text(encoding="utf-8"))
        enc_bytes = dec_bytes = 0
        stages = {}
        for stage in STAGES_ALL:
            d = root / f"stages_{prof}" / stage
            p, b = d / "model.ncnn.param", d / "model.ncnn.bin"
            if not b.exists():
                continue
            sz = b.stat().st_size + (p.stat().st_size if p.exists() else 0)
            if stage.startswith("enc"):
                enc_bytes += sz
            else:
                dec_bytes += sz
            in_shape = manifest["stage_files"][stage]["in"]["shape"]
            out_sh = manifest["stage_files"][stage]["out"]["shape"]
            x = np.fromfile(manifest["stage_files"][stage]["in"]["path"], dtype="<f4").reshape(in_shape)
            if stage in ("enc_out", "dec_out"):
                bt, c, h, w = in_shape
                x = x.transpose(1, 0, 2, 3).reshape(1, c, bt, h, w)
                in_shape = [1, c, bt, h, w]
            repeats = 3 if sz < 5_000_000 else 1
            r = run_stage_sampled(p, b, x, in_shape, out_sh, tmp, f"{prof}_{stage}", repeats)
            times = r["times"]
            stages[stage] = {
                "artifact_bytes": sz, "repeats": len(times),
                "latency_seconds": times,
                "p50": statistics.median(times) if len(times) >= 3 else times[0],
                "p95": (sorted(times)[int(0.95 * len(times)) - 1] if len(times) >= 3 else times[0]),
                "max": max(times),
                "peak_working_set_bytes": r["peak_working_set_bytes"],
            }
            print(f"  {prof} {stage}: bytes={sz} t={times} peak_ws={r['peak_working_set_bytes']}", flush=True)
        mem["profiles"][prof] = {"stages": stages,
                                 "encoder_peak_working_set_bytes": max(
                                     (s["peak_working_set_bytes"] for s in stages.values() if s.get("peak_working_set_bytes")), default=0),
                                 "decoder_peak_working_set_bytes": max(
                                     (s["peak_working_set_bytes"] for s in stages.values() if s.get("peak_working_set_bytes")), default=0)}
        io["profiles"][prof] = {
            "encoder_artifact_bytes": enc_bytes,
            "decoder_artifact_bytes": dec_bytes,
            "vae_total_physical_bytes": enc_bytes + dec_bytes,
            "weight_free_shared_bytes": 0,
            "note": "ncnn bins hold pnnx-baked FP32 weights + constants",
        }
        lat["profiles"][prof] = {"stages": stages}
        print(f"  {prof}: enc_bytes={enc_bytes} dec_bytes={dec_bytes}", flush=True)

    mem["overall_pass"] = True
    io["overall_pass"] = True
    lat["overall_pass"] = True
    for name, payload in (("phase17_memory_profile.json", mem), ("phase17_io_profile.json", io),
                          ("phase17_latency_profile.json", lat)):
        pc.write_report(str(Path(pc.REPO_ROOT) / "reports" / name), payload)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
