"""Phase 12: validate the generated staged SeedVR2 DiT block reference artifacts.

Checks, per profile A/B/C/D and for the base / last-layer variants:

  * manifest.json parses and every required stage is present/finite
  * every stage .bin exists and its byte size matches numel*4 (float32)
  * window metadata is self-consistent (window sizes sum to vid tokens,
    segment lengths == window size + per-window txt length, cu_seqlens
    monotonic and ending at total_packed, max_segment matches)
  * derived cross-checks: q/k/v concat == qkv, packed rows == total_packed,
    vid merged == window_reverse(vid windows merged), txt mean merge ==
    mean of repeated txt windows
  * SHA256 of every artifact listed in sha256_manifest.json matches

Exits 0 on success; prints one error per problem and exits 1 otherwise.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VARIANT = "synthetic_seedvr2_dit_block_c512_h4_hd128"
VARIANT_LL = VARIANT + "_last_layer"
PROFILES = ["A", "B", "C", "D"]

STAGE_ORDER = [
    "input_vid", "input_txt", "vid_shape", "txt_shape", "emb",
    "attn_norm_vid", "attn_norm_txt", "ada_attn_in_vid", "ada_attn_in_txt",
    "qkv_vid", "qkv_txt", "q_vid", "k_vid", "v_vid", "q_txt", "k_txt", "v_txt",
    "q_norm_vid", "k_norm_vid", "q_norm_txt", "k_norm_txt",
    "rope_before_q_vid", "rope_before_k_vid", "rope_after_q_vid", "rope_after_k_vid",
    "rope_after_q_txt", "rope_after_k_txt", "rope_freqs_vid", "rope_freqs_txt",
    "attention_vid_merged", "attention_txt_mean_merged",
    "packed_q", "packed_k", "packed_v",
    "attention_packed_output", "attention_vid_windows", "attention_txt_windows",
    "attention_proj_vid", "attention_proj_txt",
    "ada_attn_gate_vid", "ada_attn_gate_txt",
    "attention_residual_vid", "attention_residual_txt",
    "mlp_norm_vid", "mlp_norm_txt", "ada_mlp_in_vid", "ada_mlp_in_txt",
    "swiglu_gate_projection_vid", "swiglu_gate_projection_txt",
    "swiglu_up_projection_vid", "swiglu_up_projection_txt",
    "swiglu_product_vid", "swiglu_product_txt",
    "swiglu_output_vid", "swiglu_output_txt",
    "ada_mlp_gate_vid", "ada_mlp_gate_txt",
    "final_vid", "final_txt",
]

# last-layer block drops the txt MLP stages and the txt AdaSingle gates
# (the vid_only MLP/ada modules pass txt through untouched, so no txt
# modulation/gate stages exist for the last layer).
LL_EXCLUDE = {
    "mlp_norm_txt", "ada_mlp_in_txt", "ada_mlp_gate_txt",
    "swiglu_gate_projection_txt", "swiglu_up_projection_txt",
    "swiglu_product_txt", "swiglu_output_txt",
    "ada_attn_in_txt", "ada_attn_gate_txt",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bin(path: Path):
    return np.fromfile(path, dtype=np.float32)


def validate_profile(root: Path, profile: str, last_layer: bool, errors: list):
    ref = root / "reference" / profile
    manifest_path = ref / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"{profile}: manifest.json missing")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest.get("stages", {})
    windows = manifest.get("windows", {})
    inputs = manifest.get("inputs", {})
    if manifest.get("profile") != profile:
        errors.append(f"{profile}: manifest profile field mismatch")

    expected = [s for s in STAGE_ORDER if not (last_layer and s in LL_EXCLUDE)]
    missing = [s for s in expected if s not in stages]
    if missing:
        errors.append(f"{profile}: missing stages {missing}")
    for name, st in stages.items():
        if not st.get("finite", False):
            errors.append(f"{profile}: stage {name} not finite")
        numel = st.get("numel")
        fname = st.get("file")
        if not fname:
            errors.append(f"{profile}: stage {name} missing file")
            continue
        p = ref / fname
        if not p.exists():
            errors.append(f"{profile}: stage {name} bin missing {fname}")
            continue
        expected_bytes = numel * 4 if numel is not None else None
        if expected_bytes is not None and p.stat().st_size != expected_bytes:
            errors.append(f"{profile}: stage {name} size {p.stat().st_size} != numel*4 {expected_bytes}")
        v = load_bin(p)
        if not np.isfinite(v).all():
            errors.append(f"{profile}: stage {name} bin non-finite")

    # window / packing consistency
    batch = inputs.get("batch")
    vid_shape = inputs.get("vid_shape", [])
    txt_shape = inputs.get("txt_shape", [])
    vid_tokens = inputs.get("vid_tokens")
    txt_tokens = inputs.get("txt_tokens")
    if vid_tokens is None or txt_tokens is None or batch is None:
        errors.append(f"{profile}: inputs incomplete")
        return
    expected_vid = sum(int(np.prod(r)) for r in vid_shape)
    if vid_tokens != expected_vid:
        errors.append(f"{profile}: vid_tokens {vid_tokens} != sum(prod(vid_shape)) {expected_vid}")
    expected_txt = sum(int(r[0]) for r in txt_shape)
    if txt_tokens != expected_txt:
        errors.append(f"{profile}: txt_tokens {txt_tokens} != sum(txt_shape) {expected_txt}")
    if batch != len(vid_shape) or batch != len(txt_shape):
        errors.append(f"{profile}: batch mismatch {batch} vs {len(vid_shape)}/{len(txt_shape)}")

    wshape = windows.get("window_shape", [])
    wcount = windows.get("window_count", [])
    wsizes = windows.get("window_sizes", [])
    segs = windows.get("segment_lengths", [])
    cu_q = windows.get("cu_seqlens_q", [])
    cu_k = windows.get("cu_seqlens_k", [])
    total_windows = sum(int(c) for c in wcount)
    if total_windows != len(wshape) or total_windows != len(wsizes) or total_windows != len(segs):
        errors.append(f"{profile}: window shape/count/sizes/segments length mismatch (windows={total_windows}, shapes={len(wshape)}, sizes={len(wsizes)}, segs={len(segs)})")
        return
    if len(wshape) != len(segs):
        errors.append(f"{profile}: segment_lengths length mismatch")
    if sum(wsizes) != vid_tokens:
        errors.append(f"{profile}: sum(window_sizes) {sum(wsizes)} != vid_tokens {vid_tokens}")
    # window sizes == prod of each window shape
    for i, (sh, sz) in enumerate(zip(wshape, wsizes)):
        if int(np.prod(sh)) != sz:
            errors.append(f"{profile}: window {i} shape prod {int(np.prod(sh))} != size {sz}")
    # segment lengths == window size + txt length (txt repeated per window count)
    expect_segs = []
    win_idx = 0
    for s, cnt in enumerate(wcount):
        tl = int(txt_shape[s][0])
        for _ in range(int(cnt)):
            expect_segs.append(wsizes[win_idx] + tl)
            win_idx += 1
    if expect_segs != segs:
        errors.append(f"{profile}: segment_lengths mismatch {segs} vs expected {expect_segs}")
    if cu_q != [0] + list(np.cumsum(segs)) and cu_q != [0] + [int(x) for x in np.cumsum(segs)]:
        # cu may be int list; compare robustly
        expect_cu = [0] + [int(x) for x in np.cumsum(segs)]
        if [int(x) for x in cu_q] != expect_cu:
            errors.append(f"{profile}: cu_seqlens_q not cumsum(segments) {cu_q} vs {expect_cu}")
    if cu_k != cu_q:
        errors.append(f"{profile}: cu_seqlens_k != cu_seqlens_q")
    if not all(b >= a for a, b in zip(cu_q, cu_q[1:])):
        errors.append(f"{profile}: cu_seqlens_q not monotonic")
    if windows.get("max_segment") != (max(segs) if segs else None):
        errors.append(f"{profile}: max_segment mismatch")
    if windows.get("total_packed") != (cu_q[-1] if cu_q else None):
        errors.append(f"{profile}: total_packed mismatch")
    if "packed_q" in stages and stages["packed_q"]["logical_shape"][0] != windows.get("total_packed"):
        errors.append(f"{profile}: packed_q rows != total_packed")

    # derived cross checks
    def stage(name):
        if name not in stages:
            return None
        shape = [int(x) for x in stages[name].get("logical_shape", [])]
        v = load_bin(ref / stages[name]["file"])
        return v.reshape(shape) if shape else v

    qkv = stage("qkv_vid")
    if qkv is not None and "q_vid" in stages and "k_vid" in stages and "v_vid" in stages:
        cat = np.concatenate([stage("q_vid"), stage("k_vid"), stage("v_vid")], axis=1)
        if not np.array_equal(cat, qkv):
            errors.append(f"{profile}: q/k/v concat != qkv_vid")
    if "packed_q" in stages and "packed_v" in stages:
        if stages["packed_q"]["logical_shape"] != stages["packed_v"]["logical_shape"]:
            errors.append(f"{profile}: packed q/v shape mismatch")
    # vid merged == window_reverse(vid windows merged)
    if "attention_vid_windows" in stages and "attention_vid_merged" in stages:
        vid_win = stage("attention_vid_windows")  # [sum(win sizes), 4, 128]
        vid_merged = stage("attention_vid_merged")  # [vid_tokens, 512]
        if vid_win is not None and vid_merged is not None:
            win_order = vid_win.reshape(-1, 512)
            reverse = windows.get("reverse_index")
            if reverse and len(reverse) == vid_merged.shape[0]:
                merged_from_win = win_order[np.argsort(reverse)]
                if not np.array_equal(merged_from_win, vid_merged):
                    errors.append(f"{profile}: vid merged != window_reverse(vid windows)")
    # txt mean merge
    if "attention_txt_windows" in stages and "attention_txt_mean_merged" in stages:
        txt_win = stage("attention_txt_windows")
        txt_merged = stage("attention_txt_mean_merged")
        if txt_win is not None and txt_merged is not None:
            off = 0
            parts = []
            txt_lens = [int(r[0]) for r in txt_shape]
            counts = [int(c) for c in wcount]
            for l, n in zip(txt_lens, counts):
                block = txt_win[off:off + l * n].reshape(l, n, 4, 128).mean(1)
                parts.append(block)
                off += l * n
            if not np.allclose(np.concatenate(parts), txt_merged, atol=0, rtol=0):
                errors.append(f"{profile}: txt mean merge mismatch")


def validate_sha256(root: Path, errors: list):
    sm = root / "sha256_manifest.json"
    if not sm.exists():
        errors.append(f"{root.name}: sha256_manifest.json missing")
        return
    manifest = json.loads(sm.read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "sha256":
        errors.append(f"{root.name}: sha256 algorithm mismatch")
    for rel, expected in manifest["files"].items():
        p = root / rel
        if not p.exists():
            errors.append(f"{root.name}: sha256 file missing {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            errors.append(f"{root.name}: sha256 mismatch {rel}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "artifacts" / VARIANT)
    parser.add_argument("--last-layer-root", type=Path, default=HERE / "artifacts" / VARIANT_LL)
    parser.add_argument("--profiles", default="A,B,C,D")
    parser.add_argument("--last-layer-profiles", default="A,B")
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    errors = []
    profiles = [p for p in args.profiles.split(",") if p]
    ll_profiles = [p for p in args.last_layer_profiles.split(",") if p]
    for profile in profiles:
        validate_profile(args.root, profile, last_layer=False, errors=errors)
    for profile in ll_profiles:
        validate_profile(args.last_layer_root, profile, last_layer=True, errors=errors)
    if not args.skip_sha256:
        validate_sha256(args.root, errors)
        validate_sha256(args.last_layer_root, errors)
    if errors:
        for e in errors:
            print("ERROR:", e)
        return 1
    print(f"reference manifests OK: base {len(profiles)} profiles, last-layer {len(ll_profiles)} profiles, sha256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
