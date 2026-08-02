"""Phase 10.1: exact upstream rotary-embedding-torch MM-RoPE3D oracle generator.

Loads the precise SeedVR2 dependency version (rotary-embedding-torch==0.5.3) from
a temporary site directory, replicates SeedVR2\'s exact NaMMRotaryEmbedding3d
construction (RotaryEmbedding(dim=128 // 3, freqs_for="lang", theta=10000)) and
writes reproducible freqs / cos / sin / rotated Q/K references plus a provenance
manifest for three window shapes:

  Oracle A [1,1,1]  minimal window
  Oracle B [4,4,4]  standard window
  Oracle C [2,3,5]  asymmetric window (axis order / W-fastest guard)

Run:  D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe \
        tests\seedvr2_attention_fidelity\seedvr2_mmrope_oracle.py \
        --site .phase10_1_dependency_tmp\site
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

SEEDVR2_COMMIT = "e4de8c24441a67e1b7df56abea10645059bb1185"
PACKAGE_VERSION = "0.5.3"
PACKAGE_ARCHIVE_SHA256 = "7c0297a21301fbd0d20cfdcba33be5d1ad0a46e33454168af08413dbf3e07457"
SOURCE_FILE_SHA256 = "80a45656e8d6aa8540c96a3fe97faae643da88182e5ae61d0e71c64c9ef61c1a"
CONFIGURED_ROPE_DIM = 128
HEAD_DIM = 128
HEADS = 4
THETA = 10000.0
PER_AXIS_DIM = CONFIGURED_ROPE_DIM // 3  # 42, exactly SeedVR2 rope_dim=3 integer division
EFFECTIVE_ROTARY_DIM = 3 * PER_AXIS_DIM  # 126
IDENTITY_TAIL_DIM = HEAD_DIM - EFFECTIVE_ROTARY_DIM  # 2

# (window_t, window_h, window_w, random_seed)
CASES = {
    "a": (1, 1, 1, 11),
    "b": (4, 4, 4, 22),
    "c": (2, 3, 5, 33),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(value: torch.Tensor, path: Path) -> None:
    path.write_bytes(value.detach().cpu().contiguous().float().numpy().tobytes())


def generate(site: Path, output_root: Path) -> None:
    sys.path.insert(0, str(site))
    from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb  # noqa: PLC0415

    source_file = site / "rotary_embedding_torch" / "rotary_embedding_torch.py"
    actual_source_sha = sha256_file(source_file)
    if actual_source_sha.lower() != SOURCE_FILE_SHA256.lower():
        raise RuntimeError(f"source sha mismatch: got {actual_source_sha}")

    rope = RotaryEmbedding(dim=PER_AXIS_DIM, freqs_for="lang", theta=THETA)
    for case_name, (wt, wh, ww, seed) in CASES.items():
        target = output_root / case_name
        target.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        tokens = wt * wh * ww

        freqs = rope.get_axial_freqs(wt, wh, ww)  # [T,H,W,126] float32
        freqs_flat = freqs.reshape(tokens, EFFECTIVE_ROTARY_DIM)
        angles = freqs_flat[:, 0::2]  # first dim of each interleaved pair; equal to the odd dim
        active_pairs = EFFECTIVE_ROTARY_DIM // 2
        cos64 = torch.ones(tokens, HEAD_DIM // 2, dtype=torch.float32)
        sin64 = torch.zeros(tokens, HEAD_DIM // 2, dtype=torch.float32)
        cos64[:, :active_pairs] = angles.cos()
        sin64[:, :active_pairs] = angles.sin()

        q = torch.randn(tokens, HEADS, HEAD_DIM, dtype=torch.float32)
        k = torch.randn(tokens, HEADS, HEAD_DIM, dtype=torch.float32)
        q_rot = apply_rotary_emb(freqs_flat, q.permute(1, 0, 2).float()).permute(1, 0, 2).to(q.dtype)
        k_rot = apply_rotary_emb(freqs_flat, k.permute(1, 0, 2).float()).permute(1, 0, 2).to(k.dtype)

        write(freqs_flat, target / "freqs.bin")
        write(cos64, target / "cos.bin")
        write(sin64, target / "sin.bin")
        write(q, target / "q_input.bin")
        write(k, target / "k_input.bin")
        write(q_rot, target / "q_rotated.bin")
        write(k_rot, target / "k_rotated.bin")

        manifest = {
            "package_version": PACKAGE_VERSION,
            "package_archive_sha256": PACKAGE_ARCHIVE_SHA256,
            "source_file_sha256": SOURCE_FILE_SHA256,
            "SeedVR2_commit": SEEDVR2_COMMIT,
            "configured_rope_dim": CONFIGURED_ROPE_DIM,
            "effective_rotary_dim": EFFECTIVE_ROTARY_DIM,
            "identity_tail_dim": IDENTITY_TAIL_DIM,
            "head_dim": HEAD_DIM,
            "heads": HEADS,
            "theta": THETA,
            "per_axis_dim": PER_AXIS_DIM,
            "axis_order": ["T", "H", "W"],
            "axis_dims": [PER_AXIS_DIM, PER_AXIS_DIM, PER_AXIS_DIM],
            "freq_formula": "1 / (theta ** (arange(0, dim, 2)[:dim // 2] / dim)) with dim=42, theta=10000, each frequency duplicated r=2",
            "window": [wt, wh, ww],
            "tensor_shapes": {
                "freqs": [tokens, EFFECTIVE_ROTARY_DIM],
                "cos": [tokens, HEAD_DIM // 2],
                "sin": [tokens, HEAD_DIM // 2],
                "q_input": [tokens, HEADS, HEAD_DIM],
                "k_input": [tokens, HEADS, HEAD_DIM],
                "q_rotated": [tokens, HEADS, HEAD_DIM],
                "k_rotated": [tokens, HEADS, HEAD_DIM],
            },
            "dtype": "float32",
            "random_seed": seed,
            "token_order": "T-major W-fastest: (lt * window_h + lh) * window_w + lw",
            "window_coordinates": "per-axis arange(0..L-1); real SeedVR2 applies text-length T offset l via vid_freqs[l:l+f, :h, :w] (l=0 in this standalone window fixture)",
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"seedvr2_mmrope_oracle: generated {len(CASES)} cases under {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path(".phase10_1_dependency_tmp/site"))
    parser.add_argument("--output", type=Path, default=Path("tests/seedvr2_attention_fidelity/artifacts/mmrope_oracle"))
    args = parser.parse_args()
    generate(args.site, args.output)


if __name__ == "__main__":
    main()
