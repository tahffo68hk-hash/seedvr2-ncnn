# -*- coding: utf-8 -*-
"""Phase 17: VAE tiling/chunking contract (from official source + capture).

The official sliced path (split_size=4) was exercised in capture and proven
equivalent to whole-tensor for T=9/T=17, and the raw T=8 input crash of the
official sliced path is documented.  Profile B uses the official T=9 padded
convention (4k+1), so slicing is never required for correctness in the
profiles used here.  Writes reports/phase17_vae_tiling_contract.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402


def main() -> int:
    io_contract = json.loads(
        (Path(pc.REPO_ROOT) / "reports" / "phase17_vae_io_contract.json").read_text(encoding="utf-8"))

    report = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "real_vae_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "official_slicing": {
            "class": "VideoAutoencoderKLWrapper",
            "slicing_sample_min_size": 4,
            "temporal_downsample_factor": 4,
            "memory_state_path": "causal_conv_slice_inputs / causal_conv_gather_outputs",
            "whole_tensor_used_in_phase17": True,
            "profiles_use_4k1_padded_convention": True,
        },
        "slicing_equivalence": io_contract.get("slicing_equivalence", {}),
        "t8_sliced_crash": io_contract.get("t8_sliced_crash", {}),
        "temporal_padding_rule": {
            "rule": "pad by repeating the last frame until T = 4k+1 (cut_videos in the official runner)",
            "profile_a": "4 frames -> T=5",
            "profile_b": "8 frames -> T=9",
        },
        "host_managed": [
            "dynamic shape", "tiling", "chunking", "padding", "temporal cut",
        ],
        "ncnn_scope": "per-stage FP32 models consume the padded whole tensor; "
                     "no ncnn-side tiling is needed for the validated profiles",
        "tiling_semantics_match_official": True,
        "android_vulkan": False,
    }
    pc.write_report(str(Path(pc.REPO_ROOT) / "reports" / "phase17_vae_tiling_contract.json"), report)
    print("wrote phase17_vae_tiling_contract.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
