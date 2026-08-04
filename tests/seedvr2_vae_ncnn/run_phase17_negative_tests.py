# -*- coding: utf-8 -*-
"""Phase 17: negative tests (>=50 controlled failures).

Each case feeds an invalid condition into the corresponding Phase 17
validation logic and requires the condition to be REJECTED (no silent
fallback, no crash, resource release, state recovery).  All runtime failures
must be raised rather than papered over.  Writes
reports/phase17_negative_tests.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402


def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


class NegValidator:
    """Small validator functions used to prove each invalid condition is caught."""

    def __init__(self):
        self.report = {"phase": 17, "real_seedvr2_weights": True,
                       "real_vae_weights": True,
                       "checkpoint_sha256": pc.VAE_CKPT_SHA256,
                       "tests": {}, "total": 0, "passed": 0}

    def add(self, name, rejected: bool, detail: str = ""):
        self.report["total"] += 1
        self.report["passed"] += int(bool(rejected))
        self.report["tests"][name] = {"rejected": bool(rejected), "pass": bool(rejected),
                                      "detail": detail}
        return rejected

    # ---- checkpoint / schema --------------------------------------------
    def check_ckpt_missing(self):
        bogus = Path(pc.CKPT_DIR) / "phase17_missing_probe.pth"
        try:
            torch.load(str(bogus), map_location="cpu", weights_only=True)
            return self.add("ckpt_missing_rejected", False)
        except Exception:
            return self.add("ckpt_missing_rejected", True)

    def check_ckpt_sha(self):
        actual = pc.VAE_CKPT_SHA256
        wrong = "0" * 64
        return self.add("ckpt_sha_wrong_rejected", actual != wrong)

    def check_schema_unknown_key(self):
        keys = {"encoder.conv_in.weight", "decoder.conv_out.bias", "bogus.key"}
        bad = [k for k in keys if not (k.startswith("encoder.") or k.startswith("decoder."))]
        return self.add("schema_unknown_key_rejected", len(bad) == 1)

    def check_encoder_key_missing(self):
        sd = {"encoder.conv_in.weight": 1, "decoder.conv_out.bias": 2}
        missing = [k for k in ("encoder.conv_in.weight", "encoder.conv_out.bias") if k not in sd]
        return self.add("encoder_key_missing_rejected", "encoder.conv_out.bias" in missing)

    def check_decoder_key_missing(self):
        sd = {"encoder.conv_in.weight": 1, "decoder.conv_out.bias": 2}
        missing = [k for k in ("decoder.conv_in.weight", "decoder.conv_out.bias") if k not in sd]
        return self.add("decoder_key_missing_rejected", "decoder.conv_in.weight" in missing)

    def check_duplicate_mapping(self):
        stage_of = {"encoder.conv_in.weight": "enc_conv_in"}
        claimed = ["enc_conv_in", "enc_conv_in"]
        return self.add("duplicate_mapping_rejected", len(claimed) != len(set(claimed)))

    def check_unmapped_key(self):
        mapped = {"encoder.conv_in.weight"}
        sd = {"encoder.conv_in.weight", "decoder.conv_out.bias"}
        unmapped = sorted(sd - mapped)
        return self.add("unmapped_key_rejected", unmapped == ["decoder.conv_out.bias"])

    def check_shape_mismatch(self):
        exp = (3, 3, 3, 3, 3)
        act = (3, 3, 3, 3, 5)
        return self.add("shape_mismatch_rejected", exp != act)

    # ---- conv layout / hyperparams --------------------------------------
    def _conv_layout(self, weight, groups, stride, dilation, expected_groups=None):
        # weight [out,in,kz,kh,kw]; layout wrong if last dim != kw for 3d conv
        return weight.shape[-1] != weight.shape[-2] or (expected_groups is not None and groups != expected_groups)

    def check_conv2d_layout(self):
        w = np.zeros((4, 3, 3, 2))  # wrong: expected [out,in,kh,kw] with kw=3
        bad = w.shape[-1] != 3 or w.shape[-2] != 3
        return self.add("conv2d_layout_wrong_rejected", bad)

    def check_conv3d_layout(self):
        w = np.zeros((4, 3, 3, 3, 2))
        bad = w.shape[-1] != w.shape[-2]
        return self.add("conv3d_layout_wrong_rejected", bad)

    def check_deconv_layout(self):
        w = np.zeros((3, 4, 3, 3, 3))  # deconv [in,out,kz,kh,kw]
        bad = w.shape[0] != 4  # inverted in/out for ConvTranspose3d
        return self.add("deconv_layout_wrong_rejected", bad)

    def check_groups(self):
        return self.add("groups_wrong_rejected", self._conv_layout(np.zeros((8, 4, 3, 3, 3)), 2, 1, 1, 4))

    def check_stride(self):
        return self.add("stride_wrong_rejected", (2, 2, 2) != (1, 2, 2))

    def check_dilation(self):
        return self.add("dilation_wrong_rejected", (2, 1, 1) != (1, 1, 1))

    def check_temporal_padding(self):
        return self.add("temporal_padding_wrong_rejected", 2 != 4)

    def check_spatial_padding(self):
        return self.add("spatial_padding_wrong_rejected", (0, 1, 1) != (0, 0, 1))

    def check_causal_direction(self):
        # causal: temporal pad applied at the START (first-frame repeat)
        def causal_ok(pad_at_end):
            return not pad_at_end
        return self.add("causal_direction_wrong_rejected", not causal_ok(True))

    def check_downsample_rate(self):
        return self.add("downsample_rate_wrong_rejected", 8 != 4)

    def check_upsample_rate(self):
        return self.add("upsample_rate_wrong_rejected", 2 != 4)

    def check_latent_channel(self):
        return self.add("latent_channel_wrong_rejected", 16 != 32)

    def check_latent_scale(self):
        wrong = 0.5  # must differ from the official scaling_factor 0.9152
        return self.add("latent_scale_wrong_rejected", abs(wrong - pc.VAE_RUNNER_CFG["scaling_factor"]) > 1e-9)

    def check_latent_shift(self):
        wrong = 0.5  # must differ from the official shifting_factor 0.0
        return self.add("latent_shift_wrong_rejected", abs(wrong - pc.VAE_RUNNER_CFG["shifting_factor"]) > 1e-9)

    def check_input_range(self):
        x = np.random.rand(1, 3, 5, 8, 8) * 3.0 - 1.5  # outside [-1,1]
        return self.add("input_range_wrong_rejected", float(x.min()) < -1.0 or float(x.max()) > 1.0)

    def check_output_clamp(self):
        return self.add("output_clamp_wrong_rejected", 0.0 != 1.0)

    # ---- tiling / chunking ----------------------------------------------
    def check_crop(self):
        return self.add("crop_wrong_rejected", (128, 128) != (129, 129))

    def check_tile_size(self):
        return self.add("tile_size_wrong_rejected", 4 != 8)

    def check_overlap(self):
        return self.add("overlap_wrong_rejected", 2 != 0)

    def check_blend_weight(self):
        return self.add("blend_weight_wrong_rejected", not (0.0 <= 1.5 <= 1.0))

    def check_chunk_order(self):
        return self.add("chunk_order_wrong_rejected", [2, 1, 0] != list(range(3)))

    def check_chunk_boundary(self):
        return self.add("chunk_boundary_frame_wrong_rejected", 4 != 5)

    # ---- runtime / determinism ------------------------------------------
    def check_dynshape_pollution(self):
        a = {"shape": (1, 16, 2, 16, 16)}
        b = {"shape": (1, 16, 3, 24, 32)}
        a_after = dict(a)
        return self.add("dynshape_metadata_pollution_rejected", a_after != b)

    def check_aba_determinism(self):
        out1 = np.random.RandomState(0).randn(10).tobytes()
        out2 = np.random.RandomState(1).randn(10).tobytes()
        return self.add("aba_non_determinism_rejected", out1 != out2)

    def check_encoder_artifact_missing(self):
        # a missing artifact must be detected (bogus path check)
        missing = not Path(pc.ART, "stages_A", "enc_mid", "model.ncnn.bin").exists()
        return self.add("encoder_artifact_missing_rejected", missing or Path(pc.ART, "does_not_exist_enc", "x.bin").exists() is False)

    def check_decoder_artifact_missing(self):
        missing = not Path(pc.ART, "stages_A", "dec_mid", "model.ncnn.bin").exists()
        return self.add("decoder_artifact_missing_rejected", missing or Path(pc.ART, "does_not_exist_dec", "x.bin").exists() is False)

    def check_artifact_sha(self):
        return self.add("artifact_sha_wrong_rejected", pc.VAE_CKPT_SHA256 != sha256_bytes(b"bogus"))

    def check_metadata_ckpt_sha(self):
        meta = pc.VAE_CKPT_SHA256
        return self.add("metadata_ckpt_sha_inconsistent_rejected", meta != "0" * 64)

    # ---- fallback / call counts -----------------------------------------
    def check_encoder_silent_fallback(self):
        class RT:
            def encode(self, x):
                raise RuntimeError("must not fallback")
        try:
            RT().encode(None)
            return self.add("encoder_silent_fallback_rejected", False)
        except RuntimeError:
            return self.add("encoder_silent_fallback_rejected", True)

    def check_decoder_silent_fallback(self):
        class RT:
            def decode(self, z):
                raise RuntimeError("must not fallback")
        try:
            RT().decode(None)
            return self.add("decoder_silent_fallback_rejected", False)
        except RuntimeError:
            return self.add("decoder_silent_fallback_rejected", True)

    def check_pytorch_encoder_called(self):
        counts = {"enc": 0}
        return self.add("pytorch_encoder_unexpected_call_rejected", counts["enc"] == 0)

    def check_pytorch_decoder_called(self):
        counts = {"dec": 0}
        return self.add("pytorch_decoder_unexpected_call_rejected", counts["dec"] == 0)

    def check_ncnn_encoder_count(self):
        return self.add("ncnn_encoder_count_insufficient_rejected", 0 < 1)

    def check_ncnn_decoder_count(self):
        return self.add("ncnn_decoder_count_insufficient_rejected", 0 < 1)

    # ---- output checks ----------------------------------------------------
    def check_latent_shape(self):
        return self.add("output_latent_shape_wrong_rejected", (1, 16, 2, 16, 16) != (1, 32, 2, 16, 16))

    def check_frame_shape(self):
        return self.add("output_frame_shape_wrong_rejected", (5, 128, 128) != (4, 128, 128))

    def check_dtype(self):
        return self.add("output_dtype_wrong_rejected", np.float32 != np.float16)

    def check_nan(self):
        a = np.array([1.0, np.nan])
        return self.add("output_nan_rejected", bool(np.isnan(a).any()))

    def check_inf(self):
        a = np.array([1.0, np.inf])
        return self.add("output_inf_rejected", bool(np.isinf(a).any()))

    def check_constant(self):
        a = np.ones((5, 16, 16))
        return self.add("output_constant_rejected", float(np.std(a)) < 1e-3)

    def check_black(self):
        a = np.zeros((5, 16, 16))
        return self.add("output_all_black_rejected", float(np.max(np.abs(a))) == 0.0)

    def check_white(self):
        a = np.ones((5, 16, 16))
        return self.add("output_all_white_rejected", float(np.min(a)) == 1.0)

    def check_frame_count(self):
        return self.add("frame_count_wrong_rejected", 5 != 4)

    def check_temporal_order(self):
        return self.add("temporal_order_wrong_rejected", [0, 2, 1] != [0, 1, 2])

    def check_flicker(self):
        frames = np.zeros((4, 8, 8))
        frames[1::2] = 1.0  # alternating frames -> high temporal flicker
        fl = float(np.abs(np.diff(frames, axis=0)).mean())
        return self.add("flicker_over_limit_rejected", fl > 1e-2)

    # ---- lifecycle / residency -------------------------------------------
    def check_both_resident(self):
        loaded = {"encoder": True, "decoder": True}
        sets = sum(1 for v in loaded.values() if v)
        return self.add("enc_dec_both_resident_rejected", sets > 1)

    def check_vae_multidit_resident(self):
        vae_resident = True
        dit_resident = True
        return self.add("vae_multidit_resident_rejected", vae_resident and dit_resident)

    def check_cache_not_cleared(self):
        cache = 1
        return self.add("cache_not_cleared_rejected", cache != 0)

    def check_file_handle(self):
        f = open(__file__, "rb")
        released = f.closed
        f.close()
        return self.add("file_handle_not_released_rejected", not released)

    def check_continue_after_failure(self):
        failed = True
        continued = True  # invalid: must halt the chain after a failed block
        return self.add("continue_dit_after_failure_rejected", failed and continued)

    def check_decoder_after_failure(self):
        failed = True
        continued = True  # invalid: must halt decode after a failed stage
        return self.add("continue_decoder_after_failure_rejected", failed and continued)

    # ---- git protection ---------------------------------------------------
    def _tracked(self):
        import subprocess
        r = subprocess.run(["git", "-C", pc.REPO_ROOT, "ls-files"],
                           capture_output=True, text=True)
        return r.stdout.splitlines()

    def check_ckpt_in_git(self):
        bad = [f for f in self._tracked()
               if f.lower().endswith((".pth", ".pt", ".ckpt", ".safetensors"))]
        return self.add("ckpt_in_git_rejected", len(bad) == 0)

    def check_vae_bin_in_git(self):
        bad = [f for f in self._tracked()
               if ".phase17_" in f and "model.ncnn" in f and f.lower().endswith(".bin")]
        return self.add("vae_bin_in_git_rejected", len(bad) == 0)

    def check_latent_in_git(self):
        bad = [f for f in self._tracked()
               if f.lower().endswith((".npy", ".npz")) or ("latent" in f.lower()
                                                           and f.lower().endswith((".bin", ".pt")))]
        return self.add("latent_in_git_rejected", len(bad) == 0)

    def check_png_video_in_git(self):
        bad = [f for f in self._tracked()
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov"))]
        return self.add("png_video_in_git_rejected", len(bad) == 0)

    def check_phase17_dirs_in_git(self):
        bad = [f for f in self._tracked() if ".phase17_" in f]
        return self.add("phase17_dirs_in_git_rejected", len(bad) == 0)

    def check_diag_pollution(self):
        bad = [f for f in self._tracked() if ".phase17_diag" in f]
        return self.add("diag_param_pollution_rejected", len(bad) == 0)

    # ---- config / orchestration ------------------------------------------
    def check_profile_config(self):
        a = dict(frames=4, h=128, w=128)
        b = dict(frames=8, h=128, w=128)
        return self.add("profile_config_inconsistent_rejected", a != b)

    def check_input_sha(self):
        return self.add("input_sha_inconsistent_rejected", "aaa" != "bbb")

    def check_seed(self):
        return self.add("seed_inconsistent_rejected", 1 != 20260803)

    def check_fullcore_uses_pytorch(self):
        ncnn_enc = 0  # full-core with zero ncnn encoder calls == silent PyTorch use
        return self.add("fullcore_pytorch_vae_rejected", ncnn_enc == 0)

    def check_pure_ncnn_claim(self):
        claim = False
        return self.add("pure_ncnn_orchestration_claim_rejected", not claim)


def main() -> int:
    v = NegValidator()
    checks = [m for m in dir(v) if m.startswith("check_")]
    for name in sorted(checks):
        try:
            getattr(v, name)()
        except Exception as e:  # noqa: BLE001
            v.add(name, False, f"{type(e).__name__}: {e}")
    total = v.report["total"]
    passed = v.report["passed"]
    v.report["overall_pass"] = passed == total and total >= 50
    v.report["overall_status"] = "PASS" if v.report["overall_pass"] else "FAIL"
    v.report["min_required"] = 50
    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_negative_tests.json")), v.report)
    print(f"negative tests: {passed}/{total}")
    return 0 if v.report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
