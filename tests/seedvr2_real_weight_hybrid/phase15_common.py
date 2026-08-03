"""Phase 15 shared paths/config for real-weight multi-block streaming hybrid work.

All heavy artifacts (real checkpoint, real-weight FP32 stage models, reference
traces, captured official-context inputs, visual outputs, runtime traces,
diagnostics) live under git-ignored directories (`.phase14_local_checkpoint/`,
`.phase15_local_artifacts/`, `.phase15_export_tmp/`,
`.phase15_visual_outputs/`, `.phase15_runtime_traces/`, `.phase15_diag/`).
Only small scripts, manifests, mapping/parity JSON and reports are committed.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(r"D:\AIProjects\seedvr2-ncnn")
HERE = Path(__file__).resolve().parent

CHECKPOINT_DIR = REPO / ".phase14_local_checkpoint"
MAIN_CKPT = CHECKPOINT_DIR / "seedvr2_ema_3b.pth"
EXPECTED_MAIN_SHA256 = "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b"
VAE_CKPT = CHECKPOINT_DIR / "ema_vae.pth"
POS_EMB = CHECKPOINT_DIR / "pos_emb.pt"
NEG_EMB = CHECKPOINT_DIR / "neg_emb.pt"

SEEDVR2_ROOT = Path(r"D:\AIProjects\third_party\SeedVR2")
SEEDVR2_HEAD = "e4de8c24441a67e1b7df56abea10645059bb1185"

# Phase 14 anchor and Phase 15 block ranges (from the real checkpoint schema:
# 32 blocks, last layer = 31, normal blocks 0..30; anchor = floor(32/2) = 16).
ANCHOR = 16
LAST_LAYER_INDEX = 31
RANGE_1 = [16]
RANGE_2 = [16, 17]
RANGE_4 = [15, 16, 17, 18]
RANGES = {"range_1": RANGE_1, "range_2": RANGE_2, "range_4": RANGE_4}
BLOCK_SET = sorted(set(RANGE_4))  # [15, 16, 17, 18]


def variant_for(block_index: int) -> str:
    return f"real_seedvr2_block{block_index}_c2560_h20_hd128"


# Phase 15 local roots (git-ignored).
PHASE15_ROOT = REPO / ".phase15_local_artifacts"
EXPORT_TMP = REPO / ".phase15_export_tmp"
VISUAL_ROOT = REPO / ".phase15_visual_outputs"
TRACES = REPO / ".phase15_runtime_traces"
DIAG = REPO / ".phase15_diag"

# Phase 14 roots (read-only reuse for the block-16 anchor artifact).
PHASE14_ROOT = REPO / ".phase14_local_artifacts"
P14_VARIANT_16 = variant_for(ANCHOR)

# Real SeedVR2 3B single-block config (configs_3b/main.yaml) - same as Phase 14.
C = 2560            # hidden_dim / txt_dim
HEADS = 20
HD = 128            # head_dim
EMB = 6 * C         # AdaSingle emb_dim
QKV = 3 * C         # qkv output dim
MLP_HIDDEN = 6912   # real SwiGLU hidden
Q_MAX = 128
PAIRS = HD // 2
SCALE = HD ** -0.5

REAL_CFG = dict(
    vid_dim=C, txt_dim=C, emb_dim=EMB, heads=HEADS, head_dim=HD,
    expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
    qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
    shared_weights=True, rope_type="mmrope3d", rope_dim=128,
    is_last_layer=False, window=(4, 3, 3),
    window_method="720pwin_by_size_bysize",
)

# Executables / toolchain.
ADAPTER_EXE = REPO / "build" / "phase15-multiblock-release" / "multiblock_streaming_adapter.exe"
P14_ADAPTER_EXE = REPO / "build" / "phase14-real-weight-release" / "real_block_adapter.exe"
PNNX_EXE = REPO / "build" / "pnnx-cpu" / "src" / "pnnx.exe"
LIBTORCH = Path(r"V:\AI\LibTorch\2.7.0+cpu\lib")
BUILD_DIR = REPO / "build" / "phase15-multiblock-release"

# Python interpreters.
GPU_PY = REPO / ".conda-cuda" / "python.exe"
CPU_PY = Path(r"C:\Users\35287\.conda\envs\seedvr2-block-probe\python.exe")

# Thresholds (Phase 15 spec section 13/19).
CHAIN_REL_LIMITS = {1: 1e-3, 2: 2e-3, 4: 5e-3}
RESIDENT_STREAMING_MAX_ABS = 1e-7
VISUAL_PSNR_MIN_4 = 38.0
VISUAL_SSIM_MIN_4 = 0.98
P14_VISUAL_PSNR = 52.5   # Phase 14 single-block baseline (not to be faked)
P14_VISUAL_SSIM = 0.9975

# 19-stage runtime names; shared-weight blocks map vid/txt pairs to one artifact.
BASE_STAGES = [
    "attn_norm", "qkv_vid", "qkv_txt",
    "q_norm_vid", "q_norm_txt", "k_norm_vid", "k_norm_txt",
    "rope_apply", "attn_proj_vid", "attn_proj_txt", "mlp_norm",
    "swiglu_gate_vid", "swiglu_gate_txt",
    "swiglu_up_vid", "swiglu_up_txt",
    "swiglu_act", "swiglu_out_vid", "swiglu_out_txt",
]
HEAD_STAGES = [f"head{i}" for i in range(HEADS)]
MODEL_STAGES = BASE_STAGES + ["head"]

# Unique physical artifacts per shared-weight block (vid/txt share one export).
UNIQUE_STAGES = [
    "attn_norm", "qkv", "q_norm", "k_norm", "rope_apply",
    "attn_proj", "mlp_norm", "swiglu_gate", "swiglu_up",
    "swiglu_act", "swiglu_out", "head",
]
