"""Phase 16 shared paths/config for full SeedVR2 DiT streaming.

Derives the complete block schedule from the real checkpoint (never hardcodes
the block count): 32 real blocks 0..31; blocks 0..9 use the split .vid./.txt.
29-key layout (shared_weights=False, mm_layers=10), blocks 10..31 use the
shared .all. 15-key layout, and block 31 is the single last layer
(is_last_layer/vid_only).  Ranges: range_8 = [9..16] (8 normal blocks,
contains the Phase 14/15 anchor 16, excludes last-layer), range_16 = [8..23]
(16 normal blocks, contains anchor), range_full = [0..31].
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

# Real checkpoint-derived structure (verified: 32 blocks, last layer 31).
ANCHOR = 16
LAST_LAYER_INDEX = 31
ACTUAL_BLOCK_COUNT = 32
SPLIT_BLOCKS = list(range(0, 10))          # 29-key .vid./.txt. layout
SHARED_BLOCKS = list(range(10, 32))        # 15-key .all. layout
NORMAL_BLOCKS = list(range(0, 31))
ALL_BLOCKS = list(range(0, 32))

# Phase 16 ranges: 8/16 normal blocks centered on the Phase 14/15 anchor,
# full = all real blocks including the last layer.
RANGE_8 = list(range(9, 17))       # [9..16]
RANGE_16 = list(range(8, 24))      # [8..23]
RANGE_FULL = ALL_BLOCKS            # [0..31]
RANGES = {"range_8": RANGE_8, "range_16": RANGE_16, "range_full": RANGE_FULL}
BLOCK_SET = ALL_BLOCKS

# Phase 15 existing artifacts that can be reused read-only (hardlinks).
P15_EXISTING = [15, 16, 17, 18]


def variant_for(block_index: int) -> str:
    return f"real_seedvr2_block{block_index}_c2560_h20_hd128"


# Phase 16 local roots (git-ignored).
PHASE16_ROOT = REPO / ".phase16_local_artifacts"
EXPORT_TMP = REPO / ".phase16_export_tmp"
VISUAL_ROOT = REPO / ".phase16_visual_outputs"
TRACES = REPO / ".phase16_runtime_traces"
DIAG = REPO / ".phase16_diag"
MEM_PROFILES = REPO / ".phase16_memory_profiles"

# Phase 14/15 roots (read-only reuse).
PHASE14_ROOT = REPO / ".phase14_local_artifacts"
PHASE15_ROOT = REPO / ".phase15_local_artifacts"

# Real SeedVR2 3B single-block config (configs_3b/main.yaml).
C = 2560
HEADS = 20
HD = 128
EMB = 6 * C
QKV = 3 * C
MLP_HIDDEN = 6912
Q_MAX = 128
PAIRS = HD // 2
SCALE = HD ** -0.5
MM_LAYERS = 10  # configs_3b/main.yaml (blocks 0..9 split, 10..31 shared)

REAL_CFG = dict(
    vid_dim=C, txt_dim=C, emb_dim=EMB, heads=HEADS, head_dim=HD,
    expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
    qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
    shared_weights=True, rope_type="mmrope3d", rope_dim=128,
    is_last_layer=False, window=(4, 3, 3),
    window_method="720pwin_by_size_bysize",
)

# Executables / toolchain.
ADAPTER_EXE = REPO / "build" / "phase16-full-dit-release" / "full_dit_streaming_adapter.exe"
P15_ADAPTER_EXE = REPO / "build" / "phase15-multiblock-release" / "multiblock_streaming_adapter.exe"
P14_ADAPTER_EXE = REPO / "build" / "phase14-real-weight-release" / "real_block_adapter.exe"
PNNX_EXE = REPO / "build" / "pnnx-cpu" / "src" / "pnnx.exe"
LIBTORCH = Path(r"V:\AI\LibTorch\2.7.0+cpu\lib")
BUILD_DIR = REPO / "build" / "phase16-full-dit-release"

# Python interpreters.
GPU_PY = REPO / ".conda-cuda" / "python.exe"
CPU_PY = Path(r"C:\Users\35287\.conda\envs\seedvr2-block-probe\python.exe")

# Thresholds (Phase 16 spec sections 19/25).
CHAIN_REL_LIMITS = {8: 1e-2, 16: 2e-2, 32: 5e-2}
BLOCK_REL_LIMIT = 1e-3
VISUAL_PSNR_MIN_FULL = 34.0
VISUAL_SSIM_MIN_FULL = 0.95

# 19-stage runtime names (vid/txt pairs; split blocks 0-9 use separate physical
# artifacts per side, shared blocks 10-31 map both sides to one artifact).
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

# Unique physical artifacts for shared-weight blocks (10..31).
UNIQUE_STAGES = [
    "attn_norm", "qkv", "q_norm", "k_norm", "rope_apply",
    "attn_proj", "mlp_norm", "swiglu_gate", "swiglu_up",
    "swiglu_act", "swiglu_out", "head",
]

# Split-block (0..9) physical artifacts: per-side weighted stages + shared
# weight-free stages.
SPLIT_UNIQUE_STAGES = [
    "attn_norm", "qkv_vid", "qkv_txt",
    "q_norm_vid", "q_norm_txt", "k_norm_vid", "k_norm_txt",
    "rope_apply", "attn_proj_vid", "attn_proj_txt", "mlp_norm",
    "swiglu_gate_vid", "swiglu_gate_txt",
    "swiglu_up_vid", "swiglu_up_txt",
    "swiglu_act", "swiglu_out_vid", "swiglu_out_txt",
    "head",
]

# Weight-free stages shared across all blocks (deduplicated).
WEIGHT_FREE_STAGES = ["attn_norm", "mlp_norm", "rope_apply", "swiglu_act", "head"]
