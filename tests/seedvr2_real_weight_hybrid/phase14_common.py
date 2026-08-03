"""Phase 14 shared paths/config for real-weight SeedVR2 block work.

All heavy artifacts (real checkpoint, exported real-weight FP32 stage models,
reference bins, captured official-context inputs, visual outputs) live under
git-ignored directories (`.phase14_local_checkpoint/`,
`.phase14_local_artifacts/`, `.phase14_visual_outputs/`, `.phase14_export_tmp/`).
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

VARIANT = "real_seedvr2_block16_c2560_h20_hd128"
LOCAL_ROOT = REPO / ".phase14_local_artifacts" / VARIANT
EXPORT_TMP = REPO / ".phase14_export_tmp"
VISUAL_ROOT = REPO / ".phase14_visual_outputs"
MODELS = LOCAL_ROOT / "models"
PARAMS = LOCAL_ROOT / "params"
REFERENCE = LOCAL_ROOT / "reference"
CAPTURE = LOCAL_ROOT / "official_context_capture"

SEEDVR2_ROOT = Path(r"D:\AIProjects\third_party\SeedVR2")
SEEDVR2_HEAD = "e4de8c24441a67e1b7df56abea10645059bb1185"

# Real SeedVR2 3B single-block config (configs_3b/main.yaml).
C = 2560            # hidden_dim / txt_dim
HEADS = 20
HD = 128            # head_dim
EMB = 6 * C         # AdaSingle emb_dim
QKV = 3 * C         # qkv output dim
MLP_HIDDEN = 6912   # real SwiGLU hidden
Q_MAX = 128
PAIRS = HD // 2
SCALE = HD ** -0.5

BLOCK_INDEX = 16
BLOCK_PREFIX = f"blocks.{BLOCK_INDEX}."

REAL_CFG = dict(
    vid_dim=C, txt_dim=C, emb_dim=EMB, heads=HEADS, head_dim=HD,
    expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
    qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
    shared_weights=True, rope_type="mmrope3d", rope_dim=128,
    is_last_layer=False, window=(4, 3, 3),
    window_method="720pwin_by_size_bysize",
)

# Runtime stage names (Phase 13 base block, 19 stages).  For the shared-weight
# block 16 every vid/txt pair maps to the SAME physical artifact (one export).
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

# Unique physical artifacts exported for the shared block (vid/txt share one).
UNIQUE_STAGES = [
    "attn_norm", "qkv", "q_norm", "k_norm", "rope_apply",
    "attn_proj", "mlp_norm", "swiglu_gate", "swiglu_up",
    "swiglu_act", "swiglu_out", "head",
]
