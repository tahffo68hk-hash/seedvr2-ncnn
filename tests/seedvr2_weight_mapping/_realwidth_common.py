"""Phase 13 shared paths / config for the real-width SeedVR2 block work.

All heavy artifacts (random state_dict, exported real-width FP32 stage models,
RW-A reference bins) live under the git-ignored `.phase13_local_artifacts/`
and `.phase13_export_tmp/` directories; only small manifests, the mapping
contract JSON, scripts, tests and reports are committed.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(r"D:\AIProjects\seedvr2-ncnn")
HERE = Path(__file__).resolve().parent

VARIANT = "realwidth_seedvr2_dit_block_c2560_h20_hd128_random"
LOCAL_ROOT = REPO / ".phase13_local_artifacts" / VARIANT
EXPORT_TMP = REPO / ".phase13_export_tmp"
MODELS = LOCAL_ROOT / "models"
PARAMS = LOCAL_ROOT / "params"
REFERENCE = LOCAL_ROOT / "reference"
STATE_DICT = LOCAL_ROOT / "block_state.pt"
INVENTORY = LOCAL_ROOT / "realwidth_state_dict_inventory.json"
PROVENANCE = LOCAL_ROOT / "provenance.json"

SEED = 20260803
C = 2560            # hidden_dim / txt_dim (real SeedVR2 3B single block)
HEADS = 20
HD = 128            # head_dim (real, preserved)
EMB = 6 * C         # AdaSingle emb_dim
QKV = 3 * C         # qkv output dim
MLP_HIDDEN = 6912   # real SwiGLUMLP hidden (256 * ceil(2*2560*4/3/256))
Q_MAX = 128         # fixed capacity for rope_apply / attention heads
PAIRS = HD // 2     # 64
SCALE = HD ** -0.5

REAL_CFG = dict(
    vid_dim=C, txt_dim=C, emb_dim=EMB, heads=HEADS, head_dim=HD,
    expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
    qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
    shared_weights=False, rope_type="mmrope3d", rope_dim=128,
    is_last_layer=False, window=(4, 3, 3),
    window_method="720pwin_by_size_bysize",
)

# Stage model names required by HybridDiTBlockRuntime (base block, 19 models).
BASE_STAGES = [
    "attn_norm", "qkv_vid", "qkv_txt",
    "q_norm_vid", "q_norm_txt", "k_norm_vid", "k_norm_txt",
    "rope_apply", "attn_proj_vid", "attn_proj_txt", "mlp_norm",
    "swiglu_gate_vid", "swiglu_gate_txt",
    "swiglu_up_vid", "swiglu_up_txt",
    "swiglu_act", "swiglu_out_vid", "swiglu_out_txt",
]
# The 20 attention heads share ONE weight-free head model (`models/head`),
# exactly like Phase 12; the metadata "heads" list points all heads at it.
HEAD_STAGES = [f"head{i}" for i in range(HEADS)]  # attention-only, no weights
MODEL_STAGES = BASE_STAGES + ["head"]
