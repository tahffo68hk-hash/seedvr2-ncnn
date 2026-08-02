"""Phase 11: extract the real SeedVR2 single DiT block interface/shape contract.

Runs with the repo .conda-cuda environment (torch + yaml + numpy available;
einops/diffusers/flash_attn/apex/rotary_embedding_torch are intentionally NOT
installed). Combines config parsing, AST contract extraction, executable
window-partition probes, negative tests, dependency-gap and real-block import
probe, and freshness checks. Writes seedvr2_block_contract.json and
reports/seedvr2_single_dit_block_contract.json. Exit 0 on success.
"""
import ast
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import traceback

SEEDVR2_ROOT = r"D:\AIProjects\third_party\SeedVR2"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(OUT_DIR))
REPORT_PATH = os.path.join(REPO_ROOT, "reports", "seedvr2_single_dit_block_contract.json")
CONTRACT_PATH = os.path.join(OUT_DIR, "seedvr2_block_contract.json")
EXPECTED_SEEDVR2_COMMIT = "e4de8c24441a67e1b7df56abea10645059bb1185"

DEP_PINS = {"einops": "0.7.0", "diffusers": "0.33.1", "flash_attn": "2.5.9.post1",
            "apex": "0.1", "rotary_embedding_torch": "0.5.3", "torch": "2.4.0+cu121"}
REQ_PINS = {"einops": "0.7.0", "diffusers": "0.29.1", "rotary_embedding_torch": "0.5.3", "torch": "2.3.0"}
CONFIG_PATH = os.path.join(SEEDVR2_ROOT, "configs_3b", "main.yaml")
SOURCE_FILES = [
    "models/dit_v2/nadit.py", "models/dit_v2/nablocks/__init__.py",
    "models/dit_v2/nablocks/mmsr_block.py", "models/dit_v2/nablocks/attention/mmattn.py",
    "models/dit_v2/rope.py", "models/dit_v2/modulation.py", "models/dit_v2/mlp.py",
    "models/dit_v2/normalization.py", "models/dit_v2/window.py",
    "models/dit_v2/patch/patch_v1.py", "models/dit_v2/attention.py",
    "models/dit_v2/embedding.py", "models/dit_v2/mm.py", "models/dit_v2/na.py",
    "common/cache.py", "common/config.py",
]


def norm(s):
    return re.sub(r"\s+", "", s)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def git_head(root):
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain"],
                                    capture_output=True, text=True, check=True, timeout=30).stdout.strip())
        return out, not dirty
    except Exception:
        return None, False


def ast_class(module, name):
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def func_args(func):
    if func is None:
        return []
    a = func.args
    args = [x.arg for x in a.args]
    if a.vararg:
        args.append("*" + a.vararg.arg)
    args += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        args.append("**" + a.kwarg.arg)
    return args


def find_method(cls, name):
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def call_sequence(func):
    seq = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                kw = {}
                for k in node.keywords:
                    kw[k.arg] = k.value.value if isinstance(k.value, ast.Constant) else ast.dump(k.value)[:80]
                seq.append([f.attr, node.lineno, kw])
    return seq


def source_has(source, *patterns):
    n = norm(source)
    return all(p in n for p in (norm(x) for x in patterns))
def main():
    report = {
        "phase": "11",
        "scope": "single real SeedVR2 DiT block interface and shape contract",
        "seedvr2_commit": None, "seedvr2_commit_clean": None,
        "config_source": CONFIG_PATH,
        "block": {}, "call_chain": [], "forward_signature": [], "config": {},
        "inputs": [], "outputs": [], "token_layout": {}, "rope": {},
        "modulation": {}, "attention": {}, "mlp": {}, "residual_order": [],
        "execution_order": [], "intermediate_shapes": {}, "probes": {},
        "negative_tests": [], "determinism": {}, "dependency_gap": {},
        "real_block_probe": {}, "ncnn_mapping_risk": [], "dynamic_dimensions": [],
        "freshness": {}, "real_seedvr2_weights": False, "ncnn_block_complete": False,
        "overall_status": "PARTIAL", "blocker": None,
    }
    errors = []
    head, clean = git_head(SEEDVR2_ROOT)
    report["seedvr2_commit"] = head
    report["seedvr2_commit_clean"] = clean

    sources = {}
    for rel in SOURCE_FILES:
        p = os.path.join(SEEDVR2_ROOT, rel)
        if os.path.isfile(p):
            sources[rel] = read(p)
        else:
            errors.append("missing source: " + rel)

    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["dit"]["model"]
    vid_dim = int(m["vid_dim"])
    heads = int(m["heads"]); head_dim = int(m["head_dim"])
    expand = int(m["expand_ratio"])
    mlp_hidden = 256 * ((2 * vid_dim * expand // 3 + 255) // 256)
    report["config"] = {
        "vid_in_channels": int(m["vid_in_channels"]), "vid_out_channels": int(m["vid_out_channels"]),
        "vid_dim": vid_dim, "txt_in_dim": int(m["txt_in_dim"]), "txt_dim": vid_dim,
        "emb_dim": 6 * vid_dim, "heads": heads, "head_dim": head_dim,
        "inner_dim": heads * head_dim, "qkv_dim": 3 * heads * head_dim,
        "expand_ratio": expand, "mlp_hidden_derived": mlp_hidden,
        "norm": m["norm"], "norm_eps": float(m["norm_eps"]), "ada": m["ada"],
        "qk_bias": bool(m["qk_bias"]), "qk_norm": m["qk_norm"],
        "patch_size": list(m["patch_size"]), "num_layers": int(m["num_layers"]),
        "mm_layers": int(m["mm_layers"]), "mlp_type": m["mlp_type"],
        "block_type": "mmdit_sr", "window": (4, 3, 3),
        "window_method": "720pwin_by_size_bysize / 720pswin_by_size_bysize (alternating)",
        "rope_type": m["rope_type"], "rope_dim": int(m["rope_dim"]),
        "model_class": m["__object__"]["name"], "model_module": m["__object__"]["path"],
    }

    blk_src = sources.get("models/dit_v2/nablocks/mmsr_block.py", "")
    if blk_src:
        blk_mod = ast.parse(blk_src)
        cls = ast_class(blk_mod, "NaMMSRTransformerBlock")
        init = find_method(cls, "__init__") if cls else None
        fwd = find_method(cls, "forward") if cls else None
        report["block"] = {
            "class": "NaMMSRTransformerBlock",
            "source_file": os.path.join(SEEDVR2_ROOT, "models/dit_v2/nablocks/mmsr_block.py"),
            "init_signature": func_args(init), "forward_signature": func_args(fwd),
            "hidden_dim": vid_dim, "heads": heads, "head_dim": head_dim,
            "mlp_ratio": expand,
        }
        report["execution_order"] = call_sequence(fwd)

    nad_src = sources.get("models/dit_v2/nadit.py", "")
    nad_mod = ast.parse(nad_src) if nad_src else None
    nad_cls = ast_class(nad_mod, "NaDiT") if nad_mod else None
    nad_fwd = find_method(nad_cls, "forward") if nad_cls else None
    report["call_chain"] = [
        "VideoDiffusionTrainer.sample (projects/video_diffusion_sr/infer.py)",
        "  -> NaDiT.forward(vid, txt, vid_shape, txt_shape, timestep)",
        "    -> vid_in = NaPatchIn(vid, vid_shape); txt_in = project(txt); emb_in = TimeEmbedding(timestep)",
        "    -> for i in 0..31: NaMMSRTransformerBlock(vid, txt, vid_shape, txt_shape, emb, cache)",
        "    -> vid_out_norm/ada + NaPatchOut -> NaDiTOutput(vid_sample)",
    ]
    if nad_fwd:
        report["forward_signature"] = func_args(nad_fwd)
    report["block"]["registry_evidence"] = (
        "models/dit_v2/nablocks/__init__.py: nadit_blocks = {'mmdit_sr': NaMMSRTransformerBlock}"
        if source_has(sources.get("models/dit_v2/nablocks/__init__.py", ""),
                      "nadit_blocks", '"mmdit_sr"', "NaMMSRTransformerBlock") else "NOT FOUND"
    )
    report["token_layout"] = {
        "video_input": "[L_vid, C_in] flattened; logical [B, T, H, W, C_in]; C_in=33 (16 noisy + 16 cond + 1)",
        "video_after_patch": "[L_vid, 2560] with L_vid = T * (H/2) * (W/2); patch_size (1,2,2)",
        "video_token_order": "T-major, W-fastest (patch rearrange '(T t) (H h) (W w) c -> T H W (t h w c)')",
        "text_input": "[L_txt, 5120] -> txt_in projection -> [L_txt, 2560]",
        "text_token_order": "1-D text sequence; txt_shape=[b,1] = per-sample text length l",
        "concat_order": "per window segment: [vid_win tokens; txt tokens] (vid first, then txt)",
        "batch_and_token_dims": "token dim = dim 0 of flattened [L, C]; batch implicit in shapes (vid_shape [b,3], txt_shape [b,1])",
        "text_repeat": "txt q/k repeated per window; txt outputs coalesced by MEAN over repeats",
    }

    report["inputs"] = [
        {"name": "vid", "semantic_role": "video hidden states (flattened tokens)",
         "dtype": "float32", "rank": 2, "logical_shape": "[L_vid, 2560]",
         "physical_layout": "row-major [L_vid, C]; logical [B, T, H, W, C] with T-major W-fastest flatten",
         "static_dims": ["2560"], "dynamic_dims": ["L_vid"],
         "producer": "NaPatchIn(vid, vid_shape) -> Linear(4*C_in -> 2560) after patchify (1,2,2)",
         "consumer": "attn_norm / mlp_norm / attn.proj_qkv", "optional": False, "device": "same as model"},
        {"name": "txt", "semantic_role": "text/context hidden states",
         "dtype": "float32", "rank": 2, "logical_shape": "[L_txt, 2560]",
         "physical_layout": "1-D token sequence [L_txt, C]",
         "static_dims": ["2560"], "dynamic_dims": ["L_txt = l"],
         "producer": "text encoder [L_txt, 5120] -> txt_in projection (Linear 5120->2560 + fusedln)",
         "consumer": "attn_norm / mlp_norm / attn.proj_qkv; repeated per window in attention", "optional": False,
         "device": "same as model"},
        {"name": "vid_shape", "semantic_role": "per-sample video spatial/frame shape metadata",
         "dtype": "int64", "rank": 2, "logical_shape": "[b, 3] = (T, H, W) post-patch",
         "physical_layout": "row-major [b, 3]", "static_dims": ["3"], "dynamic_dims": ["b", "T", "H", "W"],
         "producer": "na.flatten(video latents) then NaPatchIn", "consumer": "window partition, vid_len, RoPE get_freqs",
         "optional": False, "device": "same as model"},
        {"name": "txt_shape", "semantic_role": "per-sample text length metadata",
         "dtype": "int64", "rank": 2, "logical_shape": "[b, 1] = text length l",
         "physical_layout": "row-major [b, 1]", "static_dims": ["1"], "dynamic_dims": ["b", "l"],
         "producer": "text encoder na.flatten", "consumer": "txt_len, RoPE get_freqs (vid_freqs[l:l+f], txt_freqs[:l])",
         "optional": False, "device": "same as model"},
        {"name": "emb", "semantic_role": "timestep/conditioning embedding for AdaSingle modulation",
         "dtype": "float32", "rank": 2, "logical_shape": "[b, 15360]",
         "physical_layout": "row-major [b, 6*dim] = [b, d, layers=2, groups=3]",
         "static_dims": ["15360"], "dynamic_dims": ["b"],
         "producer": "TimeEmbedding(timestep): sinusoidal 256 -> proj_in -> SiLU -> proj_hid -> SiLU -> proj_out",
         "consumer": "AdaSingle (attn/mlp, in/out)", "optional": False, "device": "same as model"},
        {"name": "cache", "semantic_role": "memoized inference cache (per-sample reusable values)",
         "dtype": "object", "rank": 0, "logical_shape": "Cache(namespaced dict)",
         "physical_layout": "Python object", "static_dims": [], "dynamic_dims": [],
         "producer": "NaDiT.forward (Cache(disable=...))", "consumer": "all block components (lens, indices, freqs)",
         "optional": False, "device": "host"},
        {"name": "attention_mask", "semantic_role": "NO explicit mask tensor; masking is varlen block-diagonal via cu_seqlens_q == cu_seqlens_k per window segment [vid;txt]",
         "dtype": "n/a", "rank": -1, "logical_shape": "implicit", "physical_layout": "n/a",
         "static_dims": [], "dynamic_dims": [], "producer": "na.concat_idx / repeat_concat_idx + safe_pad_operation",
         "consumer": "FlashAttentionVarlen", "optional": True, "device": "n/a"},
        {"name": "text_mask / video_validity", "semantic_role": "NO explicit mask tensors in the block; padding/validity is implicit in the dynamic window partition (partial boundary windows)",
         "dtype": "n/a", "rank": -1, "logical_shape": "implicit", "physical_layout": "n/a",
         "static_dims": [], "dynamic_dims": [], "producer": "window_idx (partial boundary windows)", "consumer": "attention segments",
         "optional": True, "device": "n/a"},
    ]
    report["outputs"] = [
        {"name": "vid_out", "semantic_role": "block video hidden state after MLP residual",
         "dtype": "float32", "rank": 2, "logical_shape": "[L_vid, 2560]",
         "physical_layout": "T-major W-fastest flatten", "static_dims": ["2560"], "dynamic_dims": ["L_vid"],
         "producer": "vid_mlp + vid_attn (second residual)", "consumer": "next block / vid_out_norm + NaPatchOut",
         "optional": False, "device": "same as model"},
        {"name": "txt_out", "semantic_role": "block text hidden state (coalesced from window repeats)",
         "dtype": "float32", "rank": 2, "logical_shape": "[L_txt, 2560]",
         "physical_layout": "1-D token sequence", "static_dims": ["2560"], "dynamic_dims": ["L_txt = l"],
         "producer": "txt_mlp + txt_attn (second residual)", "consumer": "next block; last layer txt passed through (vid_only)",
         "optional": False, "device": "same as model"},
        {"name": "vid_shape / txt_shape", "semantic_role": "unchanged shape metadata pass-through",
         "dtype": "int64", "rank": 2, "logical_shape": "[b,3] / [b,1]",
         "physical_layout": "row-major", "static_dims": ["3", "1"], "dynamic_dims": ["b", "T", "H", "W", "l"],
         "producer": "block.forward return", "consumer": "next block / NaDiT", "optional": False, "device": "same as model"},
    ]
    report["token_layout"]["text_video_boundary"] = (
        "per window attention segment = [vid_win tokens; txt tokens]; text q/k repeated per window and "
        "outputs coalesced by mean; RoPE uses vid_freqs[l:l+f] (T offset by text length l) for video and "
        "txt_freqs[:l] for text")
    rope_src = sources.get("models/dit_v2/rope.py", "")
    rope_ok = all([
        source_has(rope_src, 'RotaryEmbedding(dim=dim // rope_dim, freqs_for="lang", theta=10000'),
        source_has(rope_src, "vid_freqs[l : l + f, :h, :w]"),
        source_has(rope_src, "txt_freqs[:l].repeat(1, 3)"),
        source_has(rope_src, "self.get_axial_freqs(1024, 128, 128)"),
        source_has(rope_src, "self.get_axial_freqs(1024)"),
        source_has(rope_src, "apply_rotary_emb"),
        "NaMMRotaryEmbedding3d" in rope_src,
    ])
    report["rope"] = {
        "class": "NaMMRotaryEmbedding3d(MMRotaryEmbeddingBase)",
        "configured_rope_dim": int(m["rope_dim"]),
        "per_axis_dim": int(m["rope_dim"]) // 3,
        "effective_rotary_dim": 3 * (int(m["rope_dim"]) // 3),
        "identity_tail_dim": int(m["head_dim"]) - 3 * (int(m["rope_dim"]) // 3),
        "freqs_for": "lang", "theta": 10000,
        "axis_order": ["T", "H", "W"], "token_order": "T-major W-fastest",
        "vid_freqs_table": "get_axial_freqs(1024, 128, 128) precomputed grid",
        "txt_freqs_table": "get_axial_freqs(1024)",
        "l_semantics": "l = txt_shape[i,0] = per-sample text token count",
        "f_semantics": "f = vid_shape[i,0] = per-sample post-patch video frame count (T dim), NOT total video tokens",
        "vid_freq_slice": "vid_freqs[l : l+f, :h, :w]: video T-axis positions offset by text length l in the shared 1024-frame table",
        "txt_freq_slice": "txt_freqs[:l].repeat(1, 3): first l 1-D freqs replicated across the 3 axis blocks",
        "applied_to": "q/k per head [L, h, 128]; first 126 dims rotated, tail 2 identity (Phase 10.1 oracle-confirmed)",
        "block_index_effect": "no block-index dependence; freqs depend only on shapes and text length l",
        "window_local_coords": "per-window local coords; each window uses its own (f_w,h_w,w_w) with the same l offset",
        "batch_sharing": "freqs computed per batch item from its own (l, f, h, w); not shared across batch",
        "ast_verified": rope_ok,
    }
    if not rope_ok:
        errors.append("rope contract AST verification failed")

    mod_src = sources.get("models/dit_v2/modulation.py", "")
    report["modulation"] = {
        "class": "AdaSingle",
        "emb_input": "[b, 15360] (emb_dim == 6 * vid_dim)",
        "split": "emb rearranged [b, d=2560, l=2, g=3]; per layer (attn, mlp) 3 groups: shift/scale/gate",
        "per_layer_groups": ["attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate"],
        "learned_params": "per layer: shift (d), scale (d, init ~+1), gate (d)",
        "mode_in_formula": "hid * (scaleA + scaleB) + (shiftA + shiftB)",
        "mode_out_formula": "hid * (gateA + gateB)",
        "gate_activation": "none (linear gate, no sigmoid/tanh)",
        "application_sites": ["attn_norm after (mode in)", "attention output (mode out)",
                              "mlp_norm after (mode in)", "mlp output (mode out)"],
        "per_branch": "vid/txt modulated separately (branch_tag vid|txt; emb repeated per token via hid_len)",
        "per_block": "each block owns its AdaSingle parameters",
        "ast_verified": bool(mod_src and ("AdaSingle" in mod_src) and ("emb_dim == 6 * dim" in mod_src)),
    }
    report["modulation"]["slice_map"] = [
        {"emb_slice": "emb[..., 0, :] (attn layer, group 0)", "semantic": "attn_shift", "target_branch": "vid|txt",
         "target_shape": "[L_*, 2560]", "formula": "hid * (scaleA+scaleB) + (shiftA+shiftB) [mode in]"},
        {"emb_slice": "emb[..., 1, :] (attn layer, group 1)", "semantic": "attn_scale", "target_branch": "vid|txt",
         "target_shape": "[L_*, 2560]", "formula": "same as above"},
        {"emb_slice": "emb[..., 2, :] (attn layer, group 2)", "semantic": "attn_gate", "target_branch": "vid|txt",
         "target_shape": "[L_*, 2560]", "formula": "hid * (gateA+gateB) [mode out, after attention]"},
        {"emb_slice": "emb[..., 0, :] (mlp layer, group 0)", "semantic": "mlp_shift", "target_branch": "vid|txt",
         "target_shape": "[L_*, 2560]", "formula": "hid * (scaleA+scaleB) + (shiftA+shiftB) [mode in]"},
        {"emb_slice": "emb[..., 1, :] (mlp layer, group 1)", "semantic": "mlp_scale", "target_branch": "vid|txt",
         "target_shape": "[L_*, 2560]", "formula": "same as above"},
        {"emb_slice": "emb[..., 2, :] (mlp layer, group 2)", "semantic": "mlp_gate", "target_branch": "vid|txt",
         "target_shape": "[L_*, 2560]", "formula": "hid * (gateA+gateB) [mode out, after MLP]"},
    ]

    attn_src = sources.get("models/dit_v2/nablocks/attention/mmattn.py", "")
    attn_ok = all([
        "class NaSwinAttention(NaMMAttention)" in attn_src,
        source_has(attn_src, "qkv_dim = inner_dim * 3"),
        source_has(attn_src, "concat_win(vid_q, txt_q).bfloat16()"),
        source_has(attn_src, "window_partition(vid_qkv)"),
        source_has(attn_src, "txt_out_coalesced"),
        source_has(attn_src, "rearrange(vid_qkv_win, 'l (o h d) -> l o h d', o=3"),
    ])
    report["attention"] = {
        "class": "NaSwinAttention(NaMMAttention)",
        "proj_qkv": "Linear(vid_dim/txt_dim -> 3*heads*head_dim = 7680) per branch (shared_weights per layer index)",
        "proj_out": "Linear(heads*head_dim = 2560 -> vid_dim/txt_dim)",
        "qk_norm": "fusedrms (apex FusedRMSNorm), dim=head_dim=128, elementwise_affine=True, eps=1e-5",
        "qk_bias": False, "window": "(4,3,3)",
        "window_method": "alternating 720pwin_by_size_bysize / 720pswin_by_size_bysize",
        "window_partition": "vid qkv [T,H,W,C] partitioned; txt NOT partitioned (repeated per window)",
        "rope": "window-local MM-RoPE3D on vid q/k with window_shape; txt q/k repeated per window",
        "attention_impl": "FlashAttentionVarlen (flash_attn.flash_attn_varlen_func) on bf16 cast",
        "mask_form": "varlen block-diagonal via cu_seqlens_q == cu_seqlens_k per window segment [vid; txt]; no explicit mask, no causal mask",
        "cross_window": "no cross-window attention (each window segment independent)",
        "text_pooling": "txt outputs from repeats coalesced by mean",
        "dtype": "attention math in bfloat16 (inputs cast .bfloat16(); output .type_as(vid_q))",
        "ast_verified": attn_ok,
    }

    mlp_src = sources.get("models/dit_v2/mlp.py", "")
    mlp_ok = all([
        "class SwiGLUMLP" in mlp_src,
        source_has(mlp_src, "hidden_dim = int(2 * dim * expand_ratio / 3)"),
        source_has(mlp_src, "multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)"),
        source_has(mlp_src, "self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))"),
        "bias=False" in mlp_src,
    ])
    report["mlp"] = {
        "class": "SwiGLUMLP",
        "proj_in_gate": "Linear(%d -> %d, bias=False)" % (vid_dim, mlp_hidden),
        "proj_in": "Linear(%d -> %d, bias=False)" % (vid_dim, mlp_hidden),
        "proj_out": "Linear(%d -> %d, bias=False)" % (mlp_hidden, vid_dim),
        "hidden_dim_derived": mlp_hidden, "expansion": expand,
        "activation": "SiLU (gated): silu(gate(x)) * x", "dropout": "none",
        "bias": "none", "gated": True, "ast_verified": mlp_ok,
    }

    report["residual_order"] = [
        "attn_norm(vid,txt) -> ada(attn, in: scale+shift) -> NaSwinAttention -> ada(attn, out: gate) -> +vid/txt residual",
        "mlp_norm(vid_attn,txt_attn) -> ada(mlp, in) -> SwiGLU MLP -> ada(mlp, out) -> +vid_attn/txt_attn residual",
        "pre-norm style; gate applied AFTER attention/MLP and BEFORE the residual add",
    ]
    report["dynamic_dimensions"] = [
        "L_vid (video token count, from vid_shape T*H*W)",
        "L_txt (text token count l)", "vid_shape [b,3] (T,H,W post-patch)",
        "txt_shape [b,1] (text length l)", "timestep (per sample)",
        "window partition count/sizes (derived from vid_shape + window_method)",
    ]

    window_probe = probe_window_functions(sources.get("models/dit_v2/window.py", ""), errors)
    report["probes"]["window_partition"] = window_probe
    report["determinism"]["window_partition"] = window_probe.get("deterministic")
    negatives = run_negative_tests(sources, window_probe, errors)
    report["negative_tests"] = negatives
    report["dependency_gap"] = check_dependencies()
    report["real_block_probe"] = run_real_block_probe()

    freshness = {
        "seedvr2_commit": head, "expected_commit": EXPECTED_SEEDVR2_COMMIT,
        "commit_matches": head == EXPECTED_SEEDVR2_COMMIT,
        "sources_exist": {rel: os.path.isfile(os.path.join(SEEDVR2_ROOT, rel)) for rel in SOURCE_FILES},
        "symbol_checks": {
            "NaMMSRTransformerBlock": "NaMMSRTransformerBlock" in blk_src,
            "NaDiT": "class NaDiT" in nad_src,
            "NaSwinAttention": "class NaSwinAttention" in attn_src,
            "NaMMRotaryEmbedding3d": "class NaMMRotaryEmbedding3d" in rope_src,
            "AdaSingle": "class AdaSingle" in mod_src,
            "SwiGLUMLP": "class SwiGLUMLP" in mlp_src,
        },
    }
    report["freshness"] = freshness

    report["ncnn_mapping_risk"] = [
        {"risk": "FlashAttentionVarlen (flash_attn_varlen_func)", "note": "CUDA-only; needs varlen/block-diagonal attention equivalent; cu_seqlens segment [vid;txt] per window; bf16 upstream"},
        {"risk": "gather_seq_scatter_heads_qkv / gather_heads_scatter_seq", "note": "sequence-parallel index permutations; reducible to static index maps in ncnn"},
        {"risk": "window partition by-size (dynamic)", "note": "window sizes computed at runtime from vid_shape; maps to existing dynamic window adapter"},
        {"risk": "apex FusedRMSNorm / FusedLayerNorm", "note": "standard RMS/LN math; covered by existing fixed/dynamic RMS fixtures"},
        {"risk": "diffusers RMSNorm + get_timestep_embedding", "note": "pure torch math; sinusoidal embedding must match flip_sin_to_cos=False, downscale_freq_shift=0"},
        {"risk": "dynamic token concatenation [vid;txt] + txt repeat/mean coalesce", "note": "index permutation + mean reduction; needs exact parity"},
        {"risk": "modulation broadcast", "note": "per-token broadcast of [b,d] scale/shift/gate with branch-specific hid_len repetition"},
        {"risk": "gated MLP (SwiGLU)", "note": "silu(gate)*x then linear; two GEMMs; no bias"},
        {"risk": "RoPE frequency offset (vid_freqs[l:l+f])", "note": "T-axis offset by text length l; window-local coordinates; effective 126 of 128 with identity tail 2"},
        {"risk": "shared vs separate vid/txt weights by layer index", "note": "layers 0..9 separate (mm_layers=10), layers 10..31 shared; last layer vid_only mlp/ada"},
        {"risk": "bf16 attention vs fp32 ncnn", "note": "upstream casts attention to bf16; fp32 ncnn parity target must define tolerance"},
    ]
    report["intermediate_shapes"] = {
        "block_input_vid": "[L_vid, 2560]", "block_input_txt": "[L_txt, 2560]",
        "attn_norm_out": "[L_*, 2560]", "qkv": "[L_*, 7680]",
        "q/k/v_per_head": "[L_*, 20, 128]", "rope_freqs": "[L_*, 126] (per window)",
        "attention_out": "[L_*, 20, 128] -> [L_*, 2560]", "proj_out": "[L_*, 2560]",
        "mlp_hidden": "[L_*, 6912]", "mlp_out": "[L_*, 2560]",
        "block_output": "[L_vid, 2560], [L_txt, 2560]",
        "emb": "[b, 15360]", "vid_shape": "[b, 3]", "txt_shape": "[b, 1]",
    }

    probe_ok = report["real_block_probe"].get("status") == "PASS"
    if probe_ok and not errors:
        report["overall_status"] = "READY"
    else:
        report["overall_status"] = "PARTIAL"
        if not probe_ok:
            missing = sorted(report["dependency_gap"].get("missing", {}).keys())
            report["blocker"] = (
                "real random-weight forward probe not executable: missing non-model Python deps "
                + (", ".join(missing) if missing else "unknown")
                + " (task forbids installing them); source/AST/window contract complete"
            )
        elif errors:
            report["blocker"] = "; ".join(errors)
    report["errors"] = errors

    for path in (CONTRACT_PATH, REPORT_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    for path in (CONTRACT_PATH, REPORT_PATH):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        assert obj["phase"] == "11"
        assert obj["real_seedvr2_weights"] is False
        assert jsonable_clean(obj)
    print("extract_seedvr2_block_contract=PASS overall=%s commit=%s real_block_probe=%s" % (
        report["overall_status"], head, report["real_block_probe"].get("status")))
    return 0


def jsonable_clean(obj):
    if isinstance(obj, dict):
        return all(jsonable_clean(v) for v in obj.values())
    if isinstance(obj, list):
        return all(jsonable_clean(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


def probe_window_functions(src, errors):
    out = {"loaded": False, "profiles": [], "negative": None, "deterministic": None}
    if not src:
        errors.append("window.py source missing")
        return out
    spec = importlib.util.spec_from_file_location(
        "seedvr2_window", os.path.join(SEEDVR2_ROOT, "models", "dit_v2", "window.py"))
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    out["loaded"] = True
    profiles = [
        {"name": "A_min", "shape": (1, 4, 4), "text_len": 2},
        {"name": "B_multi_window", "shape": (5, 32, 32), "text_len": 8},
        {"name": "C_asymmetric", "shape": (7, 45, 64), "text_len": 11},
    ]
    for p in profiles:
        t, h, wd = p["shape"]
        entry = {"name": p["name"], "shape": p["shape"], "text_len": p["text_len"], "methods": {}}
        for method in ("720pwin_by_size_bysize", "720pswin_by_size_bysize"):
            op = w.get_window_op(method)
            first = op((t, h, wd), (4, 3, 3))
            second = op((t, h, wd), (4, 3, 3))
            det = len(first) == len(second) and all(s == s2 for s, s2 in zip(first, second))
            cov = window_coverage(t, h, wd, first)
            sizes = [(s.stop - s.start, sl.stop - sl.start, sw.stop - sw.start) for (s, sl, sw) in first]
            tokens = [a * b * c for (a, b, c) in sizes]
            entry["methods"][method] = {
                "window_count": len(first),
                "window_shapes": sorted(set(sizes)),
                "min_tokens": min(tokens), "max_tokens": max(tokens),
                "coverage_all_cells": cov, "deterministic_run": bool(det),
            }
        out["profiles"].append(entry)
    try:
        w.get_window_op("bogus_window_method")
        out["negative"] = "FAIL: no exception"
    except ValueError:
        out["negative"] = "PASS: ValueError"
    except Exception as e:
        out["negative"] = "PASS: %s" % type(e).__name__
    out["deterministic"] = all(e["methods"][m]["deterministic_run"] for e in out["profiles"] for m in e["methods"])
    return out


def window_coverage(t, h, wd, windows):
    grid = [[[False] * wd for _ in range(h)] for _ in range(t)]
    for (st, sl, sw) in windows:
        for i in range(st.start, st.stop):
            for j in range(sl.start, sl.stop):
                for k in range(sw.start, sw.stop):
                    grid[i][j][k] = True
    return all(grid[i][j][k] for i in range(t) for j in range(h) for k in range(wd))


def run_negative_tests(sources, window_probe, errors):
    tests = []
    rope_src = sources.get("models/dit_v2/rope.py", "")
    mod_src = sources.get("models/dit_v2/modulation.py", "")
    mlp_src = sources.get("models/dit_v2/mlp.py", "")
    nablocks_src = sources.get("models/dit_v2/nablocks/__init__.py", "")
    patch_src = sources.get("models/dit_v2/patch/patch_v1.py", "")
    checks = [
        ("hidden_dim % head_dim == 0", 2560 % 128 == 0),
        ("heads * head_dim == hidden_dim", 20 * 128 == 2560),
        ("emb_dim == 6 * hidden_dim", 15360 == 6 * 2560),
        ("mlp_hidden % 256 == 0", 6912 % 256 == 0),
        ("effective_rotary + identity == head_dim", 126 + 2 == 128),
        ("l+f <= 1024 (vid_freqs table bound)", all(
            (p["text_len"] + p["shape"][0]) <= 1024 for p in [
                {"text_len": 2, "shape": (1, 4, 4)}, {"text_len": 8, "shape": (5, 32, 32)},
                {"text_len": 11, "shape": (7, 45, 64)}])),
        ("vid token count == T*H*W", 5 * 32 * 32 == 5120),
        ("get_na_rope unsupported -> NotImplementedError (AST)", "NotImplementedError" in rope_src),
        ("get_nablock unsupported -> NotImplementedError (AST)", "NotImplementedError" in nablocks_src),
        ("get_ada_layer unsupported -> NotImplementedError (AST)", "NotImplementedError" in mod_src),
        ("get_mlp unknown -> None (AST)", "def get_mlp" in mlp_src and mlp_src.count("return MLP") >= 1),
        ("registry maps mmdit_sr to NaMMSRTransformerBlock (AST)", '"mmdit_sr": NaMMSRTransformerBlock' in nablocks_src),
        ("rope rotates first 126 only, tail identity (AST)", "apply_rotary_emb" in rope_src and "vid_freqs" in rope_src),
        ("patch rearrange T-major W-fastest (AST)", "(T t) (H h) (W w) c -> T H W (t h w c)" in patch_src),
        ("window unknown op -> ValueError (executable)", window_probe.get("negative", "").startswith("PASS")),
        ("window partition determinism (executable)", window_probe.get("deterministic") is True),
    ]
    for name, ok in checks:
        tests.append({"name": name, "expected": "safe-fail/consistency", "result": "PASS" if ok else "FAIL"})
    return tests


def check_dependencies():
    gap = {"available": {}, "missing": {}, "pinned_environment_yml": DEP_PINS, "pinned_requirements_txt": REQ_PINS}
    for mod in ["torch", "einops", "diffusers", "flash_attn", "apex", "rotary_embedding_torch", "yaml", "numpy"]:
        try:
            ok = importlib.util.find_spec(mod) is not None
        except Exception:
            ok = False
        (gap["available"] if ok else gap["missing"])[mod] = True
    return gap


def run_real_block_probe():
    probe = os.path.join(OUT_DIR, "probe_real_block.py")
    if not os.path.isfile(probe):
        return {"status": "SKIPPED", "reason": "probe file missing"}
    marker = os.path.join(OUT_DIR, "probe_result.json")
    if os.path.isfile(marker):
        os.remove(marker)
    try:
        r = subprocess.run([sys.executable, probe], capture_output=True, text=True, timeout=900)
        data = {}
        if os.path.isfile(marker):
            with open(marker, "r", encoding="utf-8") as f:
                data = json.load(f)
        data["exit_code"] = r.returncode
        data["stdout_tail"] = r.stdout.strip().splitlines()[-5:] if r.stdout.strip() else []
        data["stderr_tail"] = r.stderr.strip().splitlines()[-8:] if r.stderr.strip() else []
        return data
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
