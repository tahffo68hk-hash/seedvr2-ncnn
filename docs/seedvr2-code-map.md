# SeedVR2 Code Map

## Entry point

- `projects/inference_seedvr2_3b.py` and `projects/inference_seedvr2_7b.py` configure distributed CUDA inference.
- `projects/video_diffusion_sr/infer.py:VideoDiffusionInfer` owns model setup, VAE encode/decode, diffusion sampling, and the DiT call.

## Pipeline

Configuration (`configs_3b/main.yaml` or `configs_7b/main.yaml`) creates the DiT and VAE, loads checkpoints, encodes input/blur conditions with `vae_encode`, runs `inference`, then decodes latents with `vae_decode`. The inference path concatenates noisy latent tokens and condition tokens along the channel dimension and calls `NaDiT`.

## Model

The 3B configuration selects `models.dit_v2.nadit.NaDiT`: 32 layers, video width 2560, 20 heads, head dimension 128, patch size `[1,2,2]`, 33 input channels and 16 output channels. Ten layers are configured as multimodal attention blocks.

## VAE

The VAE configuration inherits `models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml`. The implementation is `models/video_vae_v3/modules/video_vae.py`, with `VideoAutoencoderKL`, causal encoder/decoder paths, 3D inflated layers, temporal slicing and memory-state handling. Attention-capable 3D variants are in `attn_video_vae.py`.

## Transformer

- `models/dit_v2/nadit.py:NaDiT`
- `models/dit_v2/na.py`: flattened variable-resolution sequence packing and window partition/reverse helpers
- `models/dit_v2/nablocks/mmsr_block.py`: multimodal SR blocks
- `models/dit_v2/nablocks/attention/mmattn.py`: multimodal and window attention

## Attention

`models/dit_v2/attention.py` defines `TorchAttention` using PyTorch scaled dot-product attention and `FlashAttentionVarlen` using `flash_attn_varlen_func`. The configured v2 window path uses the latter.

## Adaptive Window Attention

`models/dit_v2/window.py` implements `make_720Pwindows_bysize` and `make_shifted_720Pwindows_bysize`. The configured `(4,3,3)` values are target window counts in temporal/height/width, not fixed pixel window sizes. The implementation computes window sizes from the current latent resolution and uses clipped slices without explicit padding.

## Dependencies

`requirements.txt` pins Torch 2.3.0, diffusers 0.29.1, einops 0.7.0, rotary-embedding-torch 0.5.3, transformers 4.38.2, torchvision 0.18.0, OpenCV and OmegaConf. README setup additionally references FlashAttention and Apex. These were inspected only; no SeedVR2 environment was installed.
