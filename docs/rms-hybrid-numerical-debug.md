# Phase 8.1 RMS hybrid numerical debug

## Root cause

The first failing stage in the original artifact was QKV projection, before
RMSNorm: pnnx default `fp16=1` compressed model weights. This matched the
already documented fixed-attention behavior. It was not a reduction-axis,
epsilon, RoPE, mask, host-merge, or softmax fault.

| head0 stage | old fp16=1 max error | corrected fp16=0 max error |
|---|---:|---:|
| q/k/v raw projection | 9.59277e-4 | 2.38419e-7 |
| Q/K RMSNorm | 8.64774e-4 | 4.76837e-7 |
| RoPE | 8.64774e-4 | 4.76837e-7 |
| QK logits (unscaled) | 3.47042e-3 | 2.14577e-6 |
| scaled + masked logits | 1.22696e-3 | 7.45058e-7 |
| probabilities | 6.51777e-5 | 5.21541e-8 |
| AV | 2.16469e-4 | 1.34110e-7 |

Corrected head0 component measurements are: q_raw `2.38419e-7`, k_raw
`2.38419e-7`, q_rms `4.17233e-7`, k_rms `4.76837e-7`, q_rope `4.17233e-7`,
and k_rope `4.76837e-7`. Unscaled logits range from `-7.46571` to `7.37257`;
softmax did not amplify the corrected errors.

The final artifact uses five FP32 ncnn models. It needs no host RMSNorm stage
and no custom layer. Direct SeedVR2 fused-Apex parity remains outside this
fixture because Apex/diffusers are not installed.
