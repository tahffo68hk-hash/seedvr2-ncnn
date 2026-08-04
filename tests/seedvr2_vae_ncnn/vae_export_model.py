#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: graph-equivalent export wrappers for the REAL SeedVR2 VAE.

The official `InflatedCausalConv3d` / attention use torch.tile + SDPA which do
not map to ncnn runtime layers.  These wrappers reproduce the official
whole-tensor forward (memory_state=DISABLED) with ncnn-exportable ops while
keeping the exact same weights and arithmetic order:
  * extend_head -> repeated cat of the first-frame slice (no torch.tile)
  * remove_head -> cat of [t, 2:] slices
  * attention   -> explicit q/k/v linear, bmm, softmax, bmm, proj
  * group norm / silu / conv3d / pad -> native ncnn ops

Builders copy weights out of the official modules so the export never holds
the full official model graph.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExportCausalConv3d(nn.Module):
    """Graph-equivalent of InflatedCausalConv3d (whole-tensor, DISABLED)."""

    def __init__(self, conv):
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = None
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.detach().clone())
        self.stride = tuple(int(s) for s in conv.stride)
        self.padding = (0, int(conv.padding[1]), int(conv.padding[2]))
        self.temporal_padding = int(conv.temporal_padding)

    def forward(self, x):
        t = self.temporal_padding * 2
        if t > 0:
            first = x[:, :, :1]
            x = torch.cat([first] * t + [x], dim=2)
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class ExportGroupNorm(nn.Module):
    def __init__(self, gn):
        super().__init__()
        self.gn = nn.GroupNorm(gn.num_groups, gn.num_channels, eps=gn.eps, affine=True)
        with torch.no_grad():
            self.gn.weight.copy_(gn.weight)
            if gn.bias is not None:
                self.gn.bias.copy_(gn.bias)
            else:
                self.gn.bias.zero_()

    def forward(self, x):
        if x.ndim == 5:
            # GNAlt2: fold the time axis into the channel axis so the official
            # causal `rearrange("b c t h w -> (b t) c h w")` maps to ncnn ops
            # without batch inference (NCNN_BATCH).  GroupNorm(G*T, C*T) with
            # weight/bias repeated over T is mathematically identical to
            # GroupNorm(G, C) applied per frame (verified diff 0.0 in torch).
            b, c, t, h, w = x.shape
            if getattr(self, "_exp_t", None) != t or getattr(self, "_expanded", None) is None:
                expanded = nn.GroupNorm(self.gn.num_groups * t, self.gn.num_channels * t,
                                        eps=self.gn.eps, affine=True)
                with torch.no_grad():
                    expanded.weight.copy_(self.gn.weight.repeat(t))
                    if self.gn.bias is not None:
                        expanded.bias.copy_(self.gn.bias.repeat(t))
                    else:
                        expanded.bias.zero_()
                self._expanded = expanded
                self._exp_t = t
            x = x.permute(0, 2, 1, 3, 4).reshape(b, t * c, h, w)
            x = self._expanded(x)
            return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return self.gn(x)


class ExportResnet(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.norm1 = ExportGroupNorm(resnet.norm1)
        self.conv1 = ExportCausalConv3d(resnet.conv1)
        self.norm2 = ExportGroupNorm(resnet.norm2)
        self.conv2 = ExportCausalConv3d(resnet.conv2)
        self.shortcut = None
        if resnet.conv_shortcut is not None:
            self.shortcut = ExportCausalConv3d(resnet.conv_shortcut)
        self.output_scale_factor = float(resnet.output_scale_factor)

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return (x + h) / self.output_scale_factor


class ExportDownsample(nn.Module):
    def __init__(self, down):
        super().__init__()
        self.conv = ExportCausalConv3d(down.conv)
        self.pad_spatial = bool(down.use_conv and down.padding == 0 and down.spatial_down)

    def forward(self, x):
        if self.pad_spatial:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class ExportUpsample(nn.Module):
    def __init__(self, up):
        super().__init__()
        uc = up.upscale_conv
        self.conv = ExportCausalConv3d(up.conv)
        self.temporal_up = bool(up.temporal_up)
        self.xr = int(up.spatial_ratio)
        self.yr = int(up.spatial_ratio)
        self.z = int(up.temporal_ratio)
        self.norm = None
        if getattr(up, "norm", None) is not None:
            self.norm = ExportGroupNorm(up.norm)
        # The official upsampler = 1x1x1 conv (out = x*y*z*C) followed by the
        # pixel-shuffle rearrange "b (x y z c) f h w -> b c (f z) (h x) (w y)".
        # A 1x1 conv + channel-to-space is exactly a ConvTranspose3d with
        # stride=kernel=(z,x,y) and the conv weight permuted into the kernel
        # (verified bit-exact in torch).  The conv bias is carried by a
        # constant-1 extra input channel so it stays position-dependent.
        C = uc.in_channels
        C8 = uc.out_channels
        assert C8 == C * self.xr * self.yr * self.z, (C, C8, self.xr, self.yr, self.z)
        W = torch.zeros(C + 1, C, self.z, self.xr, self.yr)
        for kt in range(self.z):
            for kh in range(self.xr):
                for kw in range(self.yr):
                    m_base = ((kh * self.yr + kw) * self.z + kt) * C
                    W[:C, :, kt, kh, kw] = uc.weight[m_base:m_base + C].squeeze().t()
                    if uc.bias is not None:
                        W[C, :, kt, kh, kw] = uc.bias[m_base:m_base + C]
        self.shuffle_deconv = nn.ConvTranspose3d(
            C + 1, C, (self.z, self.xr, self.yr), stride=(self.z, self.xr, self.yr), bias=False)
        with torch.no_grad():
            self.shuffle_deconv.weight.copy_(W)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        B, C, T, H, W = x.shape
        ones = torch.ones(B, 1, T, H, W, dtype=x.dtype, device=x.device)
        x = self.shuffle_deconv(torch.cat([x, ones], dim=1))
        if self.temporal_up:
            x = torch.cat([x[:, :, :1], x[:, :, 2:]], dim=2)
        return self.conv(x)


class ExportAttention(nn.Module):
    """diffusers Attention with AttnProcessor2_0 math, explicit ops."""

    def __init__(self, attn):
        super().__init__()
        self.to_q = nn.Linear(attn.to_q.in_features, attn.to_q.out_features, bias=attn.to_q.bias is not None)
        self.to_k = nn.Linear(attn.to_k.in_features, attn.to_k.out_features, bias=attn.to_k.bias is not None)
        self.to_v = nn.Linear(attn.to_v.in_features, attn.to_v.out_features, bias=attn.to_v.bias is not None)
        self.to_out0 = nn.Linear(attn.to_out[0].in_features, attn.to_out[0].out_features,
                                 bias=attn.to_out[0].bias is not None)
        with torch.no_grad():
            self.to_q.weight.copy_(attn.to_q.weight); self.to_q.bias.copy_(attn.to_q.bias)
            self.to_k.weight.copy_(attn.to_k.weight); self.to_k.bias.copy_(attn.to_k.bias)
            self.to_v.weight.copy_(attn.to_v.weight); self.to_v.bias.copy_(attn.to_v.bias)
            self.to_out0.weight.copy_(attn.to_out[0].weight); self.to_out0.bias.copy_(attn.to_out[0].bias)
        self.heads = int(attn.heads)
        self.scale = float(attn.scale)
        self.residual = bool(attn.residual_connection)
        self.rescale_output_factor = float(attn.rescale_output_factor)
        self.group_norm = None
        if getattr(attn, "group_norm", None) is not None:
            self.group_norm = ExportGroupNorm(attn.group_norm)

    def forward(self, h):
        # Per-frame spatial attention, unrolled over the time axis so pnnx
        # never emits a batched (NCNN_BATCH) graph.  ncnn GroupNorm/Permute/
        # MultiHeadAttention/Softmax do not support the batch dimension, so the
        # T frames are processed one at a time and stacked back into BCTHW.
        # Mathematically identical to the whole-tensor path (verified in torch).
        residual = h
        B, C, T, H, W = h.shape
        hd = self.to_q.out_features // self.heads
        outs = []
        for t in range(T):
            x = h[:, :, t].reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
            if self.group_norm is not None:
                x = self.group_norm(x.transpose(1, 2)).transpose(1, 2)
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
            q = q.reshape(B, -1, self.heads, hd).permute(0, 2, 1, 3)
            k = k.reshape(B, -1, self.heads, hd).permute(0, 2, 1, 3)
            v = v.reshape(B, -1, self.heads, hd).permute(0, 2, 1, 3)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.permute(0, 2, 1, 3).reshape(B, -1, self.heads * hd)
            out = self.to_out0(out)
            outs.append(out.permute(0, 2, 1).reshape(B, C, H, W))
        out = torch.stack(outs, dim=2)  # (B, C, T, H, W)
        if self.residual:
            out = out + residual
        return out / self.rescale_output_factor


class ExportDownBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.resnets = nn.ModuleList([ExportResnet(r) for r in block.resnets])
        self.down = None
        if block.downsamplers is not None:
            self.down = ExportDownsample(block.downsamplers[0])

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.down is not None:
            x = self.down(x)
        return x


class ExportUpBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.resnets = nn.ModuleList([ExportResnet(r) for r in block.resnets])
        self.up = None
        if block.upsamplers is not None:
            self.up = ExportUpsample(block.upsamplers[0])

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.up is not None:
            x = self.up(x)
        return x


class ExportMidBlock(nn.Module):
    def __init__(self, mid):
        super().__init__()
        self.resnets = nn.ModuleList([ExportResnet(r) for r in mid.resnets])
        self.attns = nn.ModuleList([ExportAttention(a) for a in mid.attentions])

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attns, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x


class ExportEncOut(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.norm = ExportGroupNorm(enc.conv_norm_out)
        self.conv_out = ExportCausalConv3d(enc.conv_out)

    def forward(self, x):
        x = self.norm(x)
        x = F.silu(x)
        return self.conv_out(x)


class ExportDecOut(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.norm = ExportGroupNorm(dec.conv_norm_out)
        self.conv_out = ExportCausalConv3d(dec.conv_out)

    def forward(self, x):
        x = self.norm(x)
        x = F.silu(x)
        return self.conv_out(x)


def build_encoder_stages(enc) -> dict:
    return {
        "enc_conv_in": ExportCausalConv3d(enc.conv_in),
        "enc_down0": ExportDownBlock(enc.down_blocks[0]),
        "enc_down1": ExportDownBlock(enc.down_blocks[1]),
        "enc_down2": ExportDownBlock(enc.down_blocks[2]),
        "enc_down3": ExportDownBlock(enc.down_blocks[3]),
        "enc_mid": ExportMidBlock(enc.mid_block),
        "enc_out": ExportEncOut(enc),
    }


def build_decoder_stages(dec) -> dict:
    return {
        "dec_conv_in": ExportCausalConv3d(dec.conv_in),
        "dec_mid": ExportMidBlock(dec.mid_block),
        "dec_up0": ExportUpBlock(dec.up_blocks[0]),
        "dec_up1": ExportUpBlock(dec.up_blocks[1]),
        "dec_up2": ExportUpBlock(dec.up_blocks[2]),
        "dec_up3": ExportUpBlock(dec.up_blocks[3]),
        "dec_out": ExportDecOut(dec),
    }