#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch
from einops import rearrange
from inspect import isfunction
import numpy as np
from timm.models.layers import trunc_normal_


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


class Attention(torch.nn.Module):
    def __init__(
            self, dim, num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = torch.nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        
    def qkv_cal(self, q, k, v, mask=None):
        # [B, P, D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots = dots + mask
        attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]
        return out

    def forward(self, x, context=None, mask=None):
        # x [B, P_q, D]; context [B, P_kv, D]
        b, n, _ = x.shape
        kv_input = default(context, x)
        q_input = x

        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        out = self.qkv_cal(q, k, v, mask)
        # [B, P, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class STMAttention(Attention):
    def forward(self, q_input, kv_input, token_score):
        # q [B, P_q, D]; kv [B, P_kv, D]; token_score [B, P_kv, 1]
        b, n, _ = q_input.shape
        
        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        # qkv_cal
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        token_score = token_score.squeeze(-1)[:, None, None, :]  # [B, 1, 1, P_kv]
        attn = (dots + token_score).softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

