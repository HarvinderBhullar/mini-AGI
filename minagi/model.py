"""The transformer itself.

Decoder-only on the current small-model recipe: RMSNorm, rotary position
embeddings, SwiGLU feed-forward, flash attention through
scaled_dot_product_attention, tied input/output embeddings, no biases.

Positions are rotary and carry no learned parameters, which is why the context
window extends by continued training rather than by re-initialising anything -
see stream.ramp_context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from . import device as _device


# TF32 where there is a CUDA matmul path to turn it on for, and nothing
# anywhere else. Reading torch.backends.cuda at import time works on a machine
# with no CUDA - the module exists regardless - but it states an assumption
# this file no longer makes, so the backend question is asked of the one place
# that owns it.
_device.enable_tf32()


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    block: int = 512
    d_ff: int = 1408          # ~8/3 * d_model, rounded to a multiple of 64
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


# ----------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalise in the input dtype, reduce in fp32. Three details, each
        # of which exists to keep an fp32 copy of the activation out of the
        # backward graph - that copy is what dominates activation memory at
        # this depth:
        #
        #   mean(dtype=fp32)  accumulates the reduction in fp32 without
        #                     upcasting x itself, so accuracy is kept where
        #                     the reciprocal-sqrt needs it and nowhere else.
        #   x * x, not pow()  autocast keeps pow on its fp32 list, so pow()
        #                     upcasts and its backward retains the copy. mul
        #                     is not on that list.
        #   weight.to(dtype)  bf16 * fp32 promotes the product back to fp32,
        #                     so the cast is explicit.
        #
        # The cost is one bf16 rounding on the scale multiply, well under the
        # held-out noise floor.
        scale = torch.rsqrt((x * x).mean(-1, keepdim=True,
                                         dtype=torch.float32) + self.eps)
        return x * scale.to(x.dtype) * self.weight.to(x.dtype)


def build_rope(block, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                           / head_dim))
    pos = torch.arange(block, device=device).float()
    freqs = torch.outer(pos, inv)                      # [T, hd/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: [B, H, T, hd]; rotate pairs (even, odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # Masking with a cache is not the same problem as masking without one.
        # `is_causal` aligns its triangle to the TOP-LEFT of the score matrix,
        # which is correct only when the queries and the keys are the same
        # positions. With P cached positions the T queries sit at absolute
        # P..P+T-1 against P+T keys, so the triangle has to be offset by P -
        # otherwise query i sees keys 0..i instead of 0..P+i and every chunk
        # after the first attends to the wrong window.
        #
        # kv_len == T means there is no cache and the fast path is correct.
        kv_len = k.shape[2]
        if kv_len == T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            P = kv_len - T
            qi = torch.arange(T, device=q.device).unsqueeze(1) + P
            ki = torch.arange(kv_len, device=q.device).unsqueeze(0)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=(ki <= qi))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # value
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, cache)
        x = x + self.mlp(self.ln2(x))
        return x
