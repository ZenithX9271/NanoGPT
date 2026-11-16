"""Portable pre-RMSNorm/RoPE/SwiGLU decoder; aligned next-token targets.

One fused QKV projection, no persistent causal-mask buffers. Cached positions
continue monotonically; overflowing the configured context requires a new prefill.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rotary(x, positions, theta):
    dim = x.size(-1)
    frequency = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angle = positions.float()[:, None] * frequency[None, :]
    cosine, sine = angle.cos()[None, None, :, :], angle.sin()[None, None, :, :]
    a, b = x.float()[..., 0::2], x.float()[..., 1::2]
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg['n_head']
        self.width = cfg['n_embd']
        self.theta = cfg.get('rope_theta', 10000.)
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.proj = nn.Linear(self.width, self.width, bias=False)
        self.dropout = nn.Dropout(cfg.get('dropout', 0.0))

    def forward(self, x, key_mask=None, past=None, use_cache=False):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        previous = 0 if past is None else past[0].size(2)
        positions = torch.arange(previous, previous + t, device=x.device)
        q, k = rotary(q, positions, self.theta), rotary(k, positions, self.theta)
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        allowed = torch.arange(k.size(2), device=x.device)[None, :] <= positions[:, None]
        allowed = allowed[None, None, :, :]
        if key_mask is not None:
            if key_mask.shape != (b, k.size(2)):
                raise ValueError('attention_mask must cover all cached and current key positions')
            allowed = allowed & key_mask[:, None, None, :].bool()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(v.dtype) * allowed.to(v.dtype)
        y = self.dropout(weights) @ v
        return self.dropout(self.proj(y.transpose(1, 2).contiguous().view(b, t, d))), ((k, v) if use_cache else None)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg['n_embd'], cfg['intermediate_size']
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)
        self.dropout = nn.Dropout(cfg.get('dropout', 0.0))

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(cfg['n_embd']), RMSNorm(cfg['n_embd'])
        self.attn, self.ffn = Attention(cfg), SwiGLU(cfg)

    def forward(self, x, mask, past, use_cache):
        y, cache = self.attn(self.attn_norm(x), mask, past, use_cache)
        x = x + y
        return x + self.ffn(self.ffn_norm(x)), cache


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d, heads = cfg['n_embd'], cfg['n_head']
        if d % heads or (d // heads) % 2 or min(cfg['vocab_size'], cfg['block_size'], cfg['n_layer'], cfg['intermediate_size']) < 1:
            raise ValueError('Positive dimensions and even, divisible rotary head size required')
        self.embedding = nn.Embedding(cfg['vocab_size'], d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['n_layer'])])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg['vocab_size'], bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, targets=None, attention_mask=None, past=None, use_cache=False, reduction='mean'):
        if ids.dtype != torch.long or ids.ndim != 2 or ids.size(1) < 1:
            raise ValueError('Expected nonempty [batch,time] int64 token IDs')
        if torch.any((ids < 0) | (ids >= self.cfg['vocab_size'])).item():
            raise ValueError('Token ID outside vocabulary')
        if past is not None and len(past) != len(self.blocks):
            raise ValueError('Cache layer count mismatch')
        offset = 0 if past is None else past[0][0].size(2)
        if offset + ids.size(1) > self.cfg['block_size']:
            raise ValueError('Context overflow: rebuild prefill; cache eviction is not implicit')
        x, caches = self.embedding(ids), []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, attention_mask, None if past is None else past[i], use_cache)
            caches.append(cache)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape != ids.shape or torch.any((targets != -100) & ((targets < 0) | (targets >= self.cfg['vocab_size']))).item():
                raise ValueError('Targets must be aligned next-token IDs or -100')
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100, reduction='sum')
            if reduction == 'mean':
                loss = loss / targets.ne(-100).sum().clamp_min(1)
            elif reduction != 'sum':
                raise ValueError('reduction must be mean or sum')
        return (logits, loss, tuple(caches)) if use_cache else (logits, loss)

    @torch.no_grad()
    def greedy(self, ids, new_tokens, eos_id=None):
        if new_tokens < 0 or ids.size(1) + new_tokens > self.cfg['block_size']:
            raise ValueError('Generation budget exceeds context')
        old = self.training
        self.eval()
        try:
            logits, _, cache = self(ids, use_cache=True)
            for i in range(new_tokens):
                token = logits[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat((ids, token), 1)
                if eos_id is not None and bool(token.eq(eos_id).all()):
                    break
                if i + 1 < new_tokens:
                    logits, _, cache = self(token, past=cache, use_cache=True)
            return ids
        finally:
            self.train(old)
