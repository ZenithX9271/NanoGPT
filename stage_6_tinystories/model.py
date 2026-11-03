"""Parameterised Stage 5 architecture; state-dictionary names are unchanged.

FP32, manual causal attention, ReLU, pre-LN, learned positions, tied embeddings.
There is no SDPA/AMP/compile dependency. Compatible with Python 3.7 / torch 1.10.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg['n_embd'], head_size, bias=False)
        self.query = nn.Linear(cfg['n_embd'], head_size, bias=False)
        self.value = nn.Linear(cfg['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg['block_size'], cfg['block_size'])))
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        k, q = self.key(x), self.query(x)
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:x.size(1), :x.size(1)] == 0, float('-inf'))
        return self.dropout(F.softmax(scores, dim=-1)) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg, cfg['n_embd'] // cfg['n_head'])
                                   for _ in range(cfg['n_head'])])
        self.proj = nn.Linear(cfg['n_embd'], cfg['n_embd'])
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n = cfg['n_embd']
        self.net = nn.Sequential(nn.Linear(n, 4*n), nn.ReLU(), nn.Linear(4*n, n),
                                 nn.Dropout(cfg['dropout']))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffn = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg['n_embd'])
        self.ln2 = nn.LayerNorm(cfg['n_embd'])

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg['n_embd'] % cfg['n_head'] or not cfg['weight_tying']:
            raise ValueError('Stage 6A requires divisible heads and weight tying')
        self.cfg = dict(cfg)
        self.token_embedding_table = nn.Embedding(cfg['vocab_size'], cfg['n_embd'])
        self.position_embedding_table = nn.Embedding(cfg['block_size'], cfg['n_embd'])
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg['n_layer'])])
        self.ln_f = nn.LayerNorm(cfg['n_embd'])
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        # Matching the Stage 5 initialisation convention; loaded weights replace it.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        if idx.size(1) > self.cfg['block_size']:
            raise ValueError('Input exceeds configured context')
        x = (self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(idx.size(1), device=idx.device)))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, generator, pad_id=0, bos_id=1, eos_id=2,
                 temperature=0.7, top_k=50, greedy=False, stop=None):
        if ids.size(0) != 1 or (not greedy and temperature <= 0):
            raise ValueError('Generation expects one prompt and positive temperature')
        was_training = self.training
        self.eval()
        reason = 'max_tokens'
        try:
            for _ in range(max_new_tokens):
                if stop is not None and stop():
                    reason = 'walltime'
                    break
                logits, _ = self(ids[:, -self.cfg['block_size']:])
                logits = logits[:, -1, :].clone()
                logits[:, [pad_id, bos_id]] = float('-inf')
                if greedy:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits /= temperature
                    if top_k is not None:
                        values = torch.topk(logits, min(top_k, logits.size(-1))).values
                        logits.masked_fill_(logits < values[:, -1:], float('-inf'))
                    next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == eos_id:
                    reason = 'eos'
                    break
        finally:
            self.train(was_training)
        return ids, reason
