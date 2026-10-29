"""
gpt.py — NanoGPT: Full Transformer Language Model
---------------------------------------------------
Built incrementally following Andrej Karpathy's video:
"Let's build GPT: from scratch, in code, spelled out"
https://www.youtube.com/watch?v=kCc8FmEb1nY

Phases implemented here (each building on the last):
  Phase 2 — Self-Attention Head
  Phase 3 — Multi-Head Attention
  Phase 4 — Feed-Forward Network
  Phase 5 — Residual Connections + Layer Norm + Dropout
  Phase 6 — Full GPT (stacked Transformer blocks)
"""

import math
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Project root ──────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

def _root(*parts):
    return os.path.join(_PROJ_ROOT, *parts)

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(1337)

# ── Hyperparameters ──────────────────────────────────────────────────────────
# Two configs below — pick one based on your hardware.
#
# ① Small (CPU-friendly, ~2 min training, ~1.98 val loss)
batch_size    = 32     # sequences processed in parallel
block_size    = 64     # context window (T): how many chars the model looks back
max_iters     = 5000
eval_interval = 500
eval_iters    = 200
learning_rate = 3e-4
n_embd        = 64     # embedding dimension
n_head        = 4      # number of attention heads; each head = n_embd // n_head = 16
n_layer       = 4      # number of stacked Transformer blocks
dropout       = 0.0    # set > 0 (e.g. 0.2) if overfitting

# ── LR schedule, regularisation, checkpointing ───────────────────────────────
warmup_iters  = 100          # steps over which LR linearly rises to peak
min_lr        = 3e-5         # floor for cosine decay (= learning_rate / 10)
grad_clip     = 1.0          # max gradient norm; 0.0 = disabled
weight_decay  = 1e-1         # AdamW weight decay (applied to non-bias params)
checkpoint_dir = _root('stage_3_gpt_shakespeare', 'checkpoints')  # directory for saved model files

# ② Full (GPU/MPS, ~15 min, ~1.49 val loss) — uncomment to use
# batch_size    = 64
# block_size    = 256
# max_iters     = 5000
# eval_interval = 500
# eval_iters    = 200
# learning_rate = 3e-4
# n_embd        = 384
# n_head        = 6      # each head = 384 // 6 = 64 dims
# n_layer       = 6
# dropout       = 0.2


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable. Check GPU allocation and the PyTorch installation."
    )

device = 'cuda'
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

#device = 'mps' if torch.backends.mps.is_available() else \
 #        'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f"Using device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
_local_input = os.path.join(_HERE, 'shakespeare_input.txt')
_input_path  = _local_input if os.path.exists(_local_input) else ('input.txt' if os.path.exists('input.txt') else _root('input.txt'))
with open(_input_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset: {len(text):,} characters")

# ── Vocabulary ────────────────────────────────────────────────────────────────
chars      = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Train / Val split ─────────────────────────────────────────────────────────
data = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── Data loader ───────────────────────────────────────────────────────────────
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i + block_size]         for i in ix])
    y  = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ── Loss estimation ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — SELF-ATTENTION HEAD
# ════════════════════════════════════════════════════════════════════════════
#
# THE CORE IDEA (from the video):
# ─────────────────────────────────────────────────────────────────────────
# In the Bigram model, token at position t was completely isolated.
# We want tokens to "talk to each other" — but only to tokens that
# came BEFORE them (causal / decoder-only style, no peeking into the future).
#
# Step 1 — The naive approach: just average all past token embeddings.
#   x[t] = mean(emb[0], emb[1], ..., emb[t])
#   Problem: a plain mean treats all past tokens equally.
#
# Step 2 — The mathematical trick (tril + softmax):
#   Instead of a loop, we can do this in one matrix multiply:
#
#     tril = [[1, 0, 0],    # position 0 can only see itself
#             [1, 1, 0],    # position 1 can see 0 and 1
#             [1, 1, 1]]    # position 2 can see all
#
#     wei = tril / tril.sum(dim=-1, keepdim=True)  # normalise rows
#     out = wei @ x                                 # weighted sum
#
#   This is equivalent to the loop but computed for ALL positions at once.
#   Replace the hard 0s with -inf before softmax → softmax makes them 0.
#
# Step 3 — Self-Attention (data-dependent weights):
#   Plain averaging gives every past token equal weight.
#   Self-attention makes the weights *learned* and *data-dependent*:
#
#   Each token emits:
#     Query (Q): "what am I looking for?"
#     Key   (K): "what do I contain?"
#     Value (V): "what do I communicate if attended to?"
#
#   Attention score between token i and token j:
#     score(i, j) = Q[i] · K[j]   (dot product = affinity)
#
#   Scale by 1/√head_size to prevent softmax from saturating into one-hot:
#     wei = softmax( (Q @ K.T) / sqrt(head_size) )   (T, T) matrix
#
#   Output:
#     out = wei @ V    (T, head_size)
#
#   Think of it like a soft, differentiable dictionary lookup:
#     - Q = your search query
#     - K = the keys of items in the dictionary
#     - V = the values you retrieve, weighted by how well Q matches K
# ─────────────────────────────────────────────────────────────────────────

class Head(nn.Module):
    """
    One head of causal self-attention.

    head_size: dimensionality of Q, K, V projections.
    Input:  (B, T, n_embd)
    Output: (B, T, head_size)
    """

    def __init__(self, head_size):
        super().__init__()
        # Linear projections — no bias, as is conventional
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 'tril' is NOT a model parameter (no gradient) — it's a fixed mask.
        # register_buffer keeps it on the correct device without it being
        # returned by model.parameters().
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape               # C = n_embd

        k = self.key(x)                 # (B, T, head_size)
        q = self.query(x)               # (B, T, head_size)

        # ── Compute attention scores ──────────────────────────────────────
        # Scaled dot-product: dividing by sqrt(head_size) keeps variance ~1
        # so softmax doesn't collapse to a one-hot (all weight on one token).
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size**-0.5   # (B, T, T)

        # ── Causal mask: no peeking at future tokens ──────────────────────
        # Replace upper-triangle with -inf → softmax turns them into 0.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # ── Softmax → normalised attention weights ────────────────────────
        wei = F.softmax(wei, dim=-1)    # (B, T, T)
        wei = self.dropout(wei)

        # ── Weighted aggregation of Values ────────────────────────────────
        v   = self.value(x)             # (B, T, head_size)
        out = wei @ v                   # (B, T, head_size)
        return out


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — MULTI-HEAD ATTENTION
# ════════════════════════════════════════════════════════════════════════════
#
# One attention head gives the model ONE "communication channel".
# Multi-head attention runs n_head heads IN PARALLEL, each with a smaller
# head_size = n_embd // n_head, then concatenates their outputs.
#
# Why? Different heads can attend to different aspects simultaneously:
#   head 1 → maybe focuses on syntactic relationships
#   head 2 → maybe focuses on semantic similarity
#   head 3 → maybe focuses on positional proximity
#   ... (the model learns what each head specialises in)
#
# After concatenating: (B, T, n_head * head_size) = (B, T, n_embd)
# Then project back to n_embd with a linear layer.
# ─────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    n_head attention heads running in parallel, outputs concatenated
    then projected back to n_embd.
    """

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        # Projection back into the residual stream (n_head * head_size = n_embd)
        self.proj    = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate along the channel dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — FEED-FORWARD NETWORK (FFN)
# ════════════════════════════════════════════════════════════════════════════
#
# After attention, tokens have "communicated" with each other.
# But they haven't had a chance to "think" individually about what they gathered.
#
# FFN is applied to EACH TOKEN INDEPENDENTLY (same weights, different inputs):
#   x → Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd)
#
# The 4× expansion is from the original "Attention is All You Need" paper.
# It gives the model extra compute capacity per-token.
# ─────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Per-token MLP: expand by 4×, apply ReLU, project back."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection back into residual stream
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — TRANSFORMER BLOCK (Residuals + Layer Norm)
# ════════════════════════════════════════════════════════════════════════════
#
# RESIDUAL CONNECTIONS (skip connections):
# ─────────────────────────────────────────
#   x = x + attention(x)   ← NOT just attention(x), but x PLUS it
#   x = x + ffn(x)
#
# Why? In deep networks, gradients vanish as they flow backwards.
# The residual path (+x) gives gradients a "highway" to flow directly
# to earlier layers without passing through many transformations.
# At init, the sublayers contribute ~0, so the model starts as identity.
#
# LAYER NORMALISATION:
# ─────────────────────────────────────────
#   Unlike BatchNorm (normalises across the batch dimension),
#   LayerNorm normalises across the EMBEDDING dimension for each token.
#   → More stable training, works with any batch size.
#
#   Pre-LN (applied BEFORE sublayer, not after):
#     x = x + attention(LayerNorm(x))
#     x = x + ffn(LayerNorm(x))
#   This is a minor deviation from the original paper but is standard today.
# ─────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    One Transformer block:
        x → LayerNorm → MultiHeadAttention → residual add
        x → LayerNorm → FeedForward        → residual add
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size    = n_embd // n_head
        self.sa      = MultiHeadAttention(n_head, head_size)  # communication
        self.ffn     = FeedForward(n_embd)                    # computation
        self.ln1     = nn.LayerNorm(n_embd)                   # pre-attention norm
        self.ln2     = nn.LayerNorm(n_embd)                   # pre-FFN norm

    def forward(self, x):
        # Residual: x stays on the highway, sublayer adds a "delta"
        x = x + self.sa(self.ln1(x))   # attend → add
        x = x + self.ffn(self.ln2(x))  # think  → add
        return x


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — FULL GPT LANGUAGE MODEL
# ════════════════════════════════════════════════════════════════════════════
#
# Two kinds of information flow into the model:
#   1. Token embeddings   — WHAT each token is    (vocab × n_embd lookup)
#   2. Position embeddings — WHERE each token is  (block_size × n_embd lookup)
#
# Why position embeddings? Attention (without positional info) is
# permutation-EQUIVARIANT — reordering the input tokens reorders the outputs
# in the same way, but the pairwise attention weights themselves are unchanged.
# Causal masking adds an ordering constraint, but still no sense of absolute
# position. We inject position embeddings so the model knows WHERE each token
# sits in the sequence, not just WHAT it is.
#
# Full architecture:
#   idx (B, T)
#     → token_emb   (B, T, n_embd)
#     +  pos_emb    (B, T, n_embd)   ← learned position encodings
#     → Block × n_layer              ← stacked transformer blocks
#     → LayerNorm                    ← final norm before decoding
#     → Linear(n_embd → vocab_size)  ← language model head
#     → logits (B, T, vocab_size)
# ─────────────────────────────────────────────────────────────────────────

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token embedding: maps each token index → n_embd-dimensional vector
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        # Position embedding: maps each position 0..block_size-1 → n_embd vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of n_layer transformer blocks (the main engine)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        # Final layer norm (applied after the last block, before the head)
        self.ln_f   = nn.LayerNorm(n_embd)
        # Language model head: project from n_embd → vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Weight initialisation (important for stable training in deep nets)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialise weights following GPT-2 paper conventions:
          - Linear & Embedding: Normal(0, 0.02)
          - Linear biases: zeros
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) — input token indices
        targets: (B, T) — target token indices (optional, for training)
        """
        B, T = idx.shape

        # Token + position embeddings
        tok_emb = self.token_embedding_table(idx)                          # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb      # (B, T, n_embd) — broadcasts B dimension

        # Pass through all transformer blocks
        x = self.blocks(x)         # (B, T, n_embd)

        # Final layer norm
        x = self.ln_f(x)           # (B, T, n_embd)

        # Project to vocabulary logits
        logits = self.lm_head(x)   # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate max_new_tokens tokens.

        Key difference from bigram: we crop the context to block_size
        because position embeddings only go up to block_size.

        @torch.no_grad() — no gradients needed during inference; saves memory.
        model.eval() / model.train() — disables dropout during generation so
        predictions are deterministic; matters once dropout > 0.0.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens
            idx_cond = idx[:, -block_size:]          # (B, T) with T ≤ block_size
            logits, _ = self(idx_cond)               # (B, T, vocab_size)
            logits = logits[:, -1, :]                # (B, vocab_size) — last position
            probs  = F.softmax(logits, dim=-1)       # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()
        return idx


# ── Instantiate & summarise ───────────────────────────────────────────────────
model = GPTLanguageModel().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
# ~200K for small config | ~10M for full config

# ── LR schedule: linear warmup → cosine decay ────────────────────────────────
def get_lr(step):
    """
    Two-phase schedule used by GPT-2 / GPT-3:

    1. Linear warmup (0 → learning_rate) over warmup_iters steps.
       Why warmup? At init, weights are random and gradients are noisy.
       A small LR prevents a damaging first update from sending the model
       into a bad region it never recovers from.

    2. Cosine decay (learning_rate → min_lr) for the remaining steps.
       Cosine is smooth — it starts decaying slowly, speeds up in the middle,
       then slows again near the end, which empirically beats linear decay.
    """
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters          # linear ramp
    if step >= max_iters:
        return min_lr
    # Cosine interpolation between learning_rate and min_lr
    progress = (step - warmup_iters) / (max_iters - warmup_iters) # 0.0 → 1.0
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))         # 1.0 → 0.0
    return min_lr + coeff * (learning_rate - min_lr)

# ── Training ──────────────────────────────────────────────────────────────────
# Separate parameters into two groups:
#   - weight matrices / embeddings → apply weight decay (regularisation)
#   - biases / LayerNorm params    → no weight decay (they don't overfit)
no_decay  = {'bias', 'weight'}   # LayerNorm.weight & .bias kept ungrouped
decay_params    = [p for n, p in model.named_parameters()
                   if p.dim() >= 2]          # matrices: Linear, Embedding
nodecay_params  = [p for n, p in model.named_parameters()
                   if p.dim() < 2]           # biases, LN scale/shift

optimizer = torch.optim.AdamW(
    [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ],
    lr=learning_rate,
    betas=(0.9, 0.95),   # GPT-2 betas; β2=0.95 instead of 0.999 is better
                         # for models trained for few steps (less momentum lag)
)

print(f"Training GPT ({n_layer} layers, {n_head} heads, n_embd={n_embd}, block_size={block_size})...")
print(f"  Decay params: {sum(p.numel() for p in decay_params):,} | "
      f"No-decay params: {sum(p.numel() for p in nodecay_params):,}\n")

os.makedirs(checkpoint_dir, exist_ok=True)
best_val_loss = float('inf')

for step in range(max_iters):

    # ── Set LR for this step ─────────────────────────────────────────────
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ── Periodic evaluation ───────────────────────────────────────────────
    if step % eval_interval == 0 or step == max_iters - 1:
        t0 = time.time()
        losses = estimate_loss(model)
        t1 = time.time()

        # Save best checkpoint when val loss improves
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt = {
                'step':      step,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss':  best_val_loss,
                # hyperparams — so we can reconstruct the model later
                'config': dict(vocab_size=vocab_size, n_embd=n_embd,
                               n_head=n_head, n_layer=n_layer,
                               block_size=block_size, dropout=dropout),
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, 'best.pt'))
            saved = ' ← saved best'
        else:
            saved = ''

        print(f"  step {step:5d}: train {losses['train']:.4f} | "
              f"val {losses['val']:.4f} | lr {lr:.2e}{saved}")

    # ── Forward pass ──────────────────────────────────────────────────────
    t_start = time.time()
    xb, yb  = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # ── Gradient clipping ─────────────────────────────────────────────────
    # Why clip? If a batch is unusually informative, the gradient can be
    # enormous, causing a single step to overshoot badly ("gradient spike").
    # Clipping to norm=1.0 preserves the gradient direction but caps its
    # magnitude — a simple, effective safeguard.
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    # ── Throughput (tokens/sec) ───────────────────────────────────────────
    # Printed every 500 steps so the terminal isn't flooded.
    if step % eval_interval == eval_interval - 1:
        elapsed = time.time() - t_start
        tokens_per_sec = (batch_size * block_size) / elapsed
        print(f"          throughput: {tokens_per_sec:,.0f} tokens/sec")

# ── Generation — load best checkpoint ────────────────────────────────────────
best_ckpt_path = os.path.join(checkpoint_dir, 'best.pt')
if os.path.exists(best_ckpt_path):
    print(f"\nLoading best checkpoint (val loss: {best_val_loss:.4f})...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])

print("\n── Generated text (NanoGPT) ───────────────────────────────────────────")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
print("───────────────────────────────────────────────────────────────────────")
