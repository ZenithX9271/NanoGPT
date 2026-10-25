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
import platform
import resource
import sys
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(1337)

# ── Hyperparameters ──────────────────────────────────────────────────────────
# ① Small baseline — matches historical run (211,777 params)
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
warmup_iters   = 100          # steps over which LR linearly rises to peak
min_lr         = 3e-5         # floor for cosine decay (= learning_rate / 10)
grad_clip      = 1.0          # max gradient norm; 0.0 = disabled
weight_decay   = 1e-1         # AdamW weight decay (applied to weight matrices)
checkpoint_dir = 'checkpoints'

# ── Walltime safety margin ────────────────────────────────────────────────────
# Set to your PBS walltime so the script saves latest.pt and exits cleanly
# before PBS kills the job with SIGKILL. None = disabled (local runs).
WALLTIME_MINUTES = None    # e.g. 120 for a 2-hour job
MARGIN_SECONDS   = 300     # save and exit this many seconds before the limit

# ── Benchmark settings ────────────────────────────────────────────────────────
# These training steps are real gradient updates.
# The benchmark measures throughput on the ACTUAL workload, not dummy kernels.
BM_WARMUP       = 50    # steps before timing (GPU caches warm, CUDA JIT settled)
BM_WINDOWS      = 5     # number of measured windows
BM_WINDOW_STEPS = 100   # steps per window
BM_TOTAL        = BM_WARMUP + BM_WINDOWS * BM_WINDOW_STEPS  # 550

# ② Full config (GPU, ~15 min) — uncomment to use
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

# ── Device ───────────────────────────────────────────────────────────────────
# Cluster policy: CUDA is required. Fail loudly — never silently train on CPU
# when a GPU was allocated. Comment this block and uncomment the fallback below
# for local Mac/CPU development.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable. Load the correct PyTorch module and confirm that a "
        "GPU node was allocated before running this script.\n"
        "IITD HPC: module load apps/pytorch/1.10.0/gpu/intelpython3.7"
    )

device = 'cuda'
print("Using device: cuda")
print("GPU: {}".format(torch.cuda.get_device_name(0)))

# Local development fallback (Mac MPS / CPU):
# _has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
# device = 'mps' if _has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using device: {}".format(device))

# ── Dataset ───────────────────────────────────────────────────────────────────
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Dataset: {:,} characters".format(len(text)))

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
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i + block_size]         for i in ix])
    y  = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
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
        tok_emb = self.token_embedding_table(idx)                                    # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
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
            idx_cond = idx[:, -block_size:]          # (B, T) with T <= block_size
            logits, _ = self(idx_cond)               # (B, T, vocab_size)
            logits = logits[:, -1, :]                # (B, vocab_size) — last position
            probs  = F.softmax(logits, dim=-1)       # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()
        return idx


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


# ── One training step (used by both benchmark and main loop) ─────────────────
def train_step(model, optimizer, step):
    """
    Execute one forward + backward + clip + optimizer step.
    Returns (loss_value, grad_norm).
    """
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = 0.0
    if grad_clip > 0.0:
        # clip_grad_norm_ returns the pre-clip total gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

    optimizer.step()
    return loss.item(), grad_norm


# ════════════════════════════════════════════════════════════════════════════
#  EXECUTION — guarded so `import gpt` does not start training
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    t_script_start = time.time()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPTLanguageModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate parameters into two groups:
    #   - weight matrices / embeddings → apply weight decay (regularisation)
    #   - biases / LayerNorm params    → no weight decay (they don't overfit)
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() <  2]

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),   # β2=0.95 (GPT-2); less momentum lag than 0.999
    )

    # ── Environment header ────────────────────────────────────────────────────
    print("")
    print("=" * 62)
    print("  ENVIRONMENT")
    print("=" * 62)
    print("  GPU:       {}".format(torch.cuda.get_device_name(0)))
    print("  CUDA:      {}".format(torch.version.cuda))
    print("  cuDNN:     {}".format(torch.backends.cudnn.version()))
    print("  PyTorch:   {}".format(torch.__version__))
    print("  Python:    {}".format(sys.version.split()[0]))
    print("  Platform:  {}".format(platform.platform()))
    print("  Model:     {:,} params ({:,} decay | {:,} no-decay)".format(
        total_params,
        sum(p.numel() for p in decay_params),
        sum(p.numel() for p in nodecay_params),
    ))
    print("  Config:    batch={} block={} n_embd={} n_head={} n_layer={} "
          "dropout={}".format(batch_size, block_size, n_embd, n_head, n_layer, dropout))
    print("  Data:      input.txt {:,} chars | vocab {}".format(len(text), vocab_size))
    print("  Seed:      1337")
    print("=" * 62)
    print("")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss    = float('inf')
    tokens_processed = 0
    tokens_per_step  = batch_size * block_size  # no gradient accumulation

    # ════════════════════════════════════════════════════════════════════════
    #  BENCHMARK PHASE  (steps 0 … BM_TOTAL-1)
    #  These are real gradient updates — the model trains during benchmarking.
    # ════════════════════════════════════════════════════════════════════════
    print("BENCHMARK:  {} warmup + {} windows × {} steps = {} total steps".format(
        BM_WARMUP, BM_WINDOWS, BM_WINDOW_STEPS, BM_TOTAL))
    print("")

    torch.cuda.reset_peak_memory_stats()

    # ── Warmup ────────────────────────────────────────────────────────────────
    # Run without any timing measurement; allows CUDA JIT/kernel caches to settle.
    for step in range(BM_WARMUP):
        train_step(model, optimizer, step)
        tokens_processed += tokens_per_step

    torch.cuda.synchronize()
    t_warmup_done = time.time()
    print("  Warmup done ({} steps, {:.1f}s from script start).".format(
        BM_WARMUP, t_warmup_done - t_script_start))
    print("")

    # ── Measured windows ─────────────────────────────────────────────────────
    # Two kinds of timing, both reported with explicit scope:
    #
    #   Wall-clock window timing: torch.cuda.synchronize() at window START and
    #   END only. GPU pipeline runs freely between syncs. This gives the real
    #   sustained throughput you would see during a full training run.
    #
    #   Per-step CUDA event timing: torch.cuda.synchronize() after every step
    #   to read the CUDA event. This SERIALIZES the GPU pipeline — latency is
    #   accurate but throughput from this path is lower than the window figure.
    #   Labeled clearly in the output.
    window_tps  = []
    all_step_ms = []

    for w in range(BM_WINDOWS):

        # Walltime check — save and exit before PBS kills us
        if WALLTIME_MINUTES is not None:
            elapsed_sec = time.time() - t_script_start
            if elapsed_sec > WALLTIME_MINUTES * 60 - MARGIN_SECONDS:
                print("[WALLTIME] Approaching limit before window {}. Stopping.".format(w + 1))
                break

        step_offset = BM_WARMUP + w * BM_WINDOW_STEPS
        step_ms_window = []

        torch.cuda.synchronize()
        t_w_start = time.time()

        for s in range(BM_WINDOW_STEPS):
            step = step_offset + s

            # CUDA event pair for per-step timing
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end   = torch.cuda.Event(enable_timing=True)

            ev_start.record()
            train_step(model, optimizer, step)
            ev_end.record()

            # sync needed to read event; this serializes this step's measurement
            torch.cuda.synchronize()
            step_ms_window.append(ev_start.elapsed_time(ev_end))
            tokens_processed += tokens_per_step

        torch.cuda.synchronize()
        t_w_end = time.time()

        w_elapsed = t_w_end - t_w_start
        w_tokens  = BM_WINDOW_STEPS * tokens_per_step
        tps       = w_tokens / w_elapsed
        window_tps.append(tps)
        all_step_ms.extend(step_ms_window)

        step_ms_sorted = sorted(step_ms_window)
        med_ms = step_ms_sorted[len(step_ms_sorted) // 2]
        p95_ms = step_ms_sorted[int(0.95 * len(step_ms_sorted))]

        print("  Window {:d}/{:d}:  {:>10,.0f} tok/s  |  step med {:.1f}ms  p95 {:.1f}ms".format(
            w + 1, BM_WINDOWS, tps, med_ms, p95_ms))

    # ── Benchmark aggregate report ────────────────────────────────────────────
    if window_tps:
        mean_tps = sum(window_tps) / len(window_tps)
        min_tps  = min(window_tps)
        max_tps  = max(window_tps)
        var_pct  = (max_tps - min_tps) / mean_tps * 100 if mean_tps > 0 else 0.0

        all_sorted = sorted(all_step_ms)
        med_all    = all_sorted[len(all_sorted) // 2]
        p95_all    = all_sorted[int(0.95 * len(all_sorted))]

        peak_alloc_mib = torch.cuda.max_memory_allocated()  / (1024.0 ** 2)
        peak_resv_mib  = torch.cuda.max_memory_reserved()   / (1024.0 ** 2)

        # Host RAM: ru_maxrss is KB on Linux, bytes on macOS
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ram_mb = ru / 1024.0 if platform.system() == 'Linux' else ru / (1024.0 ** 2)

        # Time evaluation separately (sync before and after)
        torch.cuda.synchronize()
        t_eval_0 = time.time()
        bm_losses = estimate_loss(model)
        torch.cuda.synchronize()
        eval_sec = time.time() - t_eval_0

        # Time checkpoint save separately; use atomic replace
        tmp_path  = os.path.join(checkpoint_dir, 'benchmark_probe.pt.tmp')
        ckpt_path = os.path.join(checkpoint_dir, 'benchmark_probe.pt')
        t_ckpt_0  = time.time()
        torch.save({'model': model.state_dict(), 'step': BM_TOTAL}, tmp_path)
        os.replace(tmp_path, ckpt_path)
        ckpt_sec     = time.time() - t_ckpt_0
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024.0 ** 2)
        os.remove(ckpt_path)

        print("")
        print("=" * 62)
        print("  BENCHMARK RESULTS")
        print("=" * 62)
        print("  Windows measured:   {:d} x {:d} steps".format(len(window_tps), BM_WINDOW_STEPS))
        print("  Throughput (window clock, sync at boundaries only):")
        print("    mean  {:>10,.0f} tok/s".format(mean_tps))
        print("    min   {:>10,.0f} tok/s".format(min_tps))
        print("    max   {:>10,.0f} tok/s".format(max_tps))
        print("    var   {:>10.1f}%  (max-min)/mean".format(var_pct))
        print("  Step latency (CUDA-event, synced per step — pipeline serialized):")
        print("    median  {:.2f} ms".format(med_all))
        print("    p95     {:.2f} ms".format(p95_all))
        print("  GPU memory (peak over benchmark):")
        print("    allocated  {:.1f} MiB".format(peak_alloc_mib))
        print("    reserved   {:.1f} MiB".format(peak_resv_mib))
        print("  Host RAM peak:      {:.1f} MB".format(ram_mb))
        print("  Eval ({:d}b x2):      {:.2f}s  train {:.4f}  val {:.4f}".format(
            eval_iters, eval_sec, bm_losses['train'].item(), bm_losses['val'].item()))
        print("  Checkpoint save:    {:.2f}s  {:.1f} MB".format(ckpt_sec, ckpt_size_mb))
        print("  Startup to warmup:  {:.1f}s".format(t_warmup_done - t_script_start))
        print("  Tokens processed:   {:,}".format(tokens_processed))
        print("  Tokens/sec (est.)   {:,.0f}".format(mean_tps))
        print("")
        print("  NOTE: Earlier 54K-62K tok/s figures were unvalidated (no CUDA")
        print("  sync, single-step wall-clock). This report supersedes them.")
        print("=" * 62)
        print("")

    # ════════════════════════════════════════════════════════════════════════
    #  TRAINING PHASE  (steps BM_TOTAL … max_iters-1)
    # ════════════════════════════════════════════════════════════════════════
    print("TRAINING:  steps {:d} → {:d}".format(BM_TOTAL, max_iters - 1))
    print("")

    for step in range(BM_TOTAL, max_iters):

        # ── Walltime check — save latest.pt and exit cleanly ──────────────
        if WALLTIME_MINUTES is not None:
            elapsed_sec = time.time() - t_script_start
            if elapsed_sec > WALLTIME_MINUTES * 60 - MARGIN_SECONDS:
                print("[WALLTIME] {}s elapsed, saving latest.pt and exiting.".format(
                    int(elapsed_sec)))
                ckpt = {
                    'step':      step,
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss':  best_val_loss,
                    'tokens':    tokens_processed,
                    'config':    dict(vocab_size=vocab_size, n_embd=n_embd,
                                     n_head=n_head, n_layer=n_layer,
                                     block_size=block_size, dropout=dropout),
                }
                tmp = os.path.join(checkpoint_dir, 'latest.pt.tmp')
                torch.save(ckpt, tmp)
                os.replace(tmp, os.path.join(checkpoint_dir, 'latest.pt'))
                print("[WALLTIME] Saved latest.pt at step {:d}. Exiting.".format(step))
                sys.exit(0)

        # ── Periodic evaluation and checkpointing ─────────────────────────
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model)
            lr_now = get_lr(step)

            # Save latest.pt every eval interval (atomic replace)
            latest_ckpt = {
                'step':      step,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss':  losses['val'].item(),
                'tokens':    tokens_processed,
                'config':    dict(vocab_size=vocab_size, n_embd=n_embd,
                                  n_head=n_head, n_layer=n_layer,
                                  block_size=block_size, dropout=dropout),
            }
            tmp_latest = os.path.join(checkpoint_dir, 'latest.pt.tmp')
            torch.save(latest_ckpt, tmp_latest)
            os.replace(tmp_latest, os.path.join(checkpoint_dir, 'latest.pt'))

            # Save best.pt when val loss improves
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val'].item()
                tmp_best = os.path.join(checkpoint_dir, 'best.pt.tmp')
                torch.save(latest_ckpt, tmp_best)
                os.replace(tmp_best, os.path.join(checkpoint_dir, 'best.pt'))
                saved_tag = ' <- best'
            else:
                saved_tag = ''

            print("  step {:5d}: train {:.4f} | val {:.4f} | lr {:.2e} | "
                  "tokens {:,}{:s}".format(
                      step,
                      losses['train'].item(),
                      losses['val'].item(),
                      lr_now,
                      tokens_processed,
                      saved_tag,
                  ))

        # ── Training step ─────────────────────────────────────────────────
        t0 = time.time()
        loss_val, grad_norm = train_step(model, optimizer, step)
        tokens_processed += tokens_per_step

        # ── Per-step throughput (wall-clock, single step, no CUDA sync) ───
        # Reported every eval_interval-1 steps as a coarse indicator only.
        # NOT a validated benchmark figure — use the benchmark report above.
        if step % eval_interval == eval_interval - 1:
            elapsed = time.time() - t0
            tps_approx = tokens_per_step / elapsed if elapsed > 0 else 0.0
            print("          grad_norm {:.3f} | approx {:.0f} tok/s "
                  "(single-step, no sync)".format(grad_norm, tps_approx))

    # ════════════════════════════════════════════════════════════════════════
    #  GENERATION — load best checkpoint
    # ════════════════════════════════════════════════════════════════════════
    best_path = os.path.join(checkpoint_dir, 'best.pt')
    if os.path.exists(best_path):
        print("\nLoading best checkpoint (val loss: {:.4f})...".format(best_val_loss))
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    total_sec = time.time() - t_script_start
    print("\nTotal runtime: {:.1f}s ({:.1f} min)".format(total_sec, total_sec / 60))

    print("\n" + "-" * 62)
    print("  Generated text (NanoGPT)")
    print("-" * 62)
    context   = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    print(decode(generated))
    print("-" * 62)
