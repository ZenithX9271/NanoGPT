"""
4th_stage_gpt.py — NanoGPT Stage 4: Scaled-Up Transformer
-----------------------------------------------------------
Stage 4 increases model capacity and context length over Stage 3:

  Stage 3  →  Stage 4
  n_embd:   64   →  256      (4× wider embedding)
  n_layer:   4   →    6      (50% deeper)
  n_head:    4   →    8      (8 heads × 32 dims each)
  block_size:64  →  256      (4× longer context window)
  dropout:  0.0  →  0.2      (regularisation for larger model)
  params:  212K  →  ~4.8M

New in Stage 4:
  - Configurable temperature and top-k sampling in generate()
  - Multi-prompt generation comparison at the end
  - Separate checkpoint directory (checkpoints_s4/) — Stage 3 unaffected

Architecture building blocks (UNCHANGED from Stage 3):
  character tokenizer, learned position embeddings, manual causal attention,
  Pre-LN transformer blocks, FP32, warmup+cosine LR, gradient clipping,
  selective weight decay, atomic checkpointing, walltime exit.
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

# ── Stage 4 Hyperparameters ──────────────────────────────────────────────────
#
#   Each change serves a specific purpose:
#
#   n_embd 64 → 256   More dimensions per token → richer representations.
#   n_layer 4 → 6     More transformer blocks → deeper reasoning chains.
#   n_head  4 → 8     8 heads × 32 dims each → more attention subspaces.
#   block_size 64→256 Predictions can now use ~4 sentences of context.
#   dropout 0 → 0.2   Random zeroing during training regularises a larger model
#                     on this small (1M char) dataset. Has no effect at eval.

batch_size    = 32     # start at 32; increase if K40 memory allows
block_size    = 256    # context window in characters
max_iters     = 5000   # can extend if val loss is still improving
eval_interval = 500
eval_iters    = 200
learning_rate = 3e-4   # unchanged from Stage 3
n_embd        = 256    # embedding dimension
n_head        = 8      # attention heads; head_size = 256 // 8 = 32
n_layer       = 6      # stacked transformer blocks
dropout       = 0.2    # regularisation for larger model on small data

# ── LR schedule, regularisation, checkpointing ───────────────────────────────
warmup_iters   = 100
min_lr         = 3e-5          # learning_rate / 10
grad_clip      = 1.0
weight_decay   = 1e-1
checkpoint_dir = 'checkpoints_s4'   # separate from Stage 3's checkpoints/

# ── Walltime safety margin ────────────────────────────────────────────────────
# Set WALLTIME_MINUTES = <PBS walltime>. Script saves latest.pt and exits
# MARGIN_SECONDS before PBS sends SIGKILL.  None = disabled.
WALLTIME_MINUTES = None    # e.g. 115 for a 120-minute job
MARGIN_SECONDS   = 300

# ── Benchmark settings ────────────────────────────────────────────────────────
# Real gradient updates used for timing — not dummy kernels.
BM_WARMUP       = 50
BM_WINDOWS      = 5
BM_WINDOW_STEPS = 100
BM_TOTAL        = BM_WARMUP + BM_WINDOWS * BM_WINDOW_STEPS  # 550

# ── Stage 3 baseline (for comparison in the generation section) ──────────────
# Change these paths if Stage 3 checkpoint is in a different location.
STAGE3_CHECKPOINT = os.path.join('checkpoints', 'best.pt')
STAGE3_N_EMBD     = 64
STAGE3_N_HEAD     = 4
STAGE3_N_LAYER    = 4
STAGE3_BLOCK_SIZE = 64

# ── Device ───────────────────────────────────────────────────────────────────
# Cluster policy: require CUDA explicitly. Never silently fall back to CPU.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable. Load the correct PyTorch module and confirm a "
        "GPU allocation before running.\n"
        "IITD HPC: module load apps/pytorch/1.10.0/gpu/intelpython3.7"
    )

device = 'cuda'
print("Using device: cuda")
print("GPU: {}".format(torch.cuda.get_device_name(0)))

# Local Mac / CPU fallback (comment the CUDA block above, uncomment this):
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
    d  = train_data if split == 'train' else val_data
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
#  MODEL — building blocks identical to Stage 3; only capacity changes
# ════════════════════════════════════════════════════════════════════════════

class Head(nn.Module):
    """One head of causal self-attention. (B, T, n_embd) → (B, T, head_size)"""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                          # (B, T, hs)
        q = self.query(x)                                        # (B, T, hs)
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size**-0.5         # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)                                      # (B, T, hs)
        return wei @ v                                           # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """n_head parallel attention heads, concatenated and projected back."""

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj    = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: n_embd → 4×n_embd (ReLU) → n_embd."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer block: LayerNorm → Attention → add; LayerNorm → FFN → add."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffn  = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                              # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generate max_new_tokens tokens.

        temperature (float, default 1.0):
          Divides the logits before softmax.
          < 1.0  → sharper distribution, more predictable/conservative output
          = 1.0  → unmodified (standard sampling, same as Stage 3)
          > 1.0  → flatter distribution, more surprising/creative output

          Why it works: logits are log-probabilities (unnormalised).
          Dividing by T < 1 spreads them apart (high-probability tokens
          become relatively even more likely). Dividing by T > 1 squashes
          them together (all tokens become more equally likely).

        top_k (int or None, default None):
          Keep only the k most likely next tokens, set everything else to -inf.
          Prevents sampling from the long tail of very unlikely characters.
          None = no filtering, full vocabulary.

          Typical useful values for this character-level model: 10–50.

        NOTE: temperature and top_k affect diversity of OUTPUT SAMPLES only.
        They do NOT change the trained weights or the model's underlying
        predictions. A lower loss model will produce better output at any
        sampling setting.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to position embedding size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]              # (B, vocab_size), last position only

            # ── Temperature ───────────────────────────────────────────────
            logits = logits / temperature

            # ── Top-k filtering ───────────────────────────────────────────
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                # topk returns the k largest values and their indices
                top_values, _ = torch.topk(logits, k)
                # Mask everything below the k-th largest value to -inf
                # top_values[:, [-1]] is the smallest value among top_k
                logits[logits < top_values[:, [-1]]] = float('-inf')

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx


# ── LR schedule: linear warmup → cosine decay ────────────────────────────────
def get_lr(step):
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    if step >= max_iters:
        return min_lr
    progress = (step - warmup_iters) / (max_iters - warmup_iters)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)


# ── One training step ─────────────────────────────────────────────────────────
def train_step(model, optimizer, step):
    """Forward + backward + clip + optimizer update. Returns (loss, grad_norm)."""
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = 0.0
    if grad_clip > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip).item()

    optimizer.step()
    return loss.item(), grad_norm


# ── Generation helper ─────────────────────────────────────────────────────────
def generate_from_prompt(model, prompt_str, max_new_tokens=300,
                         temperature=1.0, top_k=None):
    """Encode prompt → generate → decode. Returns the generated string."""
    context = torch.tensor(encode(prompt_str), dtype=torch.long,
                           device=device).unsqueeze(0)
    out = model.generate(context, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())


# ════════════════════════════════════════════════════════════════════════════
#  EXECUTION — guarded so `import 4th_stage_gpt` does not start training
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    t_script_start = time.time()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPTLanguageModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # ── Optimizer — selective weight decay ────────────────────────────────────
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )

    # ── Environment header ────────────────────────────────────────────────────
    print("")
    print("=" * 62)
    print("  STAGE 4 — ENVIRONMENT")
    print("=" * 62)
    print("  GPU:       {}".format(torch.cuda.get_device_name(0)))
    print("  CUDA:      {}".format(torch.version.cuda))
    print("  cuDNN:     {}".format(torch.backends.cudnn.version()))
    print("  PyTorch:   {}".format(torch.__version__))
    print("  Python:    {}".format(sys.version.split()[0]))
    print("  Platform:  {}".format(platform.platform()))
    print("  Model:     {:,} params  ({:,} decay | {:,} no-decay)".format(
        total_params,
        sum(p.numel() for p in decay_params),
        sum(p.numel() for p in nodecay_params),
    ))
    print("  Config:    batch={} block={} n_embd={} n_head={} n_layer={} "
          "dropout={}".format(batch_size, block_size, n_embd, n_head, n_layer, dropout))
    print("  Data:      input.txt {:,} chars | vocab {}".format(len(text), vocab_size))
    print("  Token budget: {:,} tokens ({:,}M)".format(
        max_iters * batch_size * block_size,
        max_iters * batch_size * block_size // 1_000_000,
    ))
    print("  Note: Stage 3 processed 10.24M tokens; Stage 4 processes 40.96M")
    print("        (4× more, due to 4× longer context at same batch/iters)")
    print("=" * 62)
    print("")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss    = float('inf')
    tokens_processed = 0
    tokens_per_step  = batch_size * block_size   # 32 × 256 = 8,192

    # ════════════════════════════════════════════════════════════════════════
    #  BENCHMARK PHASE  (steps 0 … BM_TOTAL-1 = 550 real training steps)
    # ════════════════════════════════════════════════════════════════════════
    print("BENCHMARK:  {} warmup + {} windows × {} steps = {} steps".format(
        BM_WARMUP, BM_WINDOWS, BM_WINDOW_STEPS, BM_TOTAL))
    print("")

    torch.cuda.reset_peak_memory_stats()

    # Warmup — no timing; let CUDA JIT and caches settle
    for step in range(BM_WARMUP):
        train_step(model, optimizer, step)
        tokens_processed += tokens_per_step

    torch.cuda.synchronize()
    t_warmup_done = time.time()
    print("  Warmup done ({} steps, {:.1f}s).".format(
        BM_WARMUP, t_warmup_done - t_script_start))
    print("")

    # Measured windows
    # Wall-clock (sync at boundaries): real sustained throughput.
    # CUDA-event per step (sync per step): accurate latency, but serialized pipeline.
    window_tps  = []
    all_step_ms = []

    for w in range(BM_WINDOWS):

        if WALLTIME_MINUTES is not None:
            if time.time() - t_script_start > WALLTIME_MINUTES * 60 - MARGIN_SECONDS:
                print("[WALLTIME] Stopping benchmark at window {}.".format(w + 1))
                break

        step_offset    = BM_WARMUP + w * BM_WINDOW_STEPS
        step_ms_window = []

        torch.cuda.synchronize()
        t_w_start = time.time()

        for s in range(BM_WINDOW_STEPS):
            step = step_offset + s
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end   = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            train_step(model, optimizer, step)
            ev_end.record()
            torch.cuda.synchronize()   # needed to read event; serializes this step
            step_ms_window.append(ev_start.elapsed_time(ev_end))
            tokens_processed += tokens_per_step

        torch.cuda.synchronize()
        t_w_end = time.time()

        w_elapsed = t_w_end - t_w_start
        tps       = (BM_WINDOW_STEPS * tokens_per_step) / w_elapsed
        window_tps.append(tps)
        all_step_ms.extend(step_ms_window)

        step_ms_sorted = sorted(step_ms_window)
        med_ms = step_ms_sorted[len(step_ms_sorted) // 2]
        p95_ms = step_ms_sorted[int(0.95 * len(step_ms_sorted))]
        print("  Window {:d}/{:d}:  {:>10,.0f} tok/s  |  step med {:.1f}ms  p95 {:.1f}ms".format(
            w + 1, BM_WINDOWS, tps, med_ms, p95_ms))

    # Aggregate benchmark report
    if window_tps:
        mean_tps = sum(window_tps) / len(window_tps)
        min_tps  = min(window_tps)
        max_tps  = max(window_tps)
        var_pct  = (max_tps - min_tps) / mean_tps * 100 if mean_tps > 0 else 0.0

        all_sorted = sorted(all_step_ms)
        med_all    = all_sorted[len(all_sorted) // 2]
        p95_all    = all_sorted[int(0.95 * len(all_sorted))]

        peak_alloc_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        peak_resv_mib  = torch.cuda.max_memory_reserved()  / (1024.0 ** 2)

        ru     = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ram_mb = ru / 1024.0 if platform.system() == 'Linux' else ru / (1024.0 ** 2)

        torch.cuda.synchronize()
        t_eval_0  = time.time()
        bm_losses = estimate_loss(model)
        torch.cuda.synchronize()
        eval_sec  = time.time() - t_eval_0

        tmp_path  = os.path.join(checkpoint_dir, 'bm_probe.pt.tmp')
        ckpt_path = os.path.join(checkpoint_dir, 'bm_probe.pt')
        t_ckpt_0  = time.time()
        torch.save({'model': model.state_dict(), 'step': BM_TOTAL}, tmp_path)
        os.replace(tmp_path, ckpt_path)
        ckpt_sec     = time.time() - t_ckpt_0
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024.0 ** 2)
        os.remove(ckpt_path)

        print("")
        print("=" * 62)
        print("  BENCHMARK RESULTS — STAGE 4")
        print("=" * 62)
        print("  Throughput (window clock, sync at boundaries only):")
        print("    mean   {:>10,.0f} tok/s".format(mean_tps))
        print("    min    {:>10,.0f} tok/s".format(min_tps))
        print("    max    {:>10,.0f} tok/s".format(max_tps))
        print("    var    {:>10.1f}%  (max-min)/mean".format(var_pct))
        print("  Step latency (CUDA-event, synced per step — serialized):")
        print("    median  {:.2f} ms".format(med_all))
        print("    p95     {:.2f} ms".format(p95_all))
        print("  GPU memory:")
        print("    allocated  {:.1f} MiB".format(peak_alloc_mib))
        print("    reserved   {:.1f} MiB".format(peak_resv_mib))
        print("  Host RAM:          {:.1f} MB".format(ram_mb))
        print("  Eval ({:d}b x2):    {:.2f}s  train {:.4f}  val {:.4f}".format(
            eval_iters, eval_sec,
            bm_losses['train'].item(), bm_losses['val'].item()))
        print("  Checkpoint:        {:.2f}s  {:.1f} MB".format(ckpt_sec, ckpt_size_mb))
        print("  Startup to warmup: {:.1f}s".format(t_warmup_done - t_script_start))
        print("  Tokens processed:  {:,}".format(tokens_processed))
        # Estimated full-run time (excluding eval/checkpoint overhead)
        if mean_tps > 0:
            remaining_tokens = (max_iters - BM_TOTAL) * tokens_per_step
            est_train_sec = remaining_tokens / mean_tps
            print("  Est. remaining training time: {:.0f}s  ({:.1f} min)".format(
                est_train_sec, est_train_sec / 60))
        print("=" * 62)
        print("")

    # ════════════════════════════════════════════════════════════════════════
    #  TRAINING PHASE  (steps BM_TOTAL … max_iters-1)
    # ════════════════════════════════════════════════════════════════════════
    print("TRAINING:  steps {:d} → {:d}".format(BM_TOTAL, max_iters - 1))
    print("")

    for step in range(BM_TOTAL, max_iters):

        # Walltime exit — save latest.pt and exit before PBS SIGKILL
        if WALLTIME_MINUTES is not None:
            elapsed_sec = time.time() - t_script_start
            if elapsed_sec > WALLTIME_MINUTES * 60 - MARGIN_SECONDS:
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

        # Periodic evaluation and checkpointing
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model)
            lr_now = get_lr(step)

            # latest.pt — always updated (atomic replace)
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
            tmp_l = os.path.join(checkpoint_dir, 'latest.pt.tmp')
            torch.save(latest_ckpt, tmp_l)
            os.replace(tmp_l, os.path.join(checkpoint_dir, 'latest.pt'))

            # best.pt — only updated when val loss improves
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val'].item()
                tmp_b = os.path.join(checkpoint_dir, 'best.pt.tmp')
                torch.save(latest_ckpt, tmp_b)
                os.replace(tmp_b, os.path.join(checkpoint_dir, 'best.pt'))
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

        # Training step
        t0 = time.time()
        loss_val, grad_norm = train_step(model, optimizer, step)
        tokens_processed += tokens_per_step

        if step % eval_interval == eval_interval - 1:
            elapsed = time.time() - t0
            tps_approx = tokens_per_step / elapsed if elapsed > 0 else 0.0
            print("          grad_norm {:.3f} | approx {:.0f} tok/s "
                  "(single-step, no sync)".format(grad_norm, tps_approx))

    # ════════════════════════════════════════════════════════════════════════
    #  GENERATION — load best checkpoint, multi-prompt with temperature/top-k
    # ════════════════════════════════════════════════════════════════════════
    best_path = os.path.join(checkpoint_dir, 'best.pt')
    if os.path.exists(best_path):
        print("\nLoading Stage 4 best checkpoint (val loss: {:.4f})...".format(best_val_loss))
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    total_sec = time.time() - t_script_start
    print("\nTotal runtime: {:.1f}s ({:.1f} min)".format(total_sec, total_sec / 60))

    # Fixed prompts — use the same prompts when comparing Stage 3 vs Stage 4
    prompts = [
        ("Empty (newline)",   "\n"),
        ("King speaks",       "KING:\n"),
        ("Dialogue opener",   "ROMEO:\nWhat light through "),
        ("Verse start",       "To be, or not to be"),
    ]

    # Sampling settings — vary to show the temperature/top-k effect
    sampling_configs = [
        ("temp=0.7  top_k=50  (focused)",  dict(temperature=0.7,  top_k=50)),
        ("temp=1.0  top_k=None (standard)", dict(temperature=1.0,  top_k=None)),
        ("temp=1.2  top_k=None (creative)", dict(temperature=1.2,  top_k=None)),
    ]

    print("")
    print("=" * 62)
    print("  STAGE 4 GENERATED TEXT")
    print("  (Stage 3 best val: ~1.73 | Stage 4 target: lower)")
    print("=" * 62)

    for prompt_name, prompt_str in prompts:
        print("")
        print("  PROMPT: {}".format(prompt_name))
        print("  >>> {}".format(repr(prompt_str)))
        print("")

        for config_name, config_kwargs in sampling_configs:
            print("  -- {} --".format(config_name))
            try:
                generated = generate_from_prompt(
                    model, prompt_str, max_new_tokens=200, **config_kwargs)
                # Print with indentation for readability
                for line in generated.split('\n')[:15]:   # cap at 15 lines
                    print("  " + line)
            except Exception as e:
                print("  [Generation error: {}]".format(e))
            print("")

    print("=" * 62)
    print("")
    print("Stage 3 baseline val loss: ~1.73 (212K params, block_size=64)")
    print("Stage 4 final   val loss:  {:.4f} ({:,} params, block_size={})".format(
        best_val_loss, total_params, block_size))
    print("")
    print("To load Stage 4 model for further generation:")
    print("  ckpt = torch.load('{}', map_location='cuda')".format(best_path))
    print("  model.load_state_dict(ckpt['model'])")
