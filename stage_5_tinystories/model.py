"""
5th_stage_gpt.py — NanoGPT Stage 5 v2: TinyStories + BPE
---------------------------------------------------------
6-layer, width-256 decoder-only Transformer on TinyStories.
Reads from data/shards/v2/ (JSONL-derived, v2 pipeline).
Preserves the v1 pilot checkpoint in checkpoints_s5_pilot/.

Key changes from v1 pilot:
  - Reads v2 JSONL-derived shards (data/shards/v2/)
  - Prompt encoding: explicit, never silent BOS-only fallback
    Load prompt_ids.json (prepared on Mac with encode_prompts_mac.py)
  - Generation: suppresses PAD and BOS; EOS stays for stopping
  - 100M-token schedule (max_iters=12207, warmup=500)
  - Checkpoints saved to checkpoints_s5v2/ (preserves pilot)
  - Progress snapshots at 10M, 25M, 50M, 100M tokens

Architecture unchanged (same as v1 pilot):
  n_layer=6, n_embd=256, n_head=8, block_size=256,
  dropout=0.2, FP32, manual causal attention, ReLU FFN,
  Pre-LN, learned position embeddings, weight tying.

RUN ON GPU NODE. Python 3.7, torch 1.10 compatible.
"""

import json
import math
import os
import platform
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Project root (two levels up from stage_5_tinystories/model.py) ───────────
# All data, checkpoint, and config paths are anchored to the repo root so the
# script works regardless of which directory you launch it from.
_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)   # NanoGPT/

def _root(*parts):
    """Join path parts relative to the project root."""
    return os.path.join(_PROJ_ROOT, *parts)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = _root("stage_5_tinystories", "configs", "stage5.json")
if not os.path.exists(CONFIG_PATH):
    sys.exit("ERROR: {} not found.".format(CONFIG_PATH))
with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

# Model (same architecture as v1 pilot — no architecture changes)
n_layer    = _cfg["model"]["n_layer"]          # 6
n_embd     = _cfg["model"]["n_embd"]           # 256
n_head     = _cfg["model"]["n_head"]           # 8
block_size = _cfg["model"]["block_size"]       # 256
dropout    = _cfg["model"]["dropout"]          # 0.2

# Training — 100M token schedule
batch_size    = _cfg["training"]["batch_size"]    # 32
max_iters     = _cfg["training"]["max_iters"]     # 12207 ≈ 100M tokens
eval_interval = _cfg["training"]["eval_interval"] # 500
eval_iters    = _cfg["training"]["eval_iters"]    # 200
learning_rate = _cfg["training"]["learning_rate"] # 3e-4
warmup_iters  = _cfg["training"].get("warmup_iters_v2", 500)  # v2: longer warmup
min_lr        = _cfg["training"]["min_lr"]        # 3e-5
grad_clip     = _cfg["training"]["grad_clip"]     # 1.0
weight_decay  = _cfg["training"]["weight_decay"]  # 0.1
betas         = tuple(_cfg["training"]["betas"])  # (0.9, 0.95)
SEED          = _cfg["training"]["seed"]          # 1337

# Data (v2 paths)
SHARD_DIR  = _root("data", "shards", "v2")
TOK_PATH   = _root("data", "tokenizer", "v2", "tokenizer.json")
vocab_size = _cfg["data"]["vocab_size"]           # 8192
PAD_ID     = _cfg["data"]["pad_id"]               # 0
BOS_ID     = _cfg["data"]["bos_id"]               # 1
EOS_ID     = _cfg["data"]["eos_id"]               # 2

# Prompt IDs file (prepared on Mac with pipeline/encode_prompts_mac.py)
PROMPT_IDS_PATH = _root("stage_5_tinystories", "configs", "prompt_ids.json")

# Benchmark
BM_WARMUP       = _cfg["benchmark"]["warmup_steps"]   # 50
BM_WINDOWS      = _cfg["benchmark"]["windows"]        # 5
BM_WINDOW_STEPS = _cfg["benchmark"]["window_steps"]   # 100
BM_TOTAL        = BM_WARMUP + BM_WINDOWS * BM_WINDOW_STEPS

# Checkpointing (saved to stage_5_tinystories/checkpoints/; best model mirrored to root best_model.pt)
checkpoint_dir   = _root("stage_5_tinystories", "checkpoints")
WALLTIME_MINUTES = _cfg["checkpointing"]["walltime_minutes"]
MARGIN_SECONDS   = _cfg["checkpointing"]["margin_seconds"]

# Snapshot steps: save + generate at these token counts
SNAPSHOT_TOKENS = [10_000_000, 25_000_000, 50_000_000, 100_000_000]

torch.manual_seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
# Allow CPU when imported by tests; enforce CUDA when run directly.
# The hardware assertions and smoke test run inside main() / __name__ guard below.
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Pure-Python BPE decoder ───────────────────────────────────────────────────
# Needed for generation on the cluster where tokenizers library may not be installed.

def _build_byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xad")))
          + list(range(ord("\xae"), 0x100)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class SimpleBPEDecoder(object):
    """Decode token IDs to text using tokenizer.json (no tokenizers library)."""

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        vocab = data["model"]["vocab"]
        self._id2tok = {int(v): k for k, v in vocab.items()}
        for t in data.get("added_tokens", []):
            self._id2tok[t["id"]] = t["content"]
        self._bd = _build_byte_decoder()

    def decode(self, ids, skip=()):
        """Decode IDs to text. IDs in `skip` are silently dropped."""
        merged = "".join(self._id2tok.get(i, "") for i in ids if i not in skip)
        try:
            return bytes([self._bd[c] for c in merged
                          if c in self._bd]).decode("utf-8", errors="replace")
        except Exception:
            return merged


# ── Data loading ──────────────────────────────────────────────────────────────

class ShardedDataset(object):
    """Memory-maps v2 uint16 shards and presents a flat token ID array."""

    def __init__(self, split):
        mf_path = os.path.join(SHARD_DIR, "manifest.json")
        if not os.path.exists(mf_path):
            sys.exit(
                "ERROR: {} not found.\n"
                "Run build_token_shards.py and sync data/shards/v2/ to HPC.".format(mf_path))
        with open(mf_path) as f:
            mf = json.load(f)
        if split not in mf["splits"]:
            sys.exit("ERROR: split '{}' not in manifest.".format(split))

        arrays = []
        for sm in mf["splits"][split]["shards"]:
            p = os.path.join(SHARD_DIR, sm["file"])
            if not os.path.exists(p):
                sys.exit("ERROR: shard {} not found.".format(p))
            arrays.append(np.fromfile(p, dtype=np.uint16).astype(np.int64))

        self.data = np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)
        self.n = len(self.data)
        print("  {} split: {:,} tokens ({:.1f}M) in {:d} shard(s)".format(
            split, self.n, self.n / 1e6, len(arrays)))

    def get_batch(self, batch_size, block_size, device):
        ix = torch.randint(self.n - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i + block_size].copy()) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + block_size + 1].copy()) for i in ix])
        return x.to(device), y.to(device)


# ── Loss estimation ───────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, datasets):
    out = {}
    model.eval()
    for split, ds in datasets.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = ds.get_batch(batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ════════════════════════════════════════════════════════════════════════════
#  MODEL  (identical to v1 pilot — architecture unchanged)
# ════════════════════════════════════════════════════════════════════════════

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril",
            torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei  = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei  = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei  = F.softmax(wei, dim=-1)
        wei  = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj    = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], -1)))


class FeedForward(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        hs = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, hs)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Stage 5 GPT. Identical architecture to v1 pilot.
    Weight tying: lm_head.weight = token_embedding_table.weight.
    """

    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks   = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f     = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight   # tie
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
        x = (self.token_embedding_table(idx)
             + self.position_embedding_table(torch.arange(T, device=idx.device)))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 top_k=None, greedy=False, eos_id=None):
        """
        greedy=True : explicit argmax (never temperature=0 trick).
        eos_id      : stop when this token is produced.
        temperature : must be > 0 when greedy=False.

        SUPPRESSES PAD and BOS from being sampled.
        EOS is NOT suppressed — it is used as a stopping signal.
        """
        assert greedy or temperature > 0, "temperature must be > 0"
        self.eval()
        stop_reason = "max_tokens"

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]   # (B, vocab_size)

            # Suppress PAD and BOS (never generate them)
            logits[:, PAD_ID] = float("-inf")
            logits[:, BOS_ID] = float("-inf")

            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and idx_next.item() == eos_id:
                stop_reason = "eos"
                break

        self.train()
        return idx, stop_reason


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step):
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    if step >= max_iters:
        return min_lr
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(model, optimizer, step, datasets):
    for pg in optimizer.param_groups:
        pg["lr"] = get_lr(step)
    xb, yb = datasets["train"].get_batch(batch_size, block_size, device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = 0.0
    if grad_clip > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip).item()
    optimizer.step()
    return loss.item(), grad_norm


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def make_ckpt(model, optimizer, step, val_loss, tokens):
    return {
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss":  val_loss,
        "tokens":    tokens,
        "config": {
            "version":    "v2",
            "vocab_size": vocab_size,
            "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
            "block_size": block_size, "dropout": dropout,
            "weight_tying": True,
            "bos_id": BOS_ID, "eos_id": EOS_ID,
        },
    }


def atomic_save(ckpt, path):
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


# ── Prompt loading (explicit, no silent fallback) ─────────────────────────────

def load_prompt_ids():
    """
    Load pre-encoded prompt token IDs from configs/prompt_ids.json.
    This file must be prepared on Mac using encode_prompts_mac.py BEFORE
    running this script on HPC.

    Format:
      {"prompts": [
        {"text": "Once upon a time...", "ids": [1, 123, 456, ...]},
        ...
      ]}

    Raises RuntimeError if the file is missing or malformed.
    """
    if not os.path.exists(PROMPT_IDS_PATH):
        raise RuntimeError(
            "ERROR: {} not found.\n"
            "Generate it on Mac before the HPC run:\n"
            "  python encode_prompts_mac.py\n"
            "Then sync to HPC:\n"
            "  scp configs/prompt_ids.json <user>@hpc:/path/NanoGPT/configs/".format(
                PROMPT_IDS_PATH)
        )

    with open(PROMPT_IDS_PATH) as f:
        data = json.load(f)

    prompts = data.get("prompts", [])
    if not prompts:
        raise RuntimeError("prompt_ids.json contains no prompts.")

    # Verify each prompt has a non-empty ids list starting with BOS
    for i, p in enumerate(prompts):
        if "ids" not in p or not p["ids"]:
            raise RuntimeError(
                "prompt_ids.json entry {} has no ids field.".format(i))
        if p["ids"][0] != BOS_ID:
            raise RuntimeError(
                "prompt_ids.json entry {}: first ID should be BOS={}, got {}.".format(
                    i, BOS_ID, p["ids"][0]))

    print("Loaded {:d} prompt(s) from {}".format(len(prompts), PROMPT_IDS_PATH))
    for p in prompts:
        print("  {:d} tokens  {!r}".format(len(p["ids"]), p["text"][:60]))

    return prompts


# ── Generation ────────────────────────────────────────────────────────────────

def run_generation(model, prompts, decoder, max_new_tokens, tag, out_lines):
    """
    Run generation from all prompts with fixed sampling configs.
    Prints to console (trimmed) and appends full output to out_lines.
    """
    sampling_configs = [
        ("greedy",            dict(greedy=True)),
        ("temp=0.7 top_k=50", dict(temperature=0.7, top_k=50)),
        ("temp=1.0",          dict(temperature=1.0)),
    ]

    out_lines.append("\n{}\n".format("=" * 62))
    out_lines.append("  GENERATION @ {}\n".format(tag))
    out_lines.append("{}\n".format("=" * 62))
    print("\n" + "=" * 62)
    print("  GENERATION @ {}".format(tag))
    print("=" * 62)

    for prompt in prompts:
        prompt_text = prompt["text"]
        prompt_ids  = prompt["ids"]

        out_lines.append("\nPROMPT: {!r}  ({:d} tokens)\n".format(
            prompt_text, len(prompt_ids)))
        out_lines.append("IDs: {}\n".format(prompt_ids[:10]))
        print("\n  PROMPT: {!r}  ({:d} tokens)".format(prompt_text[:60], len(prompt_ids)))
        print("  IDs: {}...".format(prompt_ids[:8]))

        for cfg_name, cfg_kwargs in sampling_configs:
            torch.manual_seed(SEED)
            ctx = torch.tensor(prompt_ids, dtype=torch.long,
                               device=device).unsqueeze(0)

            gen_toks, stop = model.generate(
                ctx, max_new_tokens=max_new_tokens, eos_id=EOS_ID, **cfg_kwargs)

            out_ids = gen_toks[0].tolist()[len(prompt_ids):]
            if EOS_ID in out_ids:
                out_ids = out_ids[:out_ids.index(EOS_ID)]

            # Decode: skip PAD and BOS; EOS already stripped above
            text = (decoder.decode(out_ids, skip=(PAD_ID, BOS_ID))
                    if decoder else str(out_ids[:20]))

            out_lines.append("\n-- {} stop={} {:d} new tokens --\n".format(
                cfg_name, stop, len(out_ids)))
            out_lines.append(text + "\n")

            print("\n  -- {} stop={} {:d} tokens --".format(
                cfg_name, stop, len(out_ids)))
            for line in text.split("\n")[:8]:
                print("  " + line)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Hardware validation (runs only when executed directly, not on import) ──
    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: CUDA unavailable. Load the correct module before running.\n"
            "IITD HPC: module load apps/pytorch/1.10.0/gpu/intelpython3.7"
        )
    # Smoke test — catches compute-capability mismatch before any training
    try:
        _p = torch.zeros(4, 4, device="cuda") @ torch.zeros(4, 4, device="cuda")
        del _p
    except RuntimeError as _e:
        sys.exit("ERROR: CUDA forward test failed: {}".format(_e))
    # Bind device to cuda for the rest of this run
    device = "cuda"
    print("Device: cuda  |  GPU: {}".format(torch.cuda.get_device_name(0)))

    t_start = time.time()

    # ── Load prompt IDs (fail loudly if missing) ──────────────────────────────
    prompts = load_prompt_ids()

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    datasets = {
        "train":   ShardedDataset("train"),
        "dev_val": ShardedDataset("dev_val"),
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPTLanguageModel().to(device)

    # Deduplicated param groups (weight tying: shared weight counted once)
    seen_ids = set()
    unique_params = [(n, p) for n, p in model.named_parameters()
                     if id(p) not in seen_ids and not seen_ids.add(id(p))]
    total_params  = sum(p.numel() for _, p in unique_params)
    decay_params   = [p for _, p in unique_params if p.dim() >= 2]
    nodecay_params = [p for _, p in unique_params if p.dim() <  2]

    assert model.lm_head.weight is model.token_embedding_table.weight, \
        "Weight tying failed!"

    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=learning_rate, betas=betas,
    )

    # ── Decoder ───────────────────────────────────────────────────────────────
    decoder = None
    if os.path.exists(TOK_PATH):
        try:
            decoder = SimpleBPEDecoder(TOK_PATH)
            print("Decoder loaded (pure Python).")
        except Exception as e:
            print("WARNING: decoder init failed: {}".format(e))
    else:
        print("WARNING: {} not found — generation will show raw IDs.".format(TOK_PATH))

    # ── Environment ───────────────────────────────────────────────────────────
    tokens_per_step = batch_size * block_size   # 8,192

    print("")
    print("=" * 62)
    print("  STAGE 5 v2 — ENVIRONMENT")
    print("=" * 62)
    print("  GPU:      {}".format(torch.cuda.get_device_name(0)))
    print("  CUDA:     {}".format(torch.version.cuda))
    print("  PyTorch:  {}".format(torch.__version__))
    print("  Python:   {}".format(sys.version.split()[0]))
    print("  Params:   {:,} unique (tied weight counted once)".format(total_params))
    print("  Decay:    {:,}  |  No-decay: {:,}".format(
        sum(p.numel() for p in decay_params),
        sum(p.numel() for p in nodecay_params)))
    print("  Config:   lay={} embd={} head={} ctx={} drop={}".format(
        n_layer, n_embd, n_head, block_size, dropout))
    print("  Vocab:    {:,}  BOS={} EOS={} PAD={}".format(
        vocab_size, BOS_ID, EOS_ID, PAD_ID))
    print("  Schedule: {:,} iters × {:,} tok/step = {:,.0f}M BPE tokens".format(
        max_iters, tokens_per_step, max_iters * tokens_per_step / 1e6))
    print("  Warmup:   {:,} steps".format(warmup_iters))
    print("  Snapshot: {}".format(SNAPSHOT_TOKENS))
    print("=" * 62)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss    = float("inf")
    tokens_processed = 0
    gen_output_lines = []
    next_snapshot    = 0   # index into SNAPSHOT_TOKENS

    # ════════════════════════════════════════════════════════════════════════
    #  BENCHMARK
    # ════════════════════════════════════════════════════════════════════════
    print("\nBENCHMARK: {} warmup + {} × {} steps = {} total".format(
        BM_WARMUP, BM_WINDOWS, BM_WINDOW_STEPS, BM_TOTAL))

    torch.cuda.reset_peak_memory_stats()

    for step in range(BM_WARMUP):
        train_step(model, optimizer, step, datasets)
        tokens_processed += tokens_per_step

    torch.cuda.synchronize()
    t_wdone = time.time()
    print("  Warmup done ({} steps, {:.1f}s).".format(
        BM_WARMUP, t_wdone - t_start))

    window_tps, all_step_ms = [], []

    for w in range(BM_WINDOWS):
        if WALLTIME_MINUTES is not None:
            if time.time() - t_start > WALLTIME_MINUTES * 60 - MARGIN_SECONDS:
                print("[WALLTIME] Stopping benchmark."); break

        step_offset = BM_WARMUP + w * BM_WINDOW_STEPS
        step_ms_w   = []

        torch.cuda.synchronize()
        t_ws = time.time()

        for s in range(BM_WINDOW_STEPS):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            train_step(model, optimizer, step_offset + s, datasets)
            ev_e.record()
            torch.cuda.synchronize()
            step_ms_w.append(ev_s.elapsed_time(ev_e))
            tokens_processed += tokens_per_step

        torch.cuda.synchronize()
        tps = BM_WINDOW_STEPS * tokens_per_step / (time.time() - t_ws)
        window_tps.append(tps)
        all_step_ms.extend(step_ms_w)

        ms_s = sorted(step_ms_w)
        print("  Window {:d}/{:d}:  {:>10,.0f} tok/s  |  med {:.1f}ms  p95 {:.1f}ms".format(
            w + 1, BM_WINDOWS, tps,
            ms_s[len(ms_s) // 2], ms_s[int(0.95 * len(ms_s))]))

    if window_tps:
        mean_tps = sum(window_tps) / len(window_tps)
        all_s    = sorted(all_step_ms)
        pmib     = torch.cuda.max_memory_allocated() / 1024.0 ** 2
        rmib     = torch.cuda.max_memory_reserved()  / 1024.0 ** 2

        torch.cuda.synchronize()
        t_ev = time.time()
        ev_losses = estimate_loss(model, {"dev_val": datasets["dev_val"]})
        eval_sec  = time.time() - t_ev

        tmp_probe = os.path.join(checkpoint_dir, "bm_probe.pt.tmp")
        probe_path = os.path.join(checkpoint_dir, "bm_probe.pt")
        t_ck = time.time()
        torch.save({"step": BM_TOTAL, "model": model.state_dict()}, tmp_probe)
        os.replace(tmp_probe, probe_path)
        ckpt_sec = time.time() - t_ck
        ckpt_mb  = os.path.getsize(probe_path) / 1024.0 ** 2
        os.remove(probe_path)

        remain_tok = (max_iters - BM_TOTAL) * tokens_per_step
        est_sec    = remain_tok / mean_tps if mean_tps > 0 else 0

        print("")
        print("=" * 62)
        print("  BENCHMARK RESULTS — STAGE 5 v2")
        print("=" * 62)
        print("  Throughput:  mean {:>10,.0f} tok/s  min {:,.0f}  max {:,.0f}".format(
            mean_tps, min(window_tps), max(window_tps)))
        print("  Latency:     median {:.2f}ms  p95 {:.2f}ms  (serialized)".format(
            all_s[len(all_s) // 2], all_s[int(0.95 * len(all_s))]))
        print("  GPU:         alloc {:.0f} MiB  reserved {:.0f} MiB".format(pmib, rmib))
        print("  Eval:        {:.2f}s  dev_val CE={:.4f}  ppl={:.2f}".format(
            eval_sec, ev_losses["dev_val"],
            math.exp(min(ev_losses["dev_val"], 20))))
        print("  Checkpoint:  {:.2f}s  {:.1f} MB".format(ckpt_sec, ckpt_mb))
        print("  Est. remaining: {:.0f}s ({:.1f}h, {:,} steps)".format(
            est_sec, est_sec / 3600, max_iters - BM_TOTAL))
        print("=" * 62)
        print("")
        print("  Use estimated time to choose PBS walltime.")
        print("  SNAPSHOT checkpoints will be saved at: {}".format(SNAPSHOT_TOKENS))

    # ════════════════════════════════════════════════════════════════════════
    #  TRAINING
    # ════════════════════════════════════════════════════════════════════════
    print("\nTRAINING: steps {:d} → {:d}".format(BM_TOTAL, max_iters - 1))

    for step in range(BM_TOTAL, max_iters):

        # Walltime exit
        if WALLTIME_MINUTES is not None:
            if time.time() - t_start > WALLTIME_MINUTES * 60 - MARGIN_SECONDS:
                ckpt = make_ckpt(model, optimizer, step, best_val_loss, tokens_processed)
                atomic_save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))
                print("[WALLTIME] Saved latest.pt at step {:d}. Exiting.".format(step))
                sys.exit(0)

        # Snapshot at 10M / 25M / 50M / 100M tokens
        if (next_snapshot < len(SNAPSHOT_TOKENS)
                and tokens_processed >= SNAPSHOT_TOKENS[next_snapshot]):
            snap_M = SNAPSHOT_TOKENS[next_snapshot] // 1_000_000
            tag    = "{}M tokens".format(snap_M)
            print("\n[SNAPSHOT] {}".format(tag))

            losses = estimate_loss(model, datasets)
            ckpt   = make_ckpt(model, optimizer, step,
                               losses["dev_val"], tokens_processed)
            snap_path = os.path.join(checkpoint_dir,
                                     "snapshot_{}M.pt".format(snap_M))
            atomic_save(ckpt, snap_path)
            print("  train CE {:.4f}  dev_val CE {:.4f}  ppl {:.2f}".format(
                losses["train"], losses["dev_val"],
                math.exp(min(losses["dev_val"], 20))))
            print("  Checkpoint → {}".format(snap_path))

            run_generation(model, prompts, decoder, 300, tag, gen_output_lines)
            next_snapshot += 1

        # Periodic eval
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, datasets)
            val_ce = losses["dev_val"]
            lr_now = get_lr(step)

            ckpt = make_ckpt(model, optimizer, step, val_ce, tokens_processed)
            atomic_save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))

            if val_ce < best_val_loss:
                best_val_loss = val_ce
                atomic_save(ckpt, os.path.join(checkpoint_dir, "best.pt"))
                try:
                    atomic_save(ckpt, _root("best_model.pt"))
                except Exception:
                    pass  # Non-critical if root is read-only
                tag = " <- best"
            else:
                tag = ""

            print("  step {:5d}: train {:.4f} | val CE {:.4f} | ppl {:.2f} | "
                  "lr {:.2e} | tokens {:,}{}".format(
                      step, losses["train"], val_ce,
                      math.exp(min(val_ce, 20)), lr_now, tokens_processed, tag))

        t0 = time.time()
        _, grad_norm = train_step(model, optimizer, step, datasets)
        tokens_processed += tokens_per_step

        if step % eval_interval == eval_interval - 1:
            tps_approx = tokens_per_step / max(time.time() - t0, 1e-9)
            print("          grad_norm {:.3f} | ~{:.0f} tok/s".format(
                grad_norm, tps_approx))

    # ── Post-loop: 100M snapshot (fires if training crossed 100M after the last step) ──
    # The snapshot trigger inside the loop fires at the TOP of each iteration,
    # so if tokens_processed crossed 100M on the very last step the loop ends
    # without capturing it. Catch it here.
    if (next_snapshot < len(SNAPSHOT_TOKENS)
            and tokens_processed >= SNAPSHOT_TOKENS[next_snapshot]):
        snap_M = SNAPSHOT_TOKENS[next_snapshot] // 1_000_000
        tag    = "{}M tokens (post-loop)".format(snap_M)
        print("\n[SNAPSHOT] {}".format(tag))
        losses = estimate_loss(model, datasets)
        ckpt   = make_ckpt(model, optimizer, max_iters - 1,
                           losses["dev_val"], tokens_processed)
        snap_path = os.path.join(checkpoint_dir, "snapshot_{}M.pt".format(snap_M))
        atomic_save(ckpt, snap_path)
        print("  train CE {:.4f}  dev_val CE {:.4f}  ppl {:.2f}".format(
            losses["train"], losses["dev_val"],
            math.exp(min(losses["dev_val"], 20))))
        print("  Checkpoint → {}".format(snap_path))
        run_generation(model, prompts, decoder, 300, tag, gen_output_lines)
        next_snapshot += 1

    # ── Save latest.pt BEFORE reloading best.pt for generation ───────────────
    # Must happen here — after best.pt is reloaded the model state no longer
    # reflects the true final training weights.
    latest_ckpt = make_ckpt(model, optimizer, max_iters - 1,
                            best_val_loss, tokens_processed)
    atomic_save(latest_ckpt, os.path.join(checkpoint_dir, "latest.pt"))
    print("latest.pt saved (true final weights, step={}, tokens={:,}).".format(
        max_iters - 1, tokens_processed))

    # ── Final generation ──────────────────────────────────────────────────────
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        assert model.lm_head.weight is model.token_embedding_table.weight, \
            "Weight tying broken after reload!"
        print("\nLoaded best.pt (val CE: {:.4f})".format(best_val_loss))

    total_sec = time.time() - t_start
    print("Total runtime: {:.1f}s ({:.1f}h)".format(total_sec, total_sec / 3600))

    run_generation(model, prompts, decoder, 300, "FINAL (best.pt)", gen_output_lines)

    # ── Save all generation output ────────────────────────────────────────────
    gen_path = os.path.join(checkpoint_dir, "generation_all_stages.txt")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write("Stage 5 v2 | best val CE: {:.4f}\n".format(best_val_loss))
        f.writelines(gen_output_lines)
    print("\nAll generation output → {}".format(gen_path))
    print("\nTo evaluate: python evaluate_stage5.py --checkpoint {}/best.pt".format(
        checkpoint_dir))
