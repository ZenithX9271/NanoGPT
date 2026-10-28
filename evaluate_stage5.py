"""
evaluate_stage5.py — Stage 5 evaluation
-----------------------------------------
Token-weighted cross-entropy / perplexity on the held-out dev_val set.
Fixed-prompt generation with documented settings.
Saves full story outputs to a dated file.

Run on GPU node OR Mac (with CUDA fallback to CPU for evaluation only).
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join("configs", "stage5.json")

if not os.path.exists(CONFIG_PATH):
    sys.exit("ERROR: {} not found.".format(CONFIG_PATH))

with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

SHARD_DIR   = _cfg["data"].get("shard_dir", "data/shards/v2")
TOK_PATH    = _cfg["data"].get("tokenizer_path", "data/tokenizer/v2/tokenizer.json")
vocab_size  = _cfg["data"]["vocab_size"]
BOS_ID      = _cfg["data"]["bos_id"]
EOS_ID      = _cfg["data"]["eos_id"]
PAD_ID      = _cfg["data"]["pad_id"]

PROMPT_IDS_PATH = os.path.join("configs", "prompt_ids.json")

n_layer    = _cfg["model"]["n_layer"]
n_embd     = _cfg["model"]["n_embd"]
n_head     = _cfg["model"]["n_head"]
block_size = _cfg["model"]["block_size"]
dropout    = _cfg["model"]["dropout"]

checkpoint_dir = "checkpoints_s5v2"   # v2 run dir; override with --checkpoint

EVAL_BATCH  = 32    # batch size for loss evaluation
EVAL_CHUNKS = 500   # number of random windows to score

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    print("WARNING: Running on CPU. Evaluation will be slow.")
print("Eval device: {}".format(device))

# ── Import model classes from 5th_stage_gpt ───────────────────────────────────
# We re-define only what we need rather than importing the full training script
# (which would trigger the CUDA-required check).

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril",
            torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.0)   # no dropout at eval

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x);   q = self.query(x)
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj    = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(0.0)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(0.0))
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
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks   = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f     = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + \
            self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(x);  x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 top_k=None, greedy=False, eos_id=None):
        self.eval()
        stop_reason = "max_tokens"
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                assert temperature > 0
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and idx_next.item() == eos_id:
                stop_reason = "eos"; break
        return idx, stop_reason


# ── Byte-level decoder (same as in 5th_stage_gpt.py) ─────────────────────────

def _build_byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(0xa1, 0xad)) + list(range(0xae, 0x100)))
    cs = list(bs); n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

class SimpleBPEDecoder(object):
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        vocab = data["model"]["vocab"]
        self._id2tok = {int(v): k for k, v in vocab.items()}
        for t in data.get("added_tokens", []):
            self._id2tok[t["id"]] = t["content"]
        self._bd = _build_byte_decoder()

    def decode(self, ids, skip=()):
        merged = "".join(self._id2tok.get(i, "") for i in ids if i not in skip)
        try:
            return bytes([self._bd[c] for c in merged if c in self._bd]).decode("utf-8", errors="replace")
        except Exception:
            return merged


# ── Token-weighted CE evaluation ──────────────────────────────────────────────

import numpy as np

def token_weighted_ce(model, shard_dir, split, n_chunks, batch_size, block_size, device):
    """
    Compute token-weighted cross-entropy on `split`.
    Samples n_chunks random windows, each of length block_size.
    Loss is averaged over all scored tokens (not over chunks).

    This is NOT the same measure as Stage 4's character CE.
    Do NOT compare the numbers directly.
    """
    manifest_path = os.path.join(shard_dir, "manifest.json")
    with open(manifest_path) as f:
        mf = json.load(f)

    arrays = []
    for sm in mf["splits"][split]["shards"]:
        p = os.path.join(shard_dir, sm["file"])
        arrays.append(np.fromfile(p, dtype=np.uint16).astype(np.int64))
    data = np.concatenate(arrays)
    N = len(data)

    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_start in range(0, n_chunks, batch_size):
            b = min(batch_size, n_chunks - batch_start)
            ix = torch.randint(N - block_size, (b,))
            x = torch.stack([
                torch.from_numpy(data[i:i + block_size].copy()) for i in ix
            ]).to(device)
            y = torch.stack([
                torch.from_numpy(data[i + 1:i + block_size + 1].copy()) for i in ix
            ]).to(device)
            logits, loss = model(x, y)
            # Token-weighted: accumulate sum of log-probs
            total_loss   += loss.item() * b * block_size
            total_tokens += b * block_size

    ce = total_loss / total_tokens
    return ce, total_tokens


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 5 evaluation")
    parser.add_argument("--checkpoint", default=os.path.join(checkpoint_dir, "best.pt"),
                        help="Path to checkpoint (default: checkpoints_s5/best.pt)")
    parser.add_argument("--split", default="dev_val",
                        choices=["dev_val", "test"],
                        help="Evaluation split. Use 'test' only for FINAL evaluation.")
    parser.add_argument("--n_chunks", type=int, default=EVAL_CHUNKS,
                        help="Number of random windows to score")
    args = parser.parse_args()

    print("=" * 62)
    print("  STAGE 5 EVALUATION")
    print("=" * 62)
    print("  Checkpoint: {}".format(args.checkpoint))
    print("  Split:      {}".format(args.split))
    if args.split == "test":
        print("  WARNING: Using the test split. Reserve for final evaluation only.")
    print("")

    if not os.path.exists(args.checkpoint):
        sys.exit("ERROR: checkpoint not found: {}".format(args.checkpoint))

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = GPTLanguageModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    saved_step   = ckpt.get("step", "?")
    saved_tokens = ckpt.get("tokens", "?")
    saved_val_ce = ckpt.get("val_loss", "?")

    print("  Checkpoint step:   {}".format(saved_step))
    print("  Tokens processed:  {}".format(saved_tokens))
    print("  Saved val CE:      {}".format(
        "{:.4f}".format(saved_val_ce) if isinstance(saved_val_ce, float) else saved_val_ce))
    print("")

    # ── Token-weighted CE ─────────────────────────────────────────────────────
    print("[1] Token-weighted cross-entropy on {}".format(args.split))
    print("    {} random windows of {:d} tokens".format(args.n_chunks, block_size))
    torch.manual_seed(1337)
    t0 = time.time()
    ce, n_toks = token_weighted_ce(
        model, SHARD_DIR, args.split,
        n_chunks=args.n_chunks, batch_size=EVAL_BATCH,
        block_size=block_size, device=device)
    elapsed = time.time() - t0
    ppl = math.exp(min(ce, 20))

    print("    CE:         {:.4f}".format(ce))
    print("    Perplexity: {:.2f}".format(ppl))
    print("    Scored:     {:,} tokens in {:.1f}s".format(n_toks, elapsed))
    print("")
    print("  NOTE: This BPE CE is NOT comparable to Stage 4 character CE=1.5147.")
    print("  Tokenizer and data distribution both changed.")

    # ── Prompt IDs — loaded from pre-encoded file, same as 5th_stage_gpt.py ──────
    # prompt_ids.json is prepared on Mac with encode_prompts_mac.py.
    # This is the ONLY supported path on HPC where tokenizers library may be absent.
    # Fail loudly rather than silently conditioning on [BOS] alone.
    print("\n[2] Fixed-prompt generation")
    max_gen = _cfg.get("generation", {}).get("max_new_tokens", 300)

    if not os.path.exists(PROMPT_IDS_PATH):
        sys.exit(
            "ERROR: {} not found.\n"
            "Generate it on Mac:\n"
            "  python encode_prompts_mac.py\n"
            "Then commit and sync to HPC before evaluating.".format(PROMPT_IDS_PATH)
        )

    with open(PROMPT_IDS_PATH) as f:
        pid_data = json.load(f)

    prompts = pid_data.get("prompts", [])
    if not prompts:
        sys.exit("ERROR: {} contains no prompts.".format(PROMPT_IDS_PATH))

    # Verify each prompt starts with BOS
    for i, p in enumerate(prompts):
        if not p["ids"] or p["ids"][0] != BOS_ID:
            sys.exit(
                "ERROR: prompt {} in prompt_ids.json: first ID should be BOS={}.".format(
                    i, BOS_ID))

    print("  Loaded {:d} prompts from {}".format(len(prompts), PROMPT_IDS_PATH))
    for p in prompts:
        print("  {:d} tokens  {!r}".format(len(p["ids"]), p["text"][:60]))

    decoder = None
    if os.path.exists(TOK_PATH):
        decoder = SimpleBPEDecoder(TOK_PATH)

    sampling_configs = [
        ("greedy",            dict(greedy=True)),
        ("temp=0.7 top_k=50", dict(temperature=0.7, top_k=50)),
        ("temp=1.0",          dict(temperature=1.0)),
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(checkpoint_dir, "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, "stories_{}_step{}_{}.txt".format(
        args.split, saved_step, timestamp))

    lines = [
        "Stage 5 v2 generation | step={} | {} | CE={:.4f} ppl={:.2f}\n".format(
            saved_step, args.split, ce, ppl),
        "=" * 62 + "\n",
    ]

    for prompt in prompts:
        prompt_text = prompt["text"]
        prompt_ids  = prompt["ids"]   # already includes BOS as first token

        print("\n  PROMPT: {!r}  ({:d} tokens)".format(prompt_text[:60], len(prompt_ids)))
        print("  IDs: {}...".format(prompt_ids[:8]))
        lines.append("\nPROMPT: {!r}  ({:d} tokens)\n".format(prompt_text, len(prompt_ids)))
        lines.append("IDs: {}\n".format(prompt_ids[:10]))

        for cfg_name, cfg_kwargs in sampling_configs:
            torch.manual_seed(1337)

            ctx = torch.tensor(prompt_ids, dtype=torch.long,
                               device=device).unsqueeze(0)
            gen_toks, stop = model.generate(
                ctx, max_new_tokens=max_gen, eos_id=EOS_ID, **cfg_kwargs)

            out_ids = gen_toks[0].tolist()[len(prompt_ids):]
            if EOS_ID in out_ids:
                out_ids = out_ids[:out_ids.index(EOS_ID)]

            # Decode: skip PAD and BOS; EOS already stripped
            text = (decoder.decode(out_ids, skip=(PAD_ID, BOS_ID))
                    if decoder else str(out_ids[:20]))

            print("  -- {} stop={} {:d} tokens --".format(
                cfg_name, stop, len(out_ids)))
            for line in text.split("\n")[:8]:
                print("  " + line)

            lines.append("\n-- {} stop={} {:d} tokens --\n".format(
                cfg_name, stop, len(out_ids)))
            lines.append(text + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("\n  Stories saved → {}".format(out_path))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  EVALUATION SUMMARY")
    print("=" * 62)
    print("  Split:       {}".format(args.split))
    print("  CE:          {:.4f}".format(ce))
    print("  Perplexity:  {:.2f}".format(ppl))
    print("  Step:        {}".format(saved_step))
    print("  Tokens seen: {}".format(saved_tokens))
    print("  Output:      {}".format(out_path))
    print("=" * 62)


if __name__ == "__main__":
    main()
