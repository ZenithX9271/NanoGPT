"""
train_tokenizer.py — Stage 5 v2: BPE tokenizer on JSONL data
-------------------------------------------------------------
Trains an 8,192-vocabulary byte-level BPE tokenizer on TinyStories v2
JSONL training stories.

RUN ON MAC. Requires: pip install tokenizers==0.20.3

INPUT:  data/tinystories/v2/train.jsonl
OUTPUT: data/tokenizer/v2/tokenizer.json
        data/tokenizer/v2/vocab_info.json

Usage:
  python train_tokenizer.py                     # default paths
  python train_tokenizer.py --data-dir data/tinystories/v2
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone

VOCAB_SIZE    = 8192
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",  default=_root("data", "tinystories", "v2"))
parser.add_argument("--out-dir",   default=_root("data", "tokenizer", "v2"))
args, _ = parser.parse_known_args()

TRAIN_JSONL = os.path.join(args.data_dir, "train.jsonl")
OUT_DIR     = args.out_dir
TOK_OUT     = os.path.join(OUT_DIR, "tokenizer.json")
INFO_OUT    = os.path.join(OUT_DIR, "vocab_info.json")

# ── Check tokenizers ──────────────────────────────────────────────────────────
try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    import tokenizers as _tk
    print("tokenizers version: {}".format(_tk.__version__))
except ImportError:
    sys.exit("ERROR: pip install tokenizers==0.20.3")

# ── Helpers ───────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl_stories(path):
    stories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stories.append(json.loads(line)["text"])
    return stories


def story_iterator(stories):
    for s in stories:
        yield s


def analyse_lengths(tokenizer, stories, sample=5000):
    rng = random.Random(1337)
    sample_stories = rng.sample(stories, min(sample, len(stories)))
    lengths = sorted(len(tokenizer.encode(s).ids) + 2 for s in sample_stories)
    n = len(lengths)
    return {
        "sample_size":  n,
        "median":       lengths[n // 2],
        "p90":          lengths[int(0.90 * n)],
        "p95":          lengths[int(0.95 * n)],
        "max":          lengths[-1],
        "frac_gt_256":  sum(1 for l in lengths if l > 256) / n,
        "frac_gt_512":  sum(1 for l in lengths if l > 512) / n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Byte-level BPE tokenizer training v2")
    print("=" * 60)

    if not os.path.exists(TRAIN_JSONL):
        sys.exit(
            "ERROR: {} not found.\n"
            "Run: python prepare_tinystories.py".format(TRAIN_JSONL))

    print("\n[1] Loading training stories from {}".format(TRAIN_JSONL))
    stories = load_jsonl_stories(TRAIN_JSONL)
    print("  {:,} stories loaded".format(len(stories)))

    # ── Build tokenizer ───────────────────────────────────────────────────────
    print("\n[2] Initialising byte-level BPE tokenizer")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print("\n[3] Training ({:,} stories, target vocab {:,})...".format(
        len(stories), VOCAB_SIZE))
    tokenizer.train_from_iterator(story_iterator(stories), trainer=trainer)
    actual_vs = tokenizer.get_vocab_size()
    print("  Done. Actual vocab size: {:,}".format(actual_vs))

    # ── Verify special IDs ────────────────────────────────────────────────────
    print("\n[4] Special token IDs")
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    print("  <pad>={} <bos>={} <eos>={}".format(pad_id, bos_id, eos_id))
    assert pad_id == 0 and bos_id == 1 and eos_id == 2, \
        "Expected 0,1,2 — got {},{},{}".format(pad_id, bos_id, eos_id)
    assert actual_vs <= VOCAB_SIZE, "vocab {} > {}".format(actual_vs, VOCAB_SIZE)

    # ── Round-trips ───────────────────────────────────────────────────────────
    print("\n[5] Round-trip checks")
    test_strings = [
        "Once upon a time, a little girl named Lily found a red apple.",
        "Hello\nworld! Numbers: 123. Emoji: \U0001f600",
        'She said, "I don\'t know how to do it."',
        "  Leading and trailing spaces   ",
        "Accents: café, naïve, résumé.",
    ]
    all_ok = True
    for text in test_strings:
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        ok = (dec == text)
        if not ok:
            all_ok = False
        print("  {} {!r} ({} tokens)".format(
            "OK  " if ok else "FAIL", text[:40], len(enc.ids)))

    # Max ID check
    long = " ".join(test_strings)
    max_id = max(tokenizer.encode(long).ids)
    assert max_id < actual_vs
    print("  max token ID in test: {} (vocab {})".format(max_id, actual_vs))

    # ── Save / reload determinism ─────────────────────────────────────────────
    print("\n[6] Save / reload determinism")
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer.save(TOK_OUT)
    from tokenizers import Tokenizer as _T2

# ── Project root ──────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_STAGE_DIR = os.path.dirname(_HERE)          # stage_5_tinystories/
_PROJ_ROOT = os.path.dirname(_STAGE_DIR)     # NanoGPT/

def _root(*parts):
    """Join path parts relative to the project root."""
    return os.path.join(_PROJ_ROOT, *parts)

    t2 = _T2.from_file(TOK_OUT)
    sample = "Once upon a time, in a land far away, there lived a brave knight."
    assert tokenizer.encode(sample).ids == t2.encode(sample).ids, \
        "Reload changed encoding"
    print("  OK — encoding unchanged after save/reload")

    # ── Length analysis ───────────────────────────────────────────────────────
    print("\n[7] Story length analysis (sample {:,}, incl. BOS+EOS)".format(5000))
    stats = analyse_lengths(tokenizer, stories)
    for k, v in stats.items():
        if isinstance(v, float):
            print("  {:20s}: {:.3f}".format(k, v))
        else:
            print("  {:20s}: {}".format(k, v))
    print("  (median={:d}, p90={:d}, p95={:d}, max={:d})".format(
        stats["median"], stats["p90"], stats["p95"], stats["max"]))
    print("  {:.1f}% stories > 256 tokens, {:.1f}% > 512 tokens".format(
        100 * stats["frac_gt_256"], 100 * stats["frac_gt_512"]))

    # ── Save info ─────────────────────────────────────────────────────────────
    info = {
        "version":          "v2",
        "trained_utc":      datetime.now(timezone.utc).isoformat(),
        "tokenizer_lib":    "tokenizers=={}".format(_tk.__version__),
        "vocab_size":       actual_vs,
        "requested_vocab":  VOCAB_SIZE,
        "special_tokens":   {"<pad>": pad_id, "<bos>": bos_id, "<eos>": eos_id},
        "training_source":  TRAIN_JSONL,
        "training_stories": len(stories),
        "tokenizer_sha256": sha256_file(TOK_OUT),
        "round_trips_passed": all_ok,
        "length_stats":     stats,
        "pre_tokenizer":    "ByteLevel(add_prefix_space=False, use_regex=True)",
        "trainer":          "BpeTrainer(vocab_size={})".format(VOCAB_SIZE),
        "attention_policy": (
            "Stories concatenated with BOS/EOS. "
            "Causal attention may cross story boundaries. "
            "Document offsets stored for future isolation."
        ),
    }
    with open(INFO_OUT, "w") as f:
        json.dump(info, f, indent=2)
    print("\n  tokenizer.json → {}".format(TOK_OUT))
    print("  vocab_info.json → {}".format(INFO_OUT))

    print("\n" + "=" * 60)
    print("  Tokenizer ready. vocab_size={}".format(actual_vs))
    print("=" * 60)
    print("\nNext: python build_token_shards.py")


if __name__ == "__main__":
    main()
