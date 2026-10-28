"""
build_token_shards.py — Stage 5 v2: JSONL → uint16 binary shards
-----------------------------------------------------------------
Encodes TinyStories v2 JSONL stories as [BOS] tokens [EOS] uint16 sequences
and writes binary shards with a manifest.

RUN ON MAC. Requires: pip install tokenizers==0.20.3  numpy

INPUT:  data/tinystories/v2/{train,dev_val,test}.jsonl
        data/tokenizer/v2/tokenizer.json
OUTPUT: data/shards/v2/{train_NNNN.bin, dev_val.bin, test.bin}
        data/shards/v2/{*_offsets.json, manifest.json}

Each shard: flat little-endian uint16. No header.
Each story: [BOS_ID] story_tokens... [EOS_ID]
Stories are concatenated; boundaries are in *_offsets.json.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# ── Paths (v2) ────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join("data", "tinystories", "v2")
TOK_PATH     = os.path.join("data", "tokenizer", "v2", "tokenizer.json")
TOK_INFO     = os.path.join("data", "tokenizer", "v2", "vocab_info.json")
SHARD_DIR    = os.path.join("data", "shards", "v2")

SHARD_TOKENS = 10_000_000   # ~20 MB per shard as uint16

SPLITS = [
    ("train",   os.path.join(DATA_DIR, "train.jsonl")),
    ("dev_val", os.path.join(DATA_DIR, "dev_val.jsonl")),
    ("test",    os.path.join(DATA_DIR, "test.jsonl")),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("ERROR: pip install tokenizers==0.20.3")

    tok = Tokenizer.from_file(TOK_PATH)
    with open(TOK_INFO) as f:
        info = json.load(f)

    bos_id    = info["special_tokens"]["<bos>"]
    eos_id    = info["special_tokens"]["<eos>"]
    vocab_size = info["vocab_size"]
    assert vocab_size <= 65535, "vocab_size {} too large for uint16".format(vocab_size)
    print("Tokenizer: vocab={} bos={} eos={}".format(vocab_size, bos_id, eos_id))
    return tok, bos_id, eos_id, vocab_size


def load_jsonl_stories(path):
    stories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stories.append(json.loads(line)["text"])
    return stories


def encode_split(tokenizer, stories, bos_id, eos_id, split_name):
    """
    Encode all stories, returning flat token ID list and per-story offsets.
    Each story: [BOS] + story_tokens + [EOS]
    offsets[i] = [start, end) in the flat ID list.

    Verifies:
    - First token of each story is BOS
    - Last token of each story is EOS
    - No IDs exceed uint16 range
    """
    all_ids = []
    offsets = []
    malformed = 0
    bos_ok_count = 0
    eos_ok_count = 0

    for i, story in enumerate(stories):
        if i % 100_000 == 0 and i > 0:
            print("  {} encoding {:,}/{:,}...".format(split_name, i, len(stories)))
        try:
            enc = tokenizer.encode(story)
            ids = [bos_id] + enc.ids + [eos_id]
        except Exception:
            malformed += 1
            ids = [bos_id, eos_id]

        # Boundary verification
        if ids[0] == bos_id:
            bos_ok_count += 1
        if ids[-1] == eos_id:
            eos_ok_count += 1

        start = len(all_ids)
        all_ids.extend(ids)
        offsets.append([start, len(all_ids)])

    if malformed:
        print("  WARNING: {:,} stories failed encoding".format(malformed))

    max_id = max(all_ids) if all_ids else 0
    assert max_id <= 65535, "ID {} out of uint16 range".format(max_id)

    print("  Encoded {:,} stories: {:,} tokens".format(len(stories), len(all_ids)))
    print("  BOS boundary OK: {:,}/{:,}  EOS boundary OK: {:,}/{:,}".format(
        bos_ok_count, len(stories), eos_ok_count, len(stories)))

    return all_ids, offsets


def write_shards(all_ids, split_name, offsets):
    arr = np.array(all_ids, dtype=np.uint16)
    total = len(arr)
    if total == 0:
        return []

    use_multi = (split_name == "train") and (total > SHARD_TOKENS)
    shard_meta = []
    pos = 0
    idx = 0

    while pos < total:
        end = min(pos + SHARD_TOKENS, total) if use_multi else total
        chunk = arr[pos:end]
        fname = "{}_{:04d}.bin".format(split_name, idx) if use_multi \
                else "{}.bin".format(split_name)
        path = os.path.join(SHARD_DIR, fname)
        chunk.tofile(path)
        raw = chunk.tobytes()
        shard_meta.append({
            "file":    fname,
            "tokens":  len(chunk),
            "sha256":  sha256_bytes(raw),
            "offset_start": pos,
            "offset_end":   end,
        })
        print("  Wrote {} ({:,} tokens, {:.1f} MB)".format(
            fname, len(chunk), len(raw) / 1024 ** 2))
        pos = end
        idx += 1

    off_path = os.path.join(SHARD_DIR, "{}_offsets.json".format(split_name))
    with open(off_path, "w") as f:
        json.dump(offsets, f)
    print("  Offsets → {} ({:,} stories)".format(off_path, len(offsets)))

    return shard_meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Building token shards v2 (JSONL input)")
    print("=" * 60)

    for p in [TOK_PATH, TOK_INFO]:
        if not os.path.exists(p):
            sys.exit("ERROR: {} not found. Run train_tokenizer.py first.".format(p))

    for _, p in SPLITS:
        if not os.path.exists(p):
            sys.exit("ERROR: {} not found. Run prepare_tinystories.py first.".format(p))

    os.makedirs(SHARD_DIR, exist_ok=True)
    tokenizer, bos_id, eos_id, vocab_size = load_tokenizer()

    manifest = {
        "version":          "v2",
        "built_utc":        datetime.now(timezone.utc).isoformat(),
        "tokenizer_sha256": sha256_file(TOK_PATH),
        "dtype":            "uint16",
        "byte_order":       "little-endian",
        "vocab_size":       vocab_size,
        "special_ids":      {"pad": 0, "bos": bos_id, "eos": eos_id},
        "story_encoding":   "[BOS] story_tokens [EOS]",
        "attention_policy": (
            "Concatenated stories; causal mask does not block cross-story attention. "
            "EOS/BOS are learned markers. Document offsets in *_offsets.json."
        ),
        "shard_tokens_target": SHARD_TOKENS,
        "splits": {},
    }

    total_stories = 0
    total_tokens  = 0

    for split_name, jsonl_path in SPLITS:
        print("\n[{}]".format(split_name.upper()))
        stories   = load_jsonl_stories(jsonl_path)
        all_ids, offsets = encode_split(tokenizer, stories, bos_id, eos_id, split_name)
        shard_meta = write_shards(all_ids, split_name, offsets)

        manifest["splits"][split_name] = {
            "source_jsonl": jsonl_path,
            "story_count":  len(stories),
            "token_count":  len(all_ids),
            "shards":       shard_meta,
        }
        total_stories += len(stories)
        total_tokens  += len(all_ids)

    mf_path = os.path.join(SHARD_DIR, "manifest.json")
    with open(mf_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nManifest → {}".format(mf_path))

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n[VALIDATION]")
    first_shard = manifest["splits"]["train"]["shards"][0]["file"]
    arr = np.fromfile(os.path.join(SHARD_DIR, first_shard), dtype=np.uint16)
    print("  dtype: {}  min: {}  max: {}  (vocab={})".format(
        arr.dtype, arr.min(), arr.max(), vocab_size))
    assert arr.max() < vocab_size
    assert bos_id in arr
    assert eos_id in arr

    # Shifted targets check
    x = arr[:8].astype(int)
    y = arr[1:9].astype(int)
    assert x[1] == y[0], "Shifted targets broken"
    print("  First 8 IDs: {}".format(x.tolist()))
    print("  Shifted y[0]: {}  (should == x[1]={})".format(y[0], x[1]))
    print("  All validations passed.")

    print("\n" + "=" * 60)
    print("  Shard build complete.")
    print("  Total: {:,} stories  {:,} tokens  ({:.1f}M)".format(
        total_stories, total_tokens, total_tokens / 1e6))
    print("  Train: {:,} tokens  ({:.1f}M)".format(
        manifest["splits"]["train"]["token_count"],
        manifest["splits"]["train"]["token_count"] / 1e6))
    print("=" * 60)
    print("\nSync to HPC:")
    print("  rsync -avz data/shards/v2/ <user>@hpc:/path/to/NanoGPT/data/shards/v2/")
    print("  rsync -avz data/tokenizer/v2/ <user>@hpc:/path/to/NanoGPT/data/tokenizer/v2/")
    print("\nNext: python 5th_stage_gpt.py  (on GPU node)")


if __name__ == "__main__":
    main()
