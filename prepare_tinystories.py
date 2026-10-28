"""
prepare_tinystories.py — Stage 5 v2: JSONL story format
---------------------------------------------------------
Parses official TinyStories txt files, deduplicates, splits validation
50/50 (seed 1337), and writes one story per JSON line.

FORMAT CHANGE from v1:
  v1: stories separated by \\n---\\n  (flat text, fragile delimiter)
  v2: one JSON object per line: {"id": N, "text": "..."}
  Each line is a complete, self-contained record. No ambiguity with
  story content, no multi-line splitting needed.

RUN ON MAC (modern Python >= 3.10). No heavy dependencies.

INPUT:
  data/raw/TinyStories-train.txt  (~1.92 GB)
  data/raw/TinyStories-valid.txt  (~19.4 MB)

OUTPUT (JSONL, one story per line):
  data/tinystories/v2/train.jsonl
  data/tinystories/v2/dev_val.jsonl
  data/tinystories/v2/test.jsonl
  data/tinystories/v2/provenance.json

DOWNLOAD (if not already present):
  pip install huggingface_hub
  python - <<'EOF'
  from huggingface_hub import hf_hub_download
  for f in ["TinyStories-train.txt", "TinyStories-valid.txt"]:
      hf_hub_download(repo_id="roneneldan/TinyStories",
                      filename=f, repo_type="dataset", local_dir="data/raw")
  EOF

Dataset license: CDLA-Sharing-1.0
"""

import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR   = os.path.join("data", "raw")
OUT_DIR   = os.path.join("data", "tinystories", "v2")
TRAIN_RAW = os.path.join(RAW_DIR, "TinyStories-train.txt")
VALID_RAW = os.path.join(RAW_DIR, "TinyStories-valid.txt")

STORY_SEPARATOR = "<|endoftext|>"

DEDUP_SEED = 1337
VAL_SPLIT  = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────

def fingerprint(text):
    """Normalized fingerprint for deduplication (case/space-folded MD5)."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_stories(path):
    """Read a TinyStories txt file, split on <|endoftext|>, return cleaned list."""
    print("  Reading {} ...".format(path))
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split(STORY_SEPARATOR)
    stories, empty = [], 0
    for p in parts:
        s = p.strip()
        if s:
            stories.append(s)
        else:
            empty += 1
    print("  {:,} stories ({:,} empty blocks skipped)".format(len(stories), empty))
    return stories


def write_jsonl(stories, path):
    """Write list of story strings as JSONL: one {"id": N, "text": "..."} per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(stories):
            f.write(json.dumps({"id": i, "text": s}, ensure_ascii=False) + "\n")
    print("  Wrote {:,} stories → {}".format(len(stories), path))


def read_jsonl(path):
    """Read JSONL, return list of story strings."""
    stories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stories.append(json.loads(line)["text"])
    return stories


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TinyStories preparation v2 (JSONL format)")
    print("=" * 60)

    for p in [TRAIN_RAW, VALID_RAW]:
        if not os.path.exists(p):
            sys.exit(
                "ERROR: {} not found.\nSee download instructions at top of script.".format(p)
            )

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("\n[1] Parsing")
    train_raw = parse_stories(TRAIN_RAW)
    valid_raw = parse_stories(VALID_RAW)

    # ── Deduplicate train ─────────────────────────────────────────────────────
    print("\n[2] Deduplicating train")
    seen = {}
    train_dedup, train_dups = [], 0
    for s in train_raw:
        fp = fingerprint(s)
        if fp not in seen:
            seen[fp] = True
            train_dedup.append(s)
        else:
            train_dups += 1
    print("  Removed {:,} intra-train duplicates ({:,} remain)".format(
        train_dups, len(train_dedup)))

    # ── Deduplicate valid ─────────────────────────────────────────────────────
    print("\n[3] Deduplicating valid")
    val_fps = {}
    val_dedup, val_dups = [], 0
    for s in valid_raw:
        fp = fingerprint(s)
        if fp not in val_fps:
            val_fps[fp] = True
            val_dedup.append(s)
        else:
            val_dups += 1
    print("  Removed {:,} intra-valid duplicates ({:,} remain)".format(
        val_dups, len(val_dedup)))

    # ── Cross-split dedup (train ∩ val) ───────────────────────────────────────
    print("\n[4] Cross-split dedup (remove train stories appearing in valid)")
    val_fp_set = set(fingerprint(s) for s in val_dedup)
    train_clean = [s for s in train_dedup if fingerprint(s) not in val_fp_set]
    cross_rm = len(train_dedup) - len(train_clean)
    print("  Removed {:,} cross-split matches. Train final: {:,}".format(
        cross_rm, len(train_clean)))

    # ── Split valid → dev_val / test ──────────────────────────────────────────
    print("\n[5] Splitting valid 50/50 (seed {})".format(DEDUP_SEED))
    rng = random.Random(DEDUP_SEED)
    shuffled = val_dedup[:]
    rng.shuffle(shuffled)
    n_dv     = int(len(shuffled) * VAL_SPLIT)
    dev_val  = shuffled[:n_dv]
    test     = shuffled[n_dv:]
    print("  dev_val: {:,}  |  test: {:,}".format(len(dev_val), len(test)))

    # Cross-split exact-match check
    dv_fps  = set(fingerprint(s) for s in dev_val)
    t_fps   = set(fingerprint(s) for s in test)
    overlap = len(dv_fps & t_fps)
    print("  dev_val ∩ test exact overlap: {}".format(overlap))

    # ── Verify story count consistency ────────────────────────────────────────
    print("\n[6] Count consistency check")
    total_out = len(train_clean) + len(dev_val) + len(test)
    print("  train={:,}  dev_val={:,}  test={:,}  total={:,}".format(
        len(train_clean), len(dev_val), len(test), total_out))

    # ── Write JSONL ───────────────────────────────────────────────────────────
    print("\n[7] Writing JSONL files")
    write_jsonl(train_clean, os.path.join(OUT_DIR, "train.jsonl"))
    write_jsonl(dev_val,     os.path.join(OUT_DIR, "dev_val.jsonl"))
    write_jsonl(test,        os.path.join(OUT_DIR, "test.jsonl"))

    # ── Verify round-trip ─────────────────────────────────────────────────────
    print("\n[8] JSONL round-trip verification (spot check)")
    for name, stories, path in [
        ("train",   train_clean, os.path.join(OUT_DIR, "train.jsonl")),
        ("dev_val", dev_val,     os.path.join(OUT_DIR, "dev_val.jsonl")),
        ("test",    test,        os.path.join(OUT_DIR, "test.jsonl")),
    ]:
        reloaded = read_jsonl(path)
        count_ok = (len(reloaded) == len(stories))
        first_ok = (len(stories) == 0 or reloaded[0] == stories[0])
        last_ok  = (len(stories) == 0 or reloaded[-1] == stories[-1])
        print("  {} count={} first={} last={}".format(
            name,
            "OK({:,})".format(len(reloaded)) if count_ok else "FAIL",
            "OK" if first_ok else "FAIL",
            "OK" if last_ok  else "FAIL",
        ))

    # ── Provenance ────────────────────────────────────────────────────────────
    prov = {
        "version":        "v2",
        "format":         "JSONL: one {id, text} per line",
        "prepared_utc":   datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset":        "roneneldan/TinyStories",
            "license":        "CDLA-Sharing-1.0",
            "story_separator": STORY_SEPARATOR,
            "train_raw_sha256": sha256_file(TRAIN_RAW),
            "valid_raw_sha256": sha256_file(VALID_RAW),
        },
        "settings": {
            "dedup_seed":     DEDUP_SEED,
            "val_split_frac": VAL_SPLIT,
        },
        "counts": {
            "train_raw":          len(train_raw),
            "train_intra_dups":   train_dups,
            "train_cross_dups":   cross_rm,
            "train_final":        len(train_clean),
            "valid_raw":          len(valid_raw),
            "valid_intra_dups":   val_dups,
            "valid_final":        len(val_dedup),
            "dev_val":            len(dev_val),
            "test":               len(test),
            "dev_val_test_exact_overlap": overlap,
        },
    }
    prov_path = os.path.join(OUT_DIR, "provenance.json")
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    print("\n  Provenance → {}".format(prov_path))

    print("\n" + "=" * 60)
    print("  Done. Train: {:,}  Dev_val: {:,}  Test: {:,}".format(
        len(train_clean), len(dev_val), len(test)))
    print("=" * 60)
    print("\nNext: python train_tokenizer.py --data-dir data/tinystories/v2")


if __name__ == "__main__":
    main()
