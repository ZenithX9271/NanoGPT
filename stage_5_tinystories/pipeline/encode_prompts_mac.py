"""
encode_prompts_mac.py — Prepare prompt token IDs on Mac
--------------------------------------------------------
Run this on Mac (where tokenizers library is installed) BEFORE
submitting the HPC training job.

Encodes each prompt with the v2 tokenizer (prepends BOS) and saves
configs/prompt_ids.json. The training script loads this file and uses
the IDs directly — no tokenizers library needed on the cluster.

Usage:
  python encode_prompts_mac.py
  # Then sync configs/prompt_ids.json to HPC before running 5th_stage_gpt.py
"""

import json
import os
import sys

TOK_PATH = _root("data", "tokenizer", "v2", "tokenizer.json")
OUT_PATH = _root("stage_5_tinystories", "configs", "prompt_ids.json")

# ── Prompts ────────────────────────────────────────────────────────────────────
# Loaded from stage5.json; override here if you want different prompts.
CONFIG_PATH = _root("stage_5_tinystories", "configs", "stage5.json")
with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

BOS_ID = _cfg["data"]["bos_id"]   # 1
EOS_ID = _cfg["data"]["eos_id"]   # 2
PAD_ID = _cfg["data"]["pad_id"]   # 0

PROMPTS = _cfg.get("generation", {}).get("prompts", [
    "Once upon a time, a little fox found a blue box.",
    "Lily wanted to help her friend, but she did not know how.",
    "Tom put his toy under the bed. Later, he looked for it and",
])

# ── Load tokenizer ─────────────────────────────────────────────────────────────
if not os.path.exists(TOK_PATH):
    sys.exit(
        "ERROR: {} not found.\n"
        "Run train_tokenizer.py first, or sync data/tokenizer/v2/ from HPC.".format(TOK_PATH)
    )

try:
    from tokenizers import Tokenizer

# ── Project root ──────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_STAGE_DIR = os.path.dirname(_HERE)          # stage_5_tinystories/
_PROJ_ROOT = os.path.dirname(_STAGE_DIR)     # NanoGPT/

def _root(*parts):
    """Join path parts relative to the project root."""
    return os.path.join(_PROJ_ROOT, *parts)

except ImportError:
    sys.exit("ERROR: pip install tokenizers==0.20.3")

tok = Tokenizer.from_file(TOK_PATH)
actual_vs = tok.get_vocab_size()
print("Tokenizer loaded: vocab={} BOS={} EOS={}".format(actual_vs, BOS_ID, EOS_ID))

# ── Encode ─────────────────────────────────────────────────────────────────────
encoded = []
print("\nEncoding prompts:")
for prompt_text in PROMPTS:
    enc     = tok.encode(prompt_text)
    ids     = [BOS_ID] + enc.ids   # prepend BOS exactly as training does
    decoded = tok.decode(enc.ids, skip_special_tokens=False)

    # Verify round-trip of prompt text (not including BOS)
    if decoded != prompt_text:
        print("  WARNING: round-trip mismatch for {!r}".format(prompt_text[:40]))
        print("           decoded: {!r}".format(decoded[:40]))

    # Verify first ID is BOS
    assert ids[0] == BOS_ID, "ids[0]={} expected BOS={}".format(ids[0], BOS_ID)

    # Verify no ID is PAD or EOS in the prompt part (they should not appear)
    prompt_body = ids[1:]
    if PAD_ID in prompt_body:
        print("  WARNING: PAD_ID={} appears in prompt body IDs".format(PAD_ID))
    if EOS_ID in prompt_body:
        print("  WARNING: EOS_ID={} appears in prompt body IDs (may terminate early)".format(EOS_ID))

    # Check that different prompts produce different ID sequences
    encoded.append({
        "text":     prompt_text,
        "n_tokens": len(ids),
        "ids":      ids,
    })
    print("  {:d} tokens  {!r}".format(len(ids), prompt_text[:60]))
    print("           IDs: {}...".format(ids[:10]))

# Verify all prompts are distinct
id_sequences = [tuple(e["ids"]) for e in encoded]
if len(set(id_sequences)) != len(id_sequences):
    print("\nWARNING: some prompts produce identical ID sequences!")
else:
    print("\nAll {:d} prompts produce distinct ID sequences. OK".format(len(encoded)))

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    "version":    "v2",
    "tokenizer":  TOK_PATH,
    "vocab_size": actual_vs,
    "bos_id":     BOS_ID,
    "eos_id":     EOS_ID,
    "pad_id":     PAD_ID,
    "prompts":    encoded,
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print("\nSaved {} ({:d} prompts) → {}".format(
    "prompt_ids.json", len(encoded), OUT_PATH))
print("\nSync to HPC before training:")
print("  scp {} <user>@hpc:/path/to/NanoGPT/configs/".format(OUT_PATH))
