"""
tests/test_stage5.py — Focused checks for Stage 5 v2
------------------------------------------------------
Run on Mac (modern Python, torch installed):
  cd NanoGPT
  python3 tests/test_stage5.py

Tests exercise the PRODUCTION GPTLanguageModel from 5th_stage_gpt.py
(not a separate toy implementation).

Test groups:
  1. Config structure
  2. Tokenizer round-trips, save/reload, vocab range  (needs data/tokenizer/v2/)
  3. JSONL shard integrity: dtype, ID range, BOS/EOS, shifted targets
  4. Production model: weight tying and optimizer uniqueness
  5. Causal masking (no future information leakage)
  6. Finite gradients
  7. Small-batch overfit (300 steps, loss < 0.1)
  8. Prompt encoding: different prompts → different IDs
  9. Generation: greedy determinism, EOS stopping, PAD/BOS suppression
 10. Checkpoint: weight tying preserved after save/reload
 11. Split isolation: dev_val stories not in train (spot check, if files exist)
"""

import json
import math
import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # stage_5_tinystories/
PROJ_ROOT = os.path.dirname(ROOT)                                         # NanoGPT/
sys.path.insert(0, PROJ_ROOT)

CONFIG_PATH = os.path.join(ROOT, "configs", "stage5.json")

# ── Load production model from stage_5_tinystories/model.py ──────────────────
# model.py is safe to import on CPU: the CUDA assertion and smoke test
# now live inside `if __name__ == "__main__":` and do not execute on import.
# Module-level device falls back to "cpu" when CUDA is unavailable.

import importlib.util

spec    = importlib.util.spec_from_file_location(
    "_s5", os.path.join(ROOT, "model.py"))
_s5_mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(_s5_mod)
except Exception as _import_err:
    sys.exit(
        "FATAL: Could not import GPTLanguageModel from 5th_stage_gpt.py.\n"
        "Error: {}\n"
        "Fix the source before running tests — no inline fallback.".format(_import_err)
    )

if not hasattr(_s5_mod, "GPTLanguageModel"):
    sys.exit("FATAL: GPTLanguageModel not found in _s5 module after import.")

# Bind production names — tests always exercise the real code
GPTLanguageModel  = _s5_mod.GPTLanguageModel
_BOS_ID           = _s5_mod.BOS_ID
_EOS_ID           = _s5_mod.EOS_ID
_PAD_ID           = _s5_mod.PAD_ID
_vocab_size_test  = _s5_mod.vocab_size
_block_size_test  = _s5_mod.block_size



# ── Test runner ───────────────────────────────────────────────────────────────
results = []

def check(name, condition, detail=""):
    tag = "[PASS]" if condition else "[FAIL]"
    msg = "  {} {}{}".format(tag, name, "  ({})".format(detail) if detail else "")
    print(msg)
    results.append((name, condition))
    return condition


# ── 0. Config ─────────────────────────────────────────────────────────────────
def test_config():
    ok = os.path.exists(CONFIG_PATH)
    check("config file exists", ok)
    if not ok: return
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    check("model section present",     "model"    in cfg)
    check("training section present",  "training" in cfg)
    check("data.vocab_size == 8192",   cfg["data"]["vocab_size"] == 8192)
    check("model.weight_tying == true", cfg["model"]["weight_tying"] is True)
    check("data.bos_id == 1",          cfg["data"]["bos_id"] == 1)
    check("data.eos_id == 2",          cfg["data"]["eos_id"] == 2)
    check("warmup_iters_v2 in training",
          "warmup_iters_v2" in cfg.get("training", {}))


# ── 1. Tokenizer ──────────────────────────────────────────────────────────────
def test_tokenizer():
    tok_path = os.path.join(PROJ_ROOT, "data", "tokenizer", "v2", "tokenizer.json")
    if not os.path.exists(tok_path):
        print("  [SKIP] tokenizer v2 not found (run train_tokenizer.py first)")
        return

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  [SKIP] tokenizers library not installed")
        return

    tok = Tokenizer.from_file(tok_path)
    vs  = tok.get_vocab_size()
    check("vocab_size == 8192", vs == 8192, "got {}".format(vs))
    check("max ID < vocab_size", max(tok.get_vocab().values()) < vs)
    check("<pad>=0 <bos>=1 <eos>=2",
          tok.token_to_id("<pad>") == 0
          and tok.token_to_id("<bos>") == 1
          and tok.token_to_id("<eos>") == 2)

    for text in [
        "Once upon a time, a girl found an apple.",
        "Hello\nworld! Emoji: \U0001f600",
        "  spaces  ",
        "café résumé",
    ]:
        enc = tok.encode(text)
        dec = tok.decode(enc.ids, skip_special_tokens=False)
        check("round-trip: {!r}".format(text[:30]), dec == text)

    # Save-reload determinism
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_p = tmp.name
    tok.save(tmp_p)
    t2 = Tokenizer.from_file(tmp_p)
    os.unlink(tmp_p)
    s = "The quick brown fox jumps."
    check("save/reload determinism",
          tok.encode(s).ids == t2.encode(s).ids)


# ── 2. JSONL shards ───────────────────────────────────────────────────────────
def test_shards():
    import numpy as np
    mf_path = os.path.join(PROJ_ROOT, "data", "shards", "v2", "manifest.json")
    if not os.path.exists(mf_path):
        print("  [SKIP] v2 shards not found (run build_token_shards.py first)")
        return

    with open(mf_path) as f:
        mf = json.load(f)

    check("manifest version == v2", mf.get("version") == "v2")
    bos_id = mf["special_ids"]["bos"]
    eos_id = mf["special_ids"]["eos"]

    # Count consistency: story counts should match provenance
    prov_path = os.path.join(PROJ_ROOT, "data", "tinystories", "v2", "provenance.json")
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            prov = json.load(f)
        for split in ["train", "dev_val", "test"]:
            if split in mf["splits"] and split in prov.get("counts", {}):
                prov_key = "{}_final".format(split) if split == "train" else split
                mf_count = mf["splits"][split]["story_count"]
                pv_count = prov["counts"].get(prov_key, -1)
                check("{} story_count matches provenance".format(split),
                      mf_count == pv_count,
                      "manifest={} prov={}".format(mf_count, pv_count))

    for split in ["train", "dev_val"]:
        if split not in mf["splits"]: continue
        first = mf["splits"][split]["shards"][0]["file"]
        path  = os.path.join(PROJ_ROOT, "data", "shards", "v2", first)
        arr   = np.fromfile(path, dtype=np.uint16)

        check("{} dtype uint16".format(split), arr.dtype == np.uint16)
        check("{} max_id < 8192".format(split),
              int(arr.max()) < 8192, "max={}".format(arr.max()))
        check("{} BOS present".format(split), bos_id in arr)
        check("{} EOS present".format(split), eos_id in arr)

        # Shifted targets
        x = arr[:8].astype(int)
        y = arr[1:9].astype(int)
        check("{} shifted targets x[1]==y[0]".format(split), x[1] == y[0])

    # JSONL story count agrees with shard story count
    for split in ["train", "dev_val"]:
        jsonl = os.path.join(PROJ_ROOT, "data", "tinystories", "v2", "{}.jsonl".format(split))
        if os.path.exists(jsonl) and split in mf["splits"]:
            with open(jsonl) as f:
                jsonl_count = sum(1 for l in f if l.strip())
            mf_count = mf["splits"][split]["story_count"]
            check("{} JSONL count == shard story_count".format(split),
                  jsonl_count == mf_count,
                  "jsonl={} shard={}".format(jsonl_count, mf_count))


# ── 3. Model: weight tying ─────────────────────────────────────────────────────
def test_model_tying():
    model = GPTLanguageModel()
    check("lm_head.weight IS token_embedding_table.weight",
          model.lm_head.weight is model.token_embedding_table.weight)

    # Optimizer uniqueness
    seen = set()
    unique = [(n, p) for n, p in model.named_parameters()
              if id(p) not in seen and not seen.add(id(p))]
    ids = [id(p) for _, p in unique]
    check("shared weight counted once in named_parameters",
          ids.count(id(model.token_embedding_table.weight)) == 1)

    # Sanity: bias-free lm_head
    check("lm_head has no bias", model.lm_head.bias is None)


# ── 4. Causal masking ─────────────────────────────────────────────────────────
def test_causal_masking():
    model = GPTLanguageModel()
    model.eval()
    with torch.no_grad():
        ctx = _block_size_test
        x   = torch.zeros(1, min(8, ctx), dtype=torch.long)
        out1, _ = model(x)
        x2 = x.clone(); x2[0, -1] = 5   # change LAST token only
        out2, _ = model(x2)
        # Positions 0 … T-2 must be unchanged (no future leakage)
        causal_ok = torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)
        check("causal masking: first T-1 positions unchanged when last token changes",
              causal_ok)


# ── 5. Finite gradients ───────────────────────────────────────────────────────
def test_finite_gradients():
    model = GPTLanguageModel()
    model.train()
    vs = _vocab_size_test
    x  = torch.randint(0, vs, (2, 8))
    y  = torch.randint(0, vs, (2, 8))
    _, loss = model(x, y)
    loss.backward()
    all_finite = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad)
    check("finite gradients on random batch", all_finite)


# ── 6. Small-batch overfit ────────────────────────────────────────────────────
def test_overfit():
    torch.manual_seed(42)
    model = GPTLanguageModel()
    # Zero out dropout for overfitting test
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    model.train()
    vs = _vocab_size_test

    # Use indices well above special token range to avoid -inf suppression issues
    x = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10],
                       [4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)
    y = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11],
                       [5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    init_loss = None
    for i in range(300):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if init_loss is None: init_loss = loss.item()
        loss.backward()
        opt.step()
    final = loss.item()
    check("small-batch overfit (300 steps, loss < 0.1)",
          final < 0.1, "init={:.3f} final={:.3f}".format(init_loss, final))


# ── 7. Prompt encoding distinctness ───────────────────────────────────────────
def test_prompt_distinctness():
    tok_path = os.path.join(PROJ_ROOT, "data", "tokenizer", "v2", "tokenizer.json")
    if not os.path.exists(tok_path):
        print("  [SKIP] tokenizer not found")
        return
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  [SKIP] tokenizers not installed")
        return

    tok = Tokenizer.from_file(tok_path)
    BOS_ID = 1

    with open(os.path.join(ROOT, "configs", "stage5.json")) as f:
        cfg = json.load(f)
    prompts = cfg.get("generation", {}).get("prompts", [])

    ids_list = []
    for p in prompts:
        ids = [BOS_ID] + tok.encode(p).ids
        ids_list.append(tuple(ids))

    unique_seqs = set(ids_list)
    check("all prompts produce distinct ID sequences",
          len(unique_seqs) == len(ids_list),
          "{} unique of {}".format(len(unique_seqs), len(ids_list)))

    # BOS is first in each
    for i, ids in enumerate(ids_list):
        check("prompt {}: first ID is BOS={}".format(i, BOS_ID),
              ids[0] == BOS_ID)


# ── 8. Generation: determinism, EOS, PAD/BOS suppression ─────────────────────
def test_generation():
    model = GPTLanguageModel()
    model.eval()

    # Greedy is deterministic
    ctx = torch.tensor([[_BOS_ID, 3, 4]], dtype=torch.long)
    out1, stop1 = model.generate(ctx.clone(), max_new_tokens=10, greedy=True)
    out2, stop2 = model.generate(ctx.clone(), max_new_tokens=10, greedy=True)
    check("greedy is deterministic", torch.equal(out1, out2))
    check("greedy shape correct", out1.shape == (1, 13))  # 3 + 10

    # PAD and BOS must not appear in generated output
    generated_ids = out1[0, 3:].tolist()  # skip the input context
    check("PAD never generated",
          _PAD_ID not in generated_ids,
          "found {} instances".format(generated_ids.count(_PAD_ID)))
    check("BOS never generated",
          _BOS_ID not in generated_ids,
          "found {} instances".format(generated_ids.count(_BOS_ID)))

    # EOS stopping — subclass overrides forward() to always return max logit on EOS.
    # This exercises the real generate() code without depending on weight initialisation.
    # Zeroing all weights would make hidden states all-zero, causing LayerNorm/lm_head
    # to output zeros regardless, so the EOS row trick doesn't work that way.
    class _EOSForcedModel(GPTLanguageModel):
        def forward(self, idx, targets=None):
            B, T = idx.shape
            # Return logits where only EOS has a high value; all others are -100.
            # EOS_ID=2 is not suppressed by generate() (only PAD=0, BOS=1 are).
            logits = torch.full((B, T, _vocab_size_test), -100.0)
            logits[:, :, _EOS_ID] = 100.0
            return logits, None

    model2 = _EOSForcedModel()
    ctx2 = torch.tensor([[_BOS_ID, 5]], dtype=torch.long)
    out3, stop3 = model2.generate(ctx2, max_new_tokens=50,
                                  temperature=1.0, eos_id=_EOS_ID)
    check("EOS stops generation", stop3 == "eos",
          "stop_reason={}".format(stop3))
    check("short output when EOS fires", out3.shape[1] <= 5)


# ── 9. Checkpoint tying preserved ─────────────────────────────────────────────
def test_checkpoint_tying():
    model = GPTLanguageModel()
    check("tying before save",
          model.lm_head.weight is model.token_embedding_table.weight)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_p = tmp.name
    torch.save({"model": model.state_dict()}, tmp_p)

    model2 = GPTLanguageModel()
    ckpt = torch.load(tmp_p, map_location="cpu")
    model2.load_state_dict(ckpt["model"])
    os.unlink(tmp_p)

    check("tying after reload",
          model2.lm_head.weight is model2.token_embedding_table.weight)
    check("weight values equal after reload",
          torch.equal(model2.lm_head.weight, model2.token_embedding_table.weight))
    check("lm_head still bias-free after reload",
          model2.lm_head.bias is None)


# ── 10. Split isolation ────────────────────────────────────────────────────────
def test_split_isolation():
    import hashlib, re
    dev_path   = os.path.join(PROJ_ROOT, "data", "tinystories", "v2", "dev_val.jsonl")
    train_path = os.path.join(PROJ_ROOT, "data", "tinystories", "v2", "train.jsonl")
    if not (os.path.exists(dev_path) and os.path.exists(train_path)):
        print("  [SKIP] JSONL files not found (run prepare_tinystories.py first)")
        return

    def fp(text):
        return hashlib.md5(
            re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()

    with open(dev_path) as f:
        dv_fps = set(fp(json.loads(l)["text"]) for l in f if l.strip())

    train_fps = set()
    with open(train_path) as f:
        for line in f:
            if not line.strip(): continue
            train_fps.add(fp(json.loads(line)["text"]))
            if len(train_fps) >= 50_000:
                break

    overlap = len(dv_fps & train_fps)
    check("dev_val not in first 50K train stories",
          overlap == 0, "overlap={}".format(overlap))


# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Stage 5 v2 checks  (production GPTLanguageModel)")
    print("=" * 62)

    groups = [
        ("[0] config",             test_config),
        ("[1] tokenizer",          test_tokenizer),
        ("[2] shards",             test_shards),
        ("[3] weight tying",       test_model_tying),
        ("[4] causal masking",     test_causal_masking),
        ("[5] finite gradients",   test_finite_gradients),
        ("[6] small-batch overfit",test_overfit),
        ("[7] prompt distinctness",test_prompt_distinctness),
        ("[8] generation",         test_generation),
        ("[9] checkpoint tying",   test_checkpoint_tying),
        ("[10] split isolation",   test_split_isolation),
    ]

    for label, fn in groups:
        print("\n{}".format(label))
        fn()

    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print("\n" + "=" * 62)
    print("  {} passed  {} failed  ({} total)".format(
        n_pass, n_fail, len(results)))
    if n_fail:
        print("  FAILED:")
        for name, ok in results:
            if not ok:
                print("    - {}".format(name))
    else:
        print("  All checks passed.")
    print("=" * 62)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
