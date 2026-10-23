# NanoGPT — From Zero to Hero

> Following **Andrej Karpathy's** [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Video 7 of the *Neural Networks: Zero to Hero* playlist.

---

## 🗺️ Roadmap

We build the model **incrementally**, exactly as taught in the video:

| Phase | File | Description |
|-------|------|-------------|
| 1 | `bigram.py` | Bigram Language Model — baseline, no context |
| 2 | `gpt.py` | Self-Attention → Multi-Head → FFN → Residuals → LayerNorm → Full GPT |

---

## 📁 Project Structure

```
NanoGPT/
├── input.txt        # Tiny Shakespeare dataset (~1.1MB)
├── bigram.py        # Phase 1: Bigram baseline
├── gpt.py           # Phase 2–6: Full NanoGPT transformer
├── requirements.txt
└── README.md
```

---

## 🧠 Key Concepts (per Phase)

### Phase 1 — Bigram Model
- **Tokenisation**: character-level (65 unique chars in Shakespeare)
- **Embedding table**: shape `(vocab_size, vocab_size)` — row `i` = logits for next token after `i`
- **No context**: each prediction depends only on the *current* token
- **Loss**: cross-entropy. At init, expected loss ≈ `ln(65)` ≈ **4.17**

### Phase 2 — Self-Attention (the core idea)
- Tokens **communicate** with each other via Query, Key, Value projections
- Attention scores: `wei = Q @ K.T / sqrt(head_size)`
- **Causal masking**: future tokens are masked out (decoder-only)
- `V` (Value) = the actual information aggregated, weighted by attention scores

### Phase 3 — Multi-Head Attention
- Run `n_heads` attention heads in **parallel**, each with smaller `head_size`
- Concatenate outputs → project back to `n_embd`
- Gives the model multiple "communication channels"

### Phase 4 — Feed-Forward Network
- Applied **per-token**, independently: `FFN(x) = Linear → ReLU → Linear`
- Expands dimension by 4× internally (as in the original Transformer paper)
- Lets tokens "think" after communicating via attention

### Phase 5 — Residual Connections + Layer Norm
- **Residual**: `x = x + sublayer(x)` — allows gradients to flow directly
- **LayerNorm**: normalises across the embedding dimension (not batch)
- Applied *before* the sublayer (Pre-LN, slight deviation from original paper — common in practice)

### Phase 6 — Full Transformer (NanoGPT)
- Stack `n_layer` blocks, each = `MultiHeadAttention + FFN` with residuals
- Add `Dropout` for regularisation
- Train on Shakespeare, generate infinite Shakespeare! 🎭

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch

# Phase 1: Run the Bigram baseline
python bigram.py

# Phase 2: Run the full NanoGPT
python gpt.py
```

---

## 📊 Expected Results

| Model | Train Loss | Val Loss |
|-------|-----------|---------|
| Bigram (3k steps) | ~2.50 | ~2.51 |
| NanoGPT (5k steps, CPU) | ~1.80 | ~1.98 |
| NanoGPT (5k steps, GPU) | ~1.48 | ~1.49 |

> Losses are approximate. Run on M1/M2 Mac? Use `device = 'mps'` in `gpt.py` for a nice speedup!

---

## 📝 Notes

- The mathematical trick for efficient attention is: instead of explicit averaging, use **matrix multiplication with a lower-triangular mask** (tril + softmax).
- "Attention is all you need" (Vaswani et al., 2017) is the foundational paper.
- Karpathy's original repo: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
