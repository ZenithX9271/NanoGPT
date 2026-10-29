# NanoGPT Project Master Handoff & Technical Context (Stages 1 – 5)

> **Document Purpose:** This document is the comprehensive master record and engineering handoff for Shashank's NanoGPT project. It details the complete architecture, dataset evolution, training protocols, HPC cluster benchmarks, debugging milestones, directory structure, and current status from initial concept through Stage 5. Use this document as the prompt/context for any subsequent chats, architectural discussions, or future scaling stages (e.g., Stage 6 scaling, fine-tuning, or KV caching).

---

## 1. Executive Summary & Working Philosophy

- **Author & Engineer:** Shashank (IIT Delhi)
- **Goal:** Progressively build, train, measure, and deeply understand decoder-only Transformer language models from absolute scratch (inspired by Andrej Karpathy's *nanoGPT* and *Let's build GPT*), scaling from a 4K-parameter character bigram to a 10.6M-parameter Byte-Level BPE model trained on 100M tokens of TinyStories.
- **Repository:** `git@github.com:ZenithX9271/NanoGPT.git` (Branch: `main`)
- **Key Principles:**
  1. *Pedagogical & Architectural Clarity:* Every building block (scaled dot-product attention, multi-head projection, feed-forward networks, pre-LayerNorm, residual stream, weight tying, learning rate schedules) is implemented explicitly in PyTorch without high-level wrapper libraries (like HuggingFace transformers or Lightning).
  2. *Hardware-Conscious Engineering:* Development, prompt encoding, and verification occur on local Apple Silicon Mac (`mps`/`cpu`); heavy compute runs on IIT Delhi HPC clusters (NVIDIA Tesla V100 32GB via PBS workload manager).
  3. *Empirical Verification:* No milestone is declared complete without measurable outputs: loss curves, token-throughput benchmarks, validation perplexity, generation samples, and rigorous unit testing (52/52 passing tests).

---

## 2. Hardware Environments & Toolchains

| Component | Local Mac Environment | IIT Delhi HPC Cluster (`vsky009`) |
| :--- | :--- | :--- |
| **System** | Apple Silicon MacBook Air (macOS) | CentOS Linux cluster node |
| **Processor / GPU** | Apple Silicon (MPS / CPU) | NVIDIA Tesla V100-PCIE-32GB |
| **Environment** | Python 3.9+, PyTorch, Tokenizers | Python 3.7 (`apps/pytorch/1.10.0/gpu/intelpython3.7`), CUDA 11.0 |
| **Batch Scheduler** | Interactive shell | PBS (Portable Batch System) job queue (`qsub`) |
| **Primary Role** | Data preparation, BPE training, unit tests, CPU evaluation | Full-scale pretraining (100M tokens), CUDA event benchmarking |

---

## 3. Project Roadmap & Architectural Progression

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  STAGE 1: Bigram Model │ ───> │ STAGE 2: 1-Head GPT    │ ───> │ STAGE 3: Cluster GPT   │
│  • Character-level     │      │ • First Self-Attention │      │ • 4L / 64W / 4H        │
│  • 4,225 parameters    │      │ • 211,777 parameters   │      │ • 211,777 parameters   │
│  • Zero context memory │      │ • Causal mask + MLP    │      │ • Cosine LR, PBS wall  │
│  • Val Loss: 2.47      │      │ • Val Loss: 1.71 (Mac) │      │ • 65K tok/s on V100    │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
                                                                             │
                                                                             ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ STAGE 5 v2: TinyStories│ <─── │ STAGE 5 Pilot: Flawed  │ <─── │ STAGE 4: Scaled GPT    │
│ • Byte-Level BPE (8192)│      │ • First TinyStories run│      │ • 6L / 256W / 8H       │
│ • 10.6M parameters     │      │ • Boundary leakage bug │      │ • 4.8M parameters      │
│ • JSONL + uint16 shards│      │ • Prompt fallback bug  │      │ • Dropout 0.2, ctx 256 │
│ • 100M tokens trained  │      │ • Missed 100M snapshot │      │ • Val Loss: 1.49       │
│ • Best Val CE: 2.1361  │      │ • Val CE: ~2.4         │      │ • Trained on K40/V100  │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 4. Deep Dive: Stages 1 Through 5

### Stage 1: Bigram Character Baseline
- **Location:** [`stage_1_bigram/model.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_1_bigram/model.py)
- **Data:** Tiny Shakespeare (`input.txt`, 1,115,394 characters, 65 unique characters).
- **Core Concept:** A static lookup table predicting token $t_{i+1}$ given solely token $t_i$, with zero memory of preceding history.
- **Model:** `nn.Embedding(vocab_size, vocab_size)` $\to$ `(65, 65)`.
- **Parameters:** 4,225.
- **Results:**
  - Initial loss: **4.73** (close to theoretical uniform random $-\ln(1/65) \approx 4.17$).
  - Plateaued at **~2.47** validation loss.
  - Output: Unstructured character noise (`MARI he avayokis erceller thour d...`).

---

### Stage 2: Introducing Self-Attention (Minimal GPT)
- **Location:** [`stage_2_gpt_minimal/model.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_2_gpt_minimal/model.py)
- **Core Concept:** Introduces learned token embeddings, learned positional embeddings, causal scaled dot-product self-attention, and multi-layer perceptrons (MLP).
- **Architecture:**
  - Embedding width: 64, Layers: 4, Attention heads: 4 (head dimension 16).
  - Context length ($T$): 64 characters.
  - Pre-LayerNorm residual blocks:
    $$x = x + \text{Attention}(\text{LayerNorm}(x))$$
    $$x = x + \text{FFN}(\text{LayerNorm}(x))$$
  - Causal masking using lower-triangular matrix (`tril`) with $-\infty$ mask before Softmax.
- **Parameters:** 211,777.
- **Results:** Val loss dropped to **1.71** on Mac MPS; model learned genuine English words, character names, and theatrical dialogue format.

---

### Stage 3: Cluster-Grade Production Engine
- **Location:** [`stage_3_gpt_shakespeare/model.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_3_gpt_shakespeare/model.py)
- **Core Concept:** Production-hardening the Stage 2 architecture for PBS batch execution on HPC clusters without altering model capacity.
- **Engineering Upgrades:**
  1. *Cosine Annealing LR:* 100-step linear warmup to $3 \times 10^{-4}$, decaying to $3 \times 10^{-5}$.
  2. *Gradient Clipping:* Norm capped at $1.0$.
  3. *Weight Decay Splitting:* 0.1 weight decay on 2D matrices (embeddings, linear weights); 0.0 on 1D biases and LayerNorms.
  4. *Atomic Checkpointing:* Writing to `.pt.tmp` before renaming to `.pt` preventing corrupted files during unexpected termination.
  5. *PBS Walltime Safety:* Calculates time left and gracefully exits with `latest.pt` 5 minutes prior to job kill.
  6. *CUDA Event Benchmarking:* Measured real forward/backward/optimizer step latencies.
- **V100 Benchmark Results:**
  - Sustained throughput: **65,093 tokens/sec** ($\pm 1.9\%$).
  - Step latency: Median 31.10 ms, p95 33.82 ms.
  - Final validation loss: **1.74** (5,000 steps finished in 3.2 minutes).

---

### Stage 4: Scaled Shakespeare GPT
- **Location:** [`stage_4_gpt_scaled/model.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_4_gpt_scaled/model.py)
- **Core Concept:** Scale model capacity by $\approx 23\times$ while introducing regularization (dropout) to prevent overfitting on the small Shakespeare corpus.
- **Architecture Changes:**
  - Embedding width: $64 \to 256$
  - Layers: $4 \to 6$
  - Heads: $4 \to 8$ (head dimension: $256 / 8 = 32$)
  - Context length ($T$): $64 \to 256$ characters
  - Dropout: $0.0 \to 0.2$
- **Parameters:** **4,801,601** parameters.
- **Results:**
  - Validation loss reached **~1.49** (significant improvement over Stage 3's 1.74).
  - Configurable generation: added temperature and top-$k$ sampling. Output produced coherent poetic stanzas with realistic Shakespearean vocabulary.

---

### Stage 5: TinyStories, Byte-Level BPE & 100M-Token Training
- **Location:** [`stage_5_tinystories/`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_5_tinystories)
- **Motivation:** Transition from Shakespeare character prediction to grammatical English story comprehension and generation using synthetic stories (Eldan & Li, *TinyStories*).

#### A. Pilot Run (v1) & Post-Mortem Flaw Analysis
In the initial pilot run (~1,221 steps, ~10M tokens), three severe engineering flaws were uncovered:
1. *Prompt Degeneration Bug:* In generation, all distinct prompts (`"Once upon a time, a little fox..."`, `"Lily wanted to help..."`, `"Tom put his toy..."`) generated the *exact same text* word-for-word. Root cause: The cluster lacked the `tokenizers` library, causing the script to silently discard prompt strings and fall back to unconditional `[BOS]` generation.
2. *Story Boundary Leakage:* Stories were stored separated by raw `\n---\n`. Attention windows crossed story boundaries without explicit token demarcations.
3. *Missed 100M Snapshot:* The loop snapshot condition checked at the top of the step while the token counter updated at the bottom. At step 12,206, the loop exited immediately when hitting 100M, never writing `snapshot_100M.pt`.

#### B. Complete Overhaul (Stage 5 v2)
1. **Data Pipeline (`pipeline/`):**
   - [`prepare_data.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_5_tinystories/pipeline/prepare_data.py): Converts raw data to structured JSONL records: `{"id": int, "text": str}`. Deduplicated across splits using MD5 content hashing:
     - Train: 1,792,647 stories
     - Dev_Val: 10,995 stories
     - Test: 10,995 stories (held out)
   - [`train_tokenizer.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_5_tinystories/pipeline/train_tokenizer.py): Trains a custom **8,192-vocab Byte-Level BPE** tokenizer with special tokens `<pad>=0`, `<bos>=1`, `<eos>=2`.
   - [`build_token_shards.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_5_tinystories/pipeline/build_token_shards.py): Encodes JSONL into compact `uint16` binary shards with strict `<bos>` prepend and `<eos>` append on every individual story. Total: 386,725,632 train tokens.
   - [`encode_prompts_mac.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_5_tinystories/pipeline/encode_prompts_mac.py): Pre-encodes prompt strings into explicit token ID lists stored in `configs/prompt_ids.json`. Eliminates runtime tokenizer dependency on the HPC cluster.
2. **Architecture:**
   - Vocabulary: 8,192 (Byte-Level BPE)
   - Width: 256, Layers: 6, Heads: 8, Block size: 256
   - **Weight Tying:** `lm_head.weight = token_embedding_table.weight` (reduces memory and ties input representation with output logit scoring).
   - Total Parameters: **10,612,736**.
3. **Training Protocol:**
   - 12,207 steps $\times$ 32 batch size $\times$ 256 block size $\approx$ **100,000,000 tokens**.
   - Warmup: 500 steps; Cosine decay: $3 \times 10^{-4} \to 3 \times 10^{-5}$; Weight decay: 0.1; Dropout: 0.2.
   - Pure-Python BPE decoder embedded directly in model script for dependency-free text generation on clusters.
   - Suppresses `<pad>` (0) and `<bos>` (1) during generation so the model never hallucinates control tokens.
4. **Final Results:**
   - **Validation Cross-Entropy: 2.1361** (Perplexity: **~8.47**) at step 12,000 (~98.3M tokens).
   - Generates coherent, grammatically correct, multi-paragraph children's stories with consistent characters and narrative resolution.
   - All 52 unit tests in [`stage_5_tinystories/tests/test_stage5.py`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/stage_5_tinystories/tests/test_stage5.py) pass on CPU/Mac without mocks or fallback models.

---

## 5. Current Clean Repository Directory Structure

```
NanoGPT/
├── best_model.pt                      ← Champion Model: Stage 5 best checkpoint (10.6M params, val CE 2.1361)
├── README.md                          ← Main user documentation & stage overview
├── LICENSE                            ← MIT License
├── requirements.txt                   ← Dependencies (torch, numpy, tokenizers)
├── docs_stage1_to_3.md                ← Mathematical notes & derivations for early stages
├── input.txt                          ← Tiny Shakespeare corpus (1.1M characters)
│
├── stage_1_bigram/
│   ├── model.py                       ← Static lookup bigram model
│   └── output/run_output.txt          ← Stage 1 training output
│
├── stage_2_gpt_minimal/
│   ├── model.py                       ← 1-head self-attention GPT
│   ├── checkpoints/                   ← Directory for Stage 2 saved weights
│   └── output/run_output.txt          ← Stage 2 training output
│
├── stage_3_gpt_shakespeare/
│   ├── model.py                       ← 4L/64W multi-head Shakespeare GPT + HPC hardening
│   ├── model_gpu_variant.py           ← Alternative GPU/MPS benchmark configuration
│   ├── shakespeare_input.txt          ← Self-contained Shakespeare dataset
│   ├── checkpoints/
│   │   └── best.pt                    ← Stage 3 best model (~212K params)
│   └── output/run_output.txt          ← Benchmark results (65K tok/s)
│
├── stage_4_gpt_scaled/
│   ├── model.py                       ← 6L/256W Shakespeare GPT (4.8M parameters)
│   ├── train_gpu.pbs                  ← PBS submission script for K40/V100
│   ├── checkpoints/
│   │   ├── best.pt                    ← Stage 4 best model (val loss ~1.49)
│   │   └── latest.pt                  ← Final step snapshot
│   └── output/run_output.txt          ← Scaled run training log
│
├── stage_5_tinystories/
│   ├── model.py                       ← Production 10.6M GPT + 100M token schedule
│   ├── evaluate.py                    ← Perplexity scoring & fixed-prompt generation
│   ├── configs/
│   │   ├── stage5.json                ← Hyperparameter & path configuration
│   │   └── prompt_ids.json            ← Pre-encoded prompt token IDs
│   ├── checkpoints/
│   │   ├── best.pt                    ← Step 12,000 champion weights (val CE 2.1361)
│   │   ├── latest.pt                  ← Step 12,206 final weights
│   │   ├── snapshot_10M.pt            ← 10M token training snapshot
│   │   ├── snapshot_25M.pt            ← 25M token training snapshot
│   │   ├── snapshot_50M.pt            ← 50M token training snapshot
│   │   ├── snapshot_100M.pt           ← 100M token training snapshot
│   │   └── eval_outputs/              ← Generated story evaluation logs
│   ├── checkpoints_pilot/
│   │   ├── best.pt                    ← Archived v1 pilot checkpoint
│   │   ├── latest.pt
│   │   └── generation_output.txt
│   ├── pipeline/
│   │   ├── prepare_data.py            ← Raw text → cleaned JSONL splits
│   │   ├── train_tokenizer.py         ← BPE tokenizer training script
│   │   ├── build_token_shards.py      ← JSONL → binary uint16 shards with BOS/EOS
│   │   └── encode_prompts_mac.py      ← Tokenizer runner for prompt_ids.json
│   ├── hpc/
│   │   ├── train.pbs                  ← PBS batch submission script for V100
│   │   └── logs/                      ← PBS output/error logs
│   ├── tests/
│   │   └── test_stage5.py             ← 52-test verification suite (Mac/CPU compatible)
│   └── output/
│       └── run_output.txt             ← Stage 5 v2 100M-token training log
│
└── data/
    ├── raw/                           ← Raw TinyStories files (git-ignored)
    ├── tinystories/v2/                ← Cleaned train.jsonl, dev_val.jsonl, test.jsonl
    ├── tokenizer/v2/                  ← tokenizer.json & vocab_info.json
    └── shards/v2/                     ← train_*.bin, dev_val_*.bin, manifest.json
```

---

## 6. How to Run & Verify

### A. Run Unit Tests (Mac / CPU)
Tests all components (tokenizer roundtrips, shard integrity, weight tying, causal masking, finite gradients, small-batch overfit, prompt distinctness, generation suppression, checkpoint reload, and split isolation):
```bash
cd NanoGPT
python3 stage_5_tinystories/tests/test_stage5.py
```
*(Expected: 52 passed, 0 failed)*

### B. Evaluate Checkpoints
```bash
# Evaluate champion model at root:
python3 stage_5_tinystories/evaluate.py --checkpoint best_model.pt

# Evaluate Stage 5 best checkpoint:
python3 stage_5_tinystories/evaluate.py --checkpoint stage_5_tinystories/checkpoints/best.pt
```

### C. Run Shakespeare Models Locally
```bash
# Stage 1:
python3 stage_1_bigram/model.py

# Stage 3:
python3 stage_3_gpt_shakespeare/model.py
```

### D. Submit Stage 5 Training to HPC Cluster
```bash
# On IIT Delhi cluster node:
qsub stage_5_tinystories/hpc/train.pbs
```

---

## 7. Next Potential Improvements (Ideas for Future Chats)

When starting future conversations, you can use this document as your baseline and explore:
1. **Architectural Enhancements (Modern LLM Standard):**
   - Rotary Position Embeddings (RoPE) replacing absolute learned embeddings.
   - SwiGLU activations replacing ReLU in the FFN.
   - RMSNorm replacing standard LayerNorm.
   - KV Caching (`past_key_values`) for fast $O(1)$ token-by-token autoregressive inference.
2. **Model Scaling (Stage 6):**
   - Scaling up to 30M – 80M parameters (e.g., width 512, layers 8–12).
   - FlashAttention integration (`torch.nn.functional.scaled_dot_product_attention`).
   - Mixed-precision training (FP16 / BF16 with gradient scaler).
3. **Data Scaling & Blending:**
   - Expanding from TinyStories to larger corpora (e.g., OpenWebText or FineWeb-Edu subsets).
   - Instruction Fine-Tuning (SFT) on prompt-completion story instructions.
