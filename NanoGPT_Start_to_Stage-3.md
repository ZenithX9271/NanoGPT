# NanoGPT: Start to Stage-3 (Project Master Summary)

## 📌 1. Repository & Infrastructure Overview
- **GitHub Repository:** `git@github.com:ZenithX9271/NanoGPT.git` (Branch: `main`)
- **Dataset:** Tiny Shakespeare (`input.txt`) — 1,115,394 characters
- **Vocabulary:** Character-level tokenization with **65 unique characters** (`\n`, space, symbols, digits, lowercase `a-z`, uppercase `A-Z`).
- **Hardware Targets:**
  1. Local Development: Apple Silicon Mac (`mps` GPU backend)
  2. Production Training: IIT Delhi HPC Cluster (`vsky009`, NVIDIA Tesla V100 32GB GPU via PBS scheduler)

---

## 🏛️ 2. Architectural Progression Overview

```
STAGE 1: Bigram Model           STAGE 2: Full Transformer (GPT)          STAGE 3: Cluster-Grade Engine
  [bigram.py]                     [gpt.py]                                 [3rd_stage_gpt.py]
  • Static 65x65 lookup table     • Positional + Token Embeddings          • Warmup + Cosine Decay LR
  • 0 context memory              • Causal Self-Attention (Q, K, V)        • Gradient Clipping + Weight Decay
  • 4,225 params                  • Multi-Head Attention (4 heads)         • PBS Walltime & Atomic Checkpoints
  • Val Loss: ~2.47 (Plateau)     • 4-Block Residuals + LayerNorm          • CUDA Event Benchmarking
                                  • 211,777 params                         • V100 GPU Run: 65,093 tok/s
                                  • Val Loss: 1.71 on Mac MPS              • Val Loss: 1.74 (3.2 min runtime)
```

---

## 🔬 3. Stage-by-Stage Deep Dive

### Stage 1: Bigram Baseline (`bigram.py`)
- **Concept:** Simplest possible language model. It predicts the next character purely based on the current character.
- **Model Definition:**
  ```python
  self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # (65, 65)
  ```
- **Context Window ($T$):** 8 tokens (though mathematically unused by the model).
- **Parameters:** 4,225
- **Training Progression:**
  - Starts at loss **4.73** (near theoretical random uniform baseline: $-\ln(1/65) \approx 4.17$).
  - Plateaued at **~2.47** because English letter transitions cannot be predicted with zero memory of previous letters.
- **Generated Output:** Incoherent character soup with basic punctuation (`MARI he avayokis erceller thour d...`).

---

### Stage 2: Introducing Self-Attention & Transformers (`gpt.py`)
- **Concept:** Replaces blind lookup with **learned, data-dependent inter-token communication** across a 64-character context window.
- **Key Modules Implemented:**
  1. **Token + Positional Embeddings:**
     ```python
     tok_emb = self.token_embedding_table(idx)             # (B, T, n_embd)
     pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
     x = tok_emb + pos_emb                                 # (B, T, n_embd)
     ```
  2. **Causal Self-Attention (`Head`):**
     - Three linear weight matrices: $W_Q, W_K, W_V$ (each shape: `n_embd x head_size`).
     - Scaled dot-product: $\text{wei} = (Q \cdot K^T) / \sqrt{d_k}$.
     - Causal masking: Future positions filled with $-\infty$ using `torch.tril`, then normalized with `F.softmax`.
     - Value aggregation: $\text{out} = \text{wei} \cdot V$.
  3. **Multi-Head Attention (`MultiHeadAttention`):**
     - 4 parallel heads running on slices of 16 dimensions (`head_size = 64 // 4 = 16`).
     - Concatenated side-by-side: $4 \times 16 = 64$ dimensions.
     - Blended together via linear projection matrix $W_{\text{proj}}$.
  4. **Feed-Forward Network (`FeedForward`):**
     - Per-token MLP: $\text{Linear}(64 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 64)$.
  5. **Pre-LayerNorm & Residual Stream:**
     - $x = x + \text{Attention}(\text{LayerNorm}(x))$
     - $x = x + \text{MLP}(\text{LayerNorm}(x))$
  6. **Stacked Architecture:** 4 sequential blocks $\to$ final LayerNorm $\to$ `lm_head` linear layer $(64 \to 65)$.
- **Configuration (Small Baseline):**
  - `batch_size = 32`, `block_size = 64`, `n_embd = 64`, `n_head = 4`, `n_layer = 4`, `dropout = 0.0`.
  - **Parameters:** 211,777.
  - **Mac MPS Output:** Val loss dropped to **1.71**, producing real words and structured play dialogue.

---

### Stage 3: Cluster-Grade Production Engine (`3rd_stage_gpt.py` & `train_gpu.pbs`)
- **Concept:** Production hardening for high-performance computing (HPC) clusters (IIT Delhi PBS workload manager).
- **HPC Upgrades Implemented:**
  1. **Strict Hardware Assertion:** Fails loudly if CUDA is missing, preventing accidental CPU execution on allocated GPU nodes.
  2. **Learning Rate Schedule (`get_lr`):** 100-step linear warmup followed by smooth cosine decay down to `min_lr = 3e-5`.
  3. **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(1.0)` to eliminate exploding gradients.
  4. **Weight Decay Grouping:** 0.1 weight decay on 2D parameter tensors (weights, embeddings); 0.0 on 1D tensors (biases, LayerNorm).
  5. **PBS Walltime Margin:** Detects remaining walltime and automatically saves `latest.pt` 5 minutes before PBS issues `SIGKILL`.
  6. **Atomic Checkpointing:** Writes to `.pt.tmp` then renames to `.pt` (`latest.pt` and `best.pt`) to prevent file corruption.
  7. **Dual-Path Benchmarking:**
     - Windowed wall-clock timing with CUDA sync at boundaries (measures sustained real-world throughput).
     - Per-step CUDA event timing (measures precise latency percentiles: median, p95).

---

## 📊 4. IIT Delhi HPC Tesla V100 Benchmark Results

```text
==============================================================
  BENCHMARK & TRAINING REPORT (vsky009 - Tesla V100-PCIE-32GB)
==============================================================
  PyTorch:            1.10.0a0+git36449ea | CUDA 11.0
  Model Parameters:   211,777 (209,024 decay | 2,753 no-decay)
  Throughput (mean):  65,093 tokens/second (variance 1.9%)
  Step Latency:       Median: 31.10 ms | p95: 33.82 ms
  Peak VRAM Used:     Allocated: 48.0 MiB | Reserved: 62.0 MiB (<0.2% of 32GB!)
  Total Tokens:       10,237,952 tokens
  Total Runtime:      195.0s (3.2 minutes)
  Final Loss:         Train: 1.5695 | Validation: 1.7423
==============================================================
```

### Sample Output from Stage 3 Model:
```text
Clown:
Stridl'd and is not the acter bure up:
What them me?
I'at such higher's him vet?

MEXENIO:
I away, my fears, to zorn heaow,
From find you vail; I long egrivel:
In cintlatiHer drove to her in preace.

WARWICK:
Nand tear.
How my no my speak; and thew you lord.
I came at is I will
the the call leong-not what evils, when, bout wasches,
We stoong of his but than nonrupted so;
Ange the flown yould of while Prive my of.
```

---

## 🎯 5. Roadmap for the Next Session (Stage 4 & Beyond)

With 32 GB of VRAM on the Tesla V100 and only 48 MiB currently used, the next steps are:

1. **Scale to Full GPT Architecture (~10.7M Parameters):**
   - Activate the full config in `gpt.py`:
     ```python
     batch_size    = 64     # 2x batch
     block_size    = 256    # 4x context window
     max_iters     = 5000
     n_embd        = 384    # 6x embedding dimension
     n_head        = 6      # 6 attention heads
     n_layer       = 6      # 6 transformer blocks
     dropout       = 0.2    # regularization
     ```
   - Target Val Loss: $\approx \mathbf{1.48}$ (much higher coherence and grammar).
   - Expected runtime on V100: ~12 to 15 minutes.

2. **PyTorch Native FlashAttention:**
   - Replace manual `q @ k.T` and `tril` masking with:
     ```python
     out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
     ```
   - Avoids materializing the $O(T^2)$ attention matrix in VRAM, giving a 2–3× speedup.

3. **Mixed Precision Training (AMP):**
   - Wrap forward pass in `torch.cuda.amp.autocast(dtype=torch.float16)` using `GradScaler()` for Tensor Core acceleration.

4. **Pre-tokenized Binary Shards:**
   - Pre-tokenize `input.txt` to `train.bin` and `val.bin` using `numpy.memmap` (eliminates CPU data-loader overhead).

5. **Multi-GPU Distributed Data Parallel (DDP):**
   - Scale training across multiple cluster GPUs using `torchrun --nproc_per_node=N`.
