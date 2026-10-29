# NanoGPT: Stages 1–3 — Architecture and Historical Results

Corrected against current source, saved logs and checkpoint metadata on
**29 October 2026**. These notes retain the early-stage derivations; Stages 4–5
have already been implemented and are covered in the main handoff and README.

## 1. Repository, data and history

- GitHub: `git@github.com:ZenithX9271/NanoGPT.git`, branch `main`.
- Tiny Shakespeare: `input.txt`, 1,115,394 characters, 65 unique characters.
- Tokenization: character-to-integer encoding and inverse decoding.
- Historical local runs used Apple Silicon/MPS. Cluster runs used IIT Delhi GPUs,
  including K40 and V100; the reported Stage 3 benchmark is from a V100-PCIE-32GB.
- The present Stage 2 and Stage 3 model classes are identical; Stage 2's current
  script also includes cluster training machinery. Historical output is retained,
  but the directory name does not mean it contains a single-head minimal model.

| Stage | Current source | Historical filename | Parameters |
| --- | --- | --- | ---: |
| 1 | stage_1_bigram/model.py | bigram.py | 4,225 |
| 2 | stage_2_gpt_minimal/model.py | gpt.py | 211,777 |
| 3 | stage_3_gpt_shakespeare/model.py | 3rd_stage_gpt.py | 211,777 |

Earlier versions remain in Git history. The current Stage 3 training entry point
requires CUDA; it is not directly runnable as a Mac/CPU training command.

## 2. Stage 1: character bigram

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  # 65 x 65
```

This is a trainable table of next-character logits conditioned on the current
character. It has one-character conditioning and no earlier-history context.
Batching eight positions together does not make predictions depend on an
eight-character prefix. Parameters: `65 × 65 = 4,225`.

Initial CE is approximately 4.73, whereas a uniform predictor has CE
`-ln(1/65) ≈ 4.17`. Random nonuniform logits need not equal the uniform baseline.
The final logged validation CE is **2.4911**, with a plateau around **2.49**.
It learns local character transitions but cannot model longer dependencies;
generation remains mostly unintelligible.

A full 3,000-update CPU rerun reproduced every saved evaluation line during the
audit. This verifies the preserved bigram result, not Transformer training on HPC.

## 3. Stage 2: causal Transformer

Configuration: batch 32, context 64 characters, width 64, four heads, four blocks,
dropout 0.0. Unique parameters: **211,777**. The historical Mac output ends at
validation CE **1.7154**; the later current cluster script has not been shown to
reproduce the same historical trajectory.

### Token and position embeddings

```python
tok_emb = self.token_embedding_table(idx)  # (B, T, 64)
pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, 64)
x = tok_emb + pos_emb
```

The position vectors are learned absolute embeddings, not rotary embeddings.

### Causal attention

Each head projects the full 64-dimensional input into query, key and value vectors
of dimension 16. The heads do not simply partition the input into fixed slices.

```text
scores = (Q @ K.transpose(-2, -1)) / sqrt(16)
scores[future_positions] = -infinity
weights = softmax(scores, dim=-1)
out = weights @ V
```

A lower-triangular mask allows a position to attend to itself and earlier tokens.
Four independently learned head outputs concatenate into 64 dimensions, followed
by a learned output projection.

### Feed-forward and residual blocks

Each position uses an FFN:

```text
Linear(64, 256) → ReLU → Linear(256, 64)
```

Each of four blocks computes:

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

A final LayerNorm and `Linear(64, 65)` produce character logits. Historical samples
show improved words, punctuation and dialogue format; reliable semantic coherence
has not been established.

## 4. Stage 3: cluster-oriented training machinery

The model has the same capacity as Stage 2. Training support includes:

1. CUDA-required startup to detect missing GPU allocation.
2. A 100-update linear LR warmup toward 3e-4, then cosine decay toward 3e-5.
3. Gradient clipping at norm 1.0, limiting norms rather than guaranteeing stability.
4. AdamW decay 0.1 on 2D weights/embeddings and 0.0 on 1D biases/LayerNorm parameters.
5. Checkpoint writes through temporary files and `os.replace`, reducing partial-file
   exposure rather than guaranteeing protection from every storage failure.
6. Configurable walltime exits; **disabled by default** (`WALLTIME_MINUTES=None`).
7. CUDA-event latency and window wall-clock timing of actual training updates.

These features do not constitute a fully validated production/resume engine.
There is no implemented and tested restart path restoring optimizer and RNG/sampler
state. Stored optimizer state alone does not prove resumability. Evaluation uses
the shared RNG, and final-update/checkpoint accounting still needs repair.

## 5. Historical V100 benchmark: what it does and does not establish

Reported environment: Python 3.7.9, IITD PyTorch 1.10.0a0+git36449ea, CUDA 11.0,
Tesla V100-PCIE-32GB. Historical module setup:

```bash
module purge
module load apps/pytorch/1.10.0/gpu/intelpython3.7
```

| Reported metric | Value |
| --- | ---: |
| Model parameters | 211,777; 209,024 decay / 2,753 no-decay |
| Warmup and timed updates | 50 + 5 × 100 |
| Mean throughput | 65,093 processed positions/s |
| Min / max window throughput | 64,285 / 65,505 positions/s |
| Range divided by mean | 1.9%, not statistical variance or ±1.9% |
| Step median / p95 | 31.10 / 33.82 ms |
| Peak PyTorch GPU allocated / reserved | 48.0 / 62.0 MiB |
| Logged tokens at last evaluation | 10,237,952 |
| Total reported runtime | 195.0 s |
| Last logged train / validation CE | 1.5695 / 1.7423 |

The wall-clock windows include per-step CUDA synchronization used to read latency
events. Scalar reads add synchronization too. Consequently this is throughput of
an instrumented training loop, not an independent boundaries-only throughput test.
The benchmark's model-only save probe does not measure full optimizer-bearing
checkpoint overhead. Single-step unsynchronized prints are approximate indicators.
These historical figures have not been remeasured on HPC by the local audit.

PyTorch allocated memory excludes other GPU process/runtime overhead. Low allocated
memory is not proof that a proposed larger model or token budget is affordable.
New configurations need their own measured benchmark and evaluation/save overhead.

The saved Stage 3 checkpoint reports validation CE **approximately 1.7566**, while
the historical log reports **1.7423**. Both load/log evidence exist, but they have
not been reconciled to the same run. Do not describe the checkpoint as verified
to achieve the log's score.

Historical generated sample:

```text
Clown:
Stridl'd and is not the acter bure up:
What them me?
I'at such higher's him vet?

MEXENIO:
I away, my fears, to zorn heaow,
From find you vail; I long egrivel:
In cintlatiHer drove to her in preace.
```

This demonstrates learned theatrical structure while showing substantial spelling
and semantic errors.

## 6. What was actually verified locally

The audit used CPU with Python 3.9.6/PyTorch 2.8.0. It reproduced Stage 1's entire
run and checked Stage 2–4 actual model definitions for parameter counts, CE,
causality, reference-attention equivalence, finite gradients and available strict
checkpoint loading. Definitions were exercised without launching CUDA-only main
training loops. No new cluster throughput, PBS execution or exact training-resume
equivalence was demonstrated.

## 7. Subsequent implemented stages and future work

Stage 4 already uses **six layers, width 256, eight heads, context 256 characters**
and dropout 0.2: **4,833,345 parameters**, saved CE **1.514656**. The previously
proposed width-384 model, 12–15-minute runtime and target CE 1.48 were speculative,
not the implemented configuration or measured result.

Stage 5 already uses TinyStories, an 8,192-token BPE and tied embeddings:
**6,897,152 parameters**, best saved development CE **2.136093** after **98.304M**
processed BPE positions. The handoff documents checkpoint names, pipeline startup
failures and the continuation work still needed.

Keep FP32/manual attention on the existing legacy cluster stack. PyTorch 1.10 does
not provide `F.scaled_dot_product_attention` or `torch.compile`. Any future SDPA or
FlashAttention backend, mixed precision, memory mapping, DDP, architectural changes
or KV cache is an optional experiment requiring its own implementation and validation.
There is no established 2–3× speedup here. KV caching avoids recomputing prior
keys/values but attention cost still grows with the retained history.

Further training and model scaling require quality evaluation and a budget based
on measured processed positions/s for that configuration, including evaluation and
checkpoint overhead. Raw CE across character/BPE tokenizers is not directly comparable.
