# NanoGPT Project Master Handoff & Technical Context — Stages 1–5

Corrected on **29 October 2026**, using current source, checkpoint metadata, data
manifests, Git history and independent local checks. This document separates the
trained results from functionality still needing repair. Documentation corrections
do not change model weights, code or historical output files.

## 1. Project identity and working approach

- **Author:** Shashank, IIT Delhi.
- **Goal:** Understand and build decoder-only language models incrementally, inspired
  by Andrej Karpathy's *Let's build GPT* and nanoGPT, while improving generated text.
- **GitHub:** `git@github.com:ZenithX9271/NanoGPT.git`, branch `main`.
- **Canonical local repository:**
  `/Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently Doing/NanoGPT`.
- **Historical copies:** the nested `NanoGPT` and `NanoGPT_From VM` folders inside
  sibling `NanoGPT_Codex` are older snapshots, not the current reorganized repository.

The audited base commit was `0d5094e987c853da92cbf6146efc17c7fbff6a35`; an actual
GitHub query confirmed it matched remote `main`. Read access was checked; pushing
was not tested. The audit and these documentation corrections did not push changes.

Keep the implementation understandable: explicit manual causal attention,
feed-forward layers, pre-LayerNorm, residuals and learned positional embeddings.
Improve language quality through bounded experiments, without an indefinite
infrastructure detour. Do not change architecture or start Stage 6 solely because
this handoff has been corrected.

## 2. Hardware and software evidence

| Environment | Verified context | Limits of verification |
| --- | --- | --- |
| Local Mac | Audit used Python 3.9.6, PyTorch 2.8.0, Tokenizers 0.22.2, CPU | New checks were not HPC/CUDA or MPS benchmarks |
| IIT Delhi HPC | Historical K40 training and V100-PCIE-32GB runs; Python 3.7.9, PyTorch 1.10.0a0+git36449ea, CUDA 11.0 | No new allocation, PBS submission or throughput measurement in this audit |
| A100 | Access was being discussed | Access is not confirmed; progress must not depend on it |

Working historical cluster setup:

```bash
module purge
module load apps/pytorch/1.10.0/gpu/intelpython3.7
```

Use FP32 and manual causal attention with the legacy stack. SDPA and
`torch.compile` are not APIs provided by PyTorch 1.10. Modern attention kernels or
mixed precision require separate capability and correctness validation; do not
upgrade the working shared environment or enable them automatically. Available
hardware and queue rules must be confirmed for each run.

`requirements.txt` has only NumPy as an active dependency. It does not recreate the
tested local environment or install PyTorch/Tokenizers. Its commented generic CUDA
wheel suggestions are not the established IITD module setup. Tokenizer artifact
metadata records Tokenizers **0.22.2**, despite older preparation comments mentioning
0.20.3.

## 3. Verified stage progression

| Stage | Current model | Parameters | Recorded result |
| --- | --- | ---: | --- |
| 1 | Character bigram, 65 x 65 table | 4,225 | Final logged validation CE 2.4911; plateau ~2.49 |
| 2 | 4 layers, width 64, 4 heads, context 64 | 211,777 | Historical Mac log validation CE 1.7154 |
| 3 | Same model capacity; cluster-oriented trainer | 211,777 | V100 log validation CE 1.7423; local checkpoint ~1.7566 |
| 4 | 6 layers, width 256, 8 heads, context 256, dropout 0.2 | 4,833,345 | Checkpoint/log validation CE 1.514656 |
| 5 | 6 layers, width 256, 8 heads, context 256 BPE tokens, tied embeddings | 6,897,152 | Best saved development CE 2.136093 at 98.304M processed positions |

CE means cross-entropy, not accuracy. Do not compare raw character CE on Shakespeare
with BPE CE on TinyStories, or rank models across these datasets using CE alone.

### Stage 1: character bigram

Current path: `stage_1_bigram/model.py`; historical name: `bigram.py`.

Tiny Shakespeare contains 1,115,394 characters and 65 unique characters. The
trainable `nn.Embedding(65, 65)` gives next-character logits conditioned on the
current character only. It has one-character conditioning, with no earlier-history
context; it is not a fixed table or completely memoryless. Block size 8 groups
training positions but does not give this model eight-character conditioning.

Initial CE is approximately 4.73. The uniform baseline is `ln(65) ≈ 4.17`;
random nonuniform logits need not achieve that baseline. The saved output is largely
unintelligible character sequences. A complete 3,000-update CPU rerun reproduced
every saved evaluation line.

### Stage 2: small Transformer

Current path: `stage_2_gpt_minimal/model.py`; historical name: `gpt.py`.

The current architecture has token and learned position embeddings, four causal
attention heads of dimension 16, and four pre-LayerNorm residual blocks. Each FFN
is `Linear(64, 256) → ReLU → Linear(256, 64)`, followed by final LayerNorm and a
character output projection. Batch 32, context 64, dropout 0.0.

The present Stage 2 and Stage 3 model-class ASTs are identical. The Stage 2 source
also contains cluster benchmarking/training features; it is not a preserved
single-head implementation. Earlier versions remain in Git. The historical Mac
output records CE 1.7154, but that does not prove the later cluster script reproduces
the same historical trajectory.

### Stage 3: cluster training machinery

Current paths: `stage_3_gpt_shakespeare/model.py` and `model_gpu_variant.py`.

Model capacity remains 211,777 parameters. Training adds a 100-update LR warmup,
cosine decay from 3e-4 toward 3e-5, gradient clipping at norm 1.0, and AdamW groups
with decay 0.1 on 2D weights and 0.0 on 1D parameters. Clipping limits gradient
norms; it does not guarantee training stability. Main training requires CUDA.

Temporary-file writes followed by `os.replace` reduce exposure to partial checkpoint
files; they are not a guarantee against all storage failures. Walltime exit logic
exists but `WALLTIME_MINUTES=None` disables it by default. A configured exit and a
saved optimizer do not demonstrate restart correctness: no tested resume/RNG path
is implemented.

Historical V100 report:

| Metric | Reported value |
| --- | ---: |
| Mean throughput | 65,093 processed character positions/s |
| Window range divided by mean | 1.9%; not statistical variance or ±1.9% |
| CUDA-event latency | Median 31.10 ms; p95 33.82 ms |
| Peak PyTorch GPU allocated / reserved | 48.0 / 62.0 MiB |
| Logged tokens at last evaluation | 10,237,952 |
| Total reported runtime | 195.0 s |
| Last logged train / validation CE | 1.5695 / 1.7423 |

Benchmarking uses 50 warmup plus five 100-update windows; these are actual training
updates. The measured loop also synchronizes every step for latency, and scalar
reads add synchronization. Its throughput describes that instrumented workload,
not a separate free-running boundaries-only benchmark. The reported checkpoint
probe saves model state without optimizer state, so it is not full-checkpoint cost.
Single-step unsynchronized throughput prints are not validated budgets. Allocated
GPU memory is not total process/device memory or proof that scaling will be cheap.

The locally saved Stage 3 checkpoint records CE approximately 1.7566, not 1.7423.
The checkpoint and log are unreconciled run evidence; both cannot be asserted to
represent the same verified best state. The checkpoint loads with the expected shapes.

### Stage 4: scaled Shakespeare model

Current path: `stage_4_gpt_scaled/model.py`.

Width grows from 64 to 256, layers from four to six, heads from four to eight,
and context from 64 to 256 characters. Dropout becomes 0.2. Parameter count is
**4,833,345**, about 22.8 times Stage 3. Dropout is regularization, not proof that
overfitting is prevented.

Both saved Stage 4 checkpoint metadata and the training log support development
CE **1.5146560669**, not 1.49. The checkpoints record step 4999 and 40,951,808
processed positions; do not describe `latest.pt` as a verified post-loop final state.
Temperature/top-k sampling is implemented. Shakespeare-like format and vocabulary
improve, while semantic coherence still requires qualitative evaluation.

### Stage 5: TinyStories and byte-level BPE

Current model path: `stage_5_tinystories/model.py`; configuration:
`stage_5_tinystories/configs/stage5.json`.

The source is the original TinyStories text dataset associated with Ronen Eldan
and Yuanzhi Li's *TinyStories: How Small Can Language Models Be and Still Speak
Coherent English?* The project's `v2` names the revised local JSONL pipeline; it
does not establish use of the separate GPT-4-only TinyStories V2 corpus.

**Pilot and correction history:**

- The archived pilot checkpoints record 1,220 completed updates, 9,994,240
  positions and CE **3.353233**, not ~2.4. The run configured 1,221 updates.
- Without a tokenizer, the old generation path silently substituted BOS for
  different prompts. Current saved prompt IDs avoid that dependency and match the
  tokenizer exactly. The current evaluator rejects missing required prompt data.
- The old `\n---\n` storage delimiter also occurred inside 22 stories, fragmenting
  records. JSONL fixes that ambiguity. BOS/EOS existed previously: the flaw was not
  the complete absence of boundary tokens.
- A post-loop 100M snapshot check was added, but its configured threshold is still
  unreachable: 12,207 × 8,192 = **99,999,744**, short of 100M. Snapshot naming and
  final-state accounting are not fully fixed.

**Prepared data:** normalized case/whitespace fingerprints are used for duplicate
checks, with official validation data split into development and test sets.

| Split | Stories | Stored token positions |
| --- | ---: | ---: |
| Train | 1,792,647 | 386,680,799 |
| Development | 10,995 | 2,316,856 |
| Test | 10,995 | 2,315,362 |

The custom byte-level BPE has 8,192 entries, trained on training text only. IDs are
PAD=0, BOS=1, EOS=2. Each story receives BOS/EOS before encoding into uint16 shards.
Windows sample concatenated stories and may attend across document boundaries;
there is no document-isolated attention mask. The loader reads and concatenates
shards as int64 in host RAM; it is not a memory-mapped loader.

**Model:** six blocks, width 256, eight heads of dimension 32, learned absolute
positions, context 256 BPE tokens, manual causal attention, ReLU FFN, pre-LayerNorm,
dropout 0.2 and FP32. `lm_head.weight` shares the token-embedding weights, and the
output projection is bias-free. Unique parameters: **6,897,152**.

**Training:** batch 32 gives 8,192 processed positions/update. Nominal schedule
12,207 updates; v2 warmup 500; LR 3e-4 toward 3e-5; AdamW weight decay 0.1;
gradient clipping 1.0. A pure-Python decoder and pre-encoded prompt IDs support
cluster generation without Tokenizers. PAD/BOS are suppressed in generated positions;
EOS can terminate generation.

## 4. Selected checkpoint and exact saved contents

Root `best_model.pt` is byte-identical to `stage_5_tinystories/checkpoints/best.pt`.
SHA256:

```text
2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce
```

| Stage 5 checkpoint | Completed updates | Processed positions | Saved development CE |
| --- | ---: | ---: | ---: |
| snapshot_10M.pt | 1,221 | 10,002,432 | 3.164195 |
| snapshot_25M.pt | 3,052 | 25,001,984 | 2.596292 |
| snapshot_50M.pt | 6,104 | 50,003,968 | 2.302775 |
| best.pt | 12,000 | 98,304,000 | 2.136093 |
| latest.pt | 12,000 | 98,304,000 | 2.136093 |
| snapshot_100M.pt | 12,000 | 98,304,000 | 2.136093 |

The last three have identical model weights and optimizer update counters. Preserve
these artifacts, but do not assert that `latest.pt` or `snapshot_100M.pt` contains
12,207 completed updates or a final 100M-token state. Code repairs cannot recover
missing historical final weights. The accepted baseline is the honest 98.304M
state; no retraining is required solely to obtain a round number.

Its saved CE 2.1360926628 corresponds to perplexity ~8.47. Independent CPU
evaluation gave CE 2.1482 / perplexity 8.57 on a different development sample.
Outputs contain readable grammar and story fragments, but also repetition, entity
drift and failures to retain prompt facts. Consistent multi-paragraph coherence,
reasoning and instruction following have not been established.

Processed positions are sampled with replacement. The 386.7M stored training
positions do not imply that a 98.3M-position run covered a known unique fraction.

## 5. Repository layout and evidence locations

```text
NanoGPT/
├── PROJECT_HANDOFF_STAGES_1_TO_5.md
├── README.md
├── docs_stage1_to_3.md
├── requirements.txt                # Only numpy active
├── input.txt
├── best_model.pt                   # Selected Stage 5 baseline
├── stage_1_bigram/                 # model.py; output/run_output.txt
├── stage_2_gpt_minimal/             # model.py; historical output/run_output.txt
├── stage_3_gpt_shakespeare/         # model.py, model_gpu_variant.py, corpus, checkpoint, output
├── stage_4_gpt_scaled/              # model.py, stale train_gpu.pbs, checkpoints, output
├── stage_5_tinystories/
│   ├── model.py
│   ├── evaluate.py
│   ├── configs/                    # stage5.json; prompt_ids.json
│   ├── checkpoints/               # Table above; generation_all_stages.txt; evaluation outputs
│   ├── checkpoints_pilot/         # Archived v1 pilot
│   ├── pipeline/                  # prepare_data.py; train_tokenizer.py;
│   │                             # build_token_shards.py; encode_prompts_mac.py
│   ├── hpc/train.pbs
│   ├── tests/test_stage5.py
│   └── output/run_output.txt       # OLD pilot log, not full v2 run
└── data/
    ├── raw/
    ├── tinystories/v2/             # JSONL splits and provenance
    ├── tokenizer/v2/               # tokenizer.json and vocab_info.json
    └── shards/v2/                  # 41 shards and manifest.json
```

Model checkpoints, raw/prepared text, shards and PBS logs are ignored by Git and
must be transferred separately. Tokenizer/configuration files are tracked. The
best checkpoint metadata and `generation_all_stages.txt` provide later-run evidence;
`stage_5_tinystories/output/run_output.txt` is not that run's complete training log.

## 6. Checks actually executed

The audit used local CPU, not a new cluster run:

1. All **52 production-model Stage 5 tests passed**, with no inline dummy-model fallback.
2. Evaluator main completed against root `best_model.pt`: CPU, microbatch 4 rather
   than default 32, 500 development windows × 256 positions, all three prompts and
   all nine generation settings. Outputs went to a temporary directory.
3. Full Stage 1 3,000-update rerun reproduced all saved evaluation lines.
4. Stage 2–4 actual model definitions passed parameter-count, CE, causal masking,
   reference-attention and finite-gradient checks; available checkpoints loaded strictly.
5. Full JSONL counts, tokenizer and all 41 shard hashes/ranges/lengths passed;
   BOS/EOS counts matched story counts. Saved prompt IDs matched tokenizer encoding.
6. Full normalized cross-split fingerprint checks found no overlap. This is not a
   near-duplicate guarantee. The test split was inspected for integrity, not scored
   for model selection.
7. Ten Python files parsed with Python 3.7 grammar; two failed. Syntax parsing
   alone is not an execution test under IITD's legacy PyTorch build.

No new HPC training, PBS execution, CUDA throughput validation or final-test-set
model scoring was performed. Passing the model suite does not verify all entry points.

## 7. Current commands and unresolved implementation gaps

From the repository root, with dependencies and artifacts already available:

```bash
python3 stage_5_tinystories/tests/test_stage5.py
python3 stage_5_tinystories/evaluate.py --checkpoint best_model.pt --split dev_val --n_chunks 500
python3 stage_1_bigram/model.py
```

The evaluator selects CUDA/MPS/CPU automatically, uses default microbatch 32, and
saves generated stories in the Stage 5 evaluation-output directory. The CPU audit
overrides above are not CLI options. Stage 3's main training entry point requires
CUDA and is not a Mac/CPU quickstart.

**Preparation paths currently fail after reorganization:**

- `pipeline/train_tokenizer.py`: `_root` used before definition; even `--help` fails.
- `pipeline/build_token_shards.py`: path helper interrupts a try/except; `IndentationError`.
- `pipeline/encode_prompts_mac.py`: misplaced helper; `SyntaxError`, plus early `_root` use.

These are source regressions, not evidence that the audited pre-existing artifacts
are corrupt. Do not present the preparation sequence as runnable until repaired.

**Training and cluster gaps:**

- No explicit resume path or saved/restored RNG/sampler state. A stored optimizer
  is not evidence of uninterrupted-versus-resumed equivalence, especially with dropout.
- Evaluation and snapshot generation consume/reset the shared training RNG.
- Final checkpoint code mixes step indices with completed-update counts and
  passes best validation loss as latest-state loss. The 100M threshold misses the
  configured schedule as described above.
- Walltime safety is disabled by default and needs coverage of startup,
  evaluation, generation and checkpoint overhead. A checkpoint on exit has not
  been demonstrated as a correct resumable walltime workflow.
- Stage 4's `train_gpu.pbs` activates outdated Conda and calls removed root `gpt.py`.
- Stage 5's PBS uses the correct module and new model path, but its requested
  root `logs/` output differs from the created `hpc/logs/` directory. Confirm path
  handling and PBS variable expansion on the cluster before submission.
- Current per-step-synchronized throughput and model-only save probes cannot
  establish free-running throughput or full resumable-checkpoint overhead.

Documentation changes record these gaps; they do not implement the repairs.

## 8. Git history and continuation direction

| Commit | Project milestone |
| --- | --- |
| 4e730fc | Initial bigram |
| b0546f0, 36e12ea | Small Transformer and updates |
| 248a37d | Cluster benchmark/PBS work and V100 output |
| 6dbfef8, f4dc45a | Early stage-name renames |
| ceeb37a | Stage 4–5 scripts, pipeline, tests and results |
| d89b411 | Stage-directory reorganization; introduced preparation startup regressions |
| 0d5094e | Original comprehensive handoff, subsequently corrected |

Next bounded implementation work: repair preparation startup, checkpoint/resume
accounting, RNG isolation and cluster run paths. Then compare the current Stage 5
baseline against a measured continuation experiment using fixed development prompts,
multiple seeds, prompt-memory checks and an explicit LR/compute budget. Do not
enlarge the model just to mask provenance errors or treat old throughput as a
validated budget for a new configuration.

Later goals remain broader pretraining, narrow SFT, preference learning,
reasoning/tools/agents, pruning, quantization and distillation, gated by evaluation
and measured compute. RoPE, SwiGLU, RMSNorm or larger models are optional experiments,
not established improvements here. KV caching avoids recomputing old keys/values;
attention over a growing history still costs work, so per-token decoding is not
O(1) in history length. No fixed speedup, runtime or future validation score is promised.
