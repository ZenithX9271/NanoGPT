# NanoGPT — Shashank's Language Model Training Journey

A from-scratch implementation of GPT-style language models, following Karpathy's
course and progressively scaling up. Each stage is self-contained and builds on
the previous one.

---

## Project Structure

```
NanoGPT/
│
├── best_model.pt                 ← Champion model: Stage 5 best checkpoint (10.6M params, val CE 2.1361)
│
├── stage_1_bigram/               ← Stage 1: Bigram character model
│   ├── model.py                  #   Baseline: predict next char from current char only
│   └── output/run_output.txt     #   Training log
│
├── stage_2_gpt_minimal/          ← Stage 2: First Transformer
│   ├── model.py                  #   Single self-attention head, small MLP
│   └── output/run_output.txt     #   Training log
│
├── stage_3_gpt_shakespeare/      ← Stage 3: Multi-head GPT on Shakespeare
│   ├── model.py                  #   4 layers, width 64, 4 heads, ctx 64, ~212K params
│   ├── model_gpu_variant.py      #   GPU-adapted version (K40 benchmark)
│   ├── shakespeare_input.txt     #   Training corpus (tiny Shakespeare)
│   ├── checkpoints/              #   Stage 3 model checkpoints
│   │   └── best.pt               #   Best Stage 3 checkpoint (~212K params)
│   └── output/run_output.txt     #   Training log + benchmark results
│
├── stage_4_gpt_scaled/           ← Stage 4: Scaled-up Shakespeare GPT
│   ├── model.py                  #   6 layers, width 256, 8 heads, ctx 256, ~4.8M params
│   ├── train_gpu.pbs             #   PBS job script (IITD HPC, K40)
│   ├── checkpoints/              #   Stage 4 model checkpoints
│   │   ├── best.pt               #   Best Stage 4 checkpoint (~4.8M params, val loss ~1.49)
│   │   └── latest.pt             #   Final step snapshot
│   └── output/run_output.txt     #   Training log + generation samples
│
├── stage_5_tinystories/          ← Stage 5: BPE + TinyStories (100M tokens)
│   ├── model.py                  #   Same 6L/256W architecture, 8192-vocab BPE tokenizer
│   ├── evaluate.py               #   Perplexity + generation evaluation script
│   ├── configs/
│   │   ├── stage5.json           #   All hyperparameters in one place
│   │   └── prompt_ids.json       #   Pre-encoded prompt token IDs (Mac → HPC)
│   ├── checkpoints/              #   Stage 5 v2 checkpoints & training snapshots
│   │   ├── best.pt               #   Best checkpoint (step 12,000, val CE 2.1361)
│   │   ├── latest.pt             #   Final step 12,206 snapshot
│   │   ├── snapshot_10M.pt       #   10M tokens snapshot
│   │   ├── snapshot_25M.pt       #   25M tokens snapshot
│   │   ├── snapshot_50M.pt       #   50M tokens snapshot
│   │   ├── snapshot_100M.pt      #   100M tokens snapshot
│   │   └── eval_outputs/         #   Evaluated stories output folder
│   ├── checkpoints_pilot/        #   Stage 5 v1 pilot run archives (preserved)
│   │   ├── best.pt
│   │   └── latest.pt
│   ├── pipeline/                 #   One-time data preparation (run on Mac)
│   │   ├── prepare_data.py       #   Download TinyStories → clean JSONL records
│   │   ├── train_tokenizer.py    #   Train 8192-token byte-level BPE tokenizer
│   │   ├── build_token_shards.py #   Encode stories → uint16 binary shards
│   │   └── encode_prompts_mac.py #   Pre-encode prompts (saves configs/prompt_ids.json)
│   ├── hpc/                      #   HPC job scripts and logs
│   │   ├── train.pbs             #   PBS job script (IITD HPC, V100)
│   │   └── logs/                 #   PBS stdout/stderr logs from HPC runs
│   ├── tests/
│   │   └── test_stage5.py        #   52-check test suite (runs on Mac/CPU)
│   └── output/
│       └── run_output.txt        #   Training log from the 100M-token HPC run
│
├── data/                         ← Shared datasets and tokenizer artifacts
│   ├── raw/                      #   Raw TinyStories .txt files (download separately)
│   ├── tinystories/v2/           #   Cleaned JSONL splits (train / dev_val / test)
│   ├── tokenizer/v2/             #   Trained BPE tokenizer (tokenizer.json)
│   └── shards/v2/                #   Binary token shards (train_*.bin, manifest.json)
│
├── input.txt                     ← Tiny Shakespeare dataset for local Stages 1–4 runs
├── docs_stage1_to_3.md           ← Notes and derivations for Stages 1–3
├── requirements.txt              ← Python dependencies
└── LICENSE
```

---

## Stages at a Glance

| Stage | Model | Params | Dataset | Key idea | Checkpoints |
|-------|-------|--------|---------|----------|-------------|
| 1 | Bigram | ~100 | Shakespeare | Predict next token from current char only | — |
| 2 | 1-head GPT | ~10K | Shakespeare | First self-attention, small MLP | — |
| 3 | 4L/64W GPT | 212K | Shakespeare | Multi-head, residuals, LayerNorm | `stage_3_gpt_shakespeare/checkpoints/` |
| 4 | 6L/256W GPT | 4.8M | Shakespeare | Larger capacity, dropout, longer ctx | `stage_4_gpt_scaled/checkpoints/` |
| 5 | 6L/256W GPT | 10.6M | TinyStories (100M tokens) | BPE tokenizer, structured data pipeline | `stage_5_tinystories/checkpoints/` |

**Champion model:** [`best_model.pt`](file:///Users/shash/Library/CloudStorage/OneDrive-IITDelhi/Currently%20Doing/NanoGPT/best_model.pt) at the project root is the overall best model across all stages (Stage 5 v2 at step 12,000, 100M tokens seen, val CE 2.1361, ~8.47 perplexity).

---

## Quickstart

### Run Stage 3 (Mac / CPU)
```bash
cd NanoGPT
python3 stage_3_gpt_shakespeare/model.py
```

### Run Stage 5 tests (Mac / CPU)
```bash
cd NanoGPT
python3 stage_5_tinystories/tests/test_stage5.py
```

### Evaluate the Champion Model
```bash
cd NanoGPT
# Evaluates the root champion model:
python3 stage_5_tinystories/evaluate.py --checkpoint best_model.pt

# Or evaluates Stage 5's best checkpoint:
python3 stage_5_tinystories/evaluate.py
```

### Prepare Stage 5 data (Mac, one-time)
```bash
cd NanoGPT
python3 stage_5_tinystories/pipeline/prepare_data.py
python3 stage_5_tinystories/pipeline/train_tokenizer.py
python3 stage_5_tinystories/pipeline/build_token_shards.py
python3 stage_5_tinystories/pipeline/encode_prompts_mac.py
```

### Train Stage 5 on HPC (V100)
```bash
# Copy NanoGPT/ to HPC, then:
qsub stage_5_tinystories/hpc/train.pbs
```

---

## Checkpoint Details

### Champion Checkpoint (Root)
- **`best_model.pt`**: Step 12,000 | Val CE: 2.1361 | ~98.3M tokens trained. The best language model trained from scratch in this repository.

### Stage 5 Checkpoints (`stage_5_tinystories/checkpoints/`)
| File | Step | Val CE | Tokens seen | Description |
|------|------|--------|-------------|-------------|
| `snapshot_10M.pt`  | ~1,220  | — | 10M  | Early language patterns |
| `snapshot_25M.pt`  | ~3,050  | — | 25M  | Basic grammar & vocabulary |
| `snapshot_50M.pt`  | ~6,100  | — | 50M  | Coherent sentence structures |
| `best.pt`          | 12,000  | 2.1361 | ~98.3M | Lowest validation cross-entropy |
| `snapshot_100M.pt` | 12,000  | 2.1361 | 100M | Final schedule completion snapshot |
| `latest.pt`        | 12,206  | —      | 100M | Final training step snapshot |

---

## Requirements

```
torch>=1.10
numpy
tokenizers    # for data prep and tests on Mac
```

Install: `pip install -r requirements.txt`
