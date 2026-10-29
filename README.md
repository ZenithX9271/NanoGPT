# NanoGPT — Shashank's Language Model Training Journey

A from-scratch implementation of GPT-style language models, following Andrej
Karpathy's course and progressing from a character bigram to TinyStories training.
The architecture remains explicit PyTorch with FP32 and manual causal attention.

Documentation corrected against source, data manifests, checkpoint metadata and
local verification on **29 October 2026**. The trained Stage 5 baseline is valid;
several preparation and training-management paths still need repair.

## Stages at a glance

| Stage | Current implementation | Parameters | Data | Recorded validation CE |
| --- | --- | ---: | --- | ---: |
| 1 | Character bigram, 65 x 65 embedding table | 4,225 | Tiny Shakespeare | ~2.49 |
| 2 | 4 layers, width 64, 4 heads, context 64 | 211,777 | Tiny Shakespeare | 1.7154, historical log |
| 3 | Same model capacity, cluster training machinery | 211,777 | Tiny Shakespeare | 1.7423 in log; ~1.7566 in saved checkpoint |
| 4 | 6 layers, width 256, 8 heads, context 256 | 4,833,345 | Tiny Shakespeare | 1.514656 |
| 5 | 6 layers, width 256, 8 heads, context 256 BPE tokens, tied embeddings | 6,897,152 | TinyStories | 2.136093 |

The present Stage 2 and Stage 3 model classes are identical. Stage 2's current
script also contains cluster machinery; its directory name does not identify a
preserved single-head model. Earlier implementations remain in Git history.
The Stage 3 log and checkpoint have not been reconciled to one run.

CE is cross-entropy, not an accuracy percentage. Character CE on Shakespeare and
BPE CE on TinyStories cannot be compared directly. The Stage 5 checkpoint is the
selected story-model baseline; it is not an overall winner established by comparing
CE across different datasets and tokenizers.

## Selected Stage 5 baseline

`best_model.pt` at the project root is byte-identical to
`stage_5_tinystories/checkpoints/best.pt`:

- **6,897,152 unique parameters**, including tied input/output embeddings.
- **12,000 completed optimizer updates; 98,304,000 processed token positions (98.304M).**
- Saved sampled development CE **2.1360926628**, perplexity approximately **8.47**.
- Independent CPU audit: CE **2.1482**, perplexity **8.57**, on 128,000 sampled
  development tokens. This uses a different evaluation sample from the saved score.
- Generates readable phrases and story continuations, with remaining repetition,
  character drift and prompt-memory errors. Consistent narrative coherence is not
  established.

SHA256 of root `best_model.pt`:

```text
2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce
```

## Project structure and artifact provenance

```text
NanoGPT/
├── PROJECT_HANDOFF_STAGES_1_TO_5.md  # Detailed verified context and remaining gaps
├── README.md
├── docs_stage1_to_3.md              # Early-stage architecture and historical results
├── requirements.txt                # Only numpy is an active requirement
├── input.txt                       # Tiny Shakespeare: 1,115,394 characters / 65 chars
├── best_model.pt                   # Selected Stage 5 baseline; locally stored, ignored by Git
├── stage_1_bigram/
│   ├── model.py
│   └── output/run_output.txt
├── stage_2_gpt_minimal/
│   ├── model.py                    # Current source: 4-layer, 4-head GPT
│   └── output/run_output.txt       # Historical Stage 2 output
├── stage_3_gpt_shakespeare/
│   ├── model.py                    # CUDA-required training entry point
│   ├── model_gpu_variant.py        # Alternative historical GPU script
│   ├── shakespeare_input.txt
│   ├── checkpoints/best.pt
│   └── output/run_output.txt       # Historical V100 benchmark/training log
├── stage_4_gpt_scaled/
│   ├── model.py
│   ├── train_gpu.pbs               # Stale environment/entry point; needs repair
│   ├── checkpoints/               # best.pt and latest.pt; both record step 4999
│   └── output/run_output.txt
├── stage_5_tinystories/
│   ├── model.py                    # 6.9M model; nominal ~100M-position schedule
│   ├── evaluate.py
│   ├── configs/                    # stage5.json and prompt_ids.json
│   ├── checkpoints/               # See exact checkpoint contents below
│   ├── checkpoints_pilot/         # Archived v1 pilot checkpoints
│   ├── pipeline/                  # Existing artifacts valid; three entry points broken
│   ├── hpc/train.pbs              # Legacy module; log/walltime setup needs review
│   ├── tests/test_stage5.py
│   └── output/run_output.txt       # v1 ~10M pilot output, NOT full v2 training log
└── data/
    ├── raw/                       # Original TinyStories text files
    ├── tinystories/v2/            # Prepared JSONL splits and provenance
    ├── tokenizer/v2/              # Tokenizer JSON and metadata
    └── shards/v2/                 # uint16 shards and manifest
```

Checkpoints, raw/prepared text, shards and job logs are ignored by Git. A fresh
clone does not include all artifacts required for evaluation or training; they
must be transferred separately. Tokenizer files and configurations are tracked.

## Stage 5 data and checkpoint details

The project uses the original TinyStories text source. The directory label `v2`
means this project's revised JSONL pipeline, not the separate TinyStories V2 corpus.

| Split | Stories | Stored BPE tokens, including BOS/EOS |
| --- | ---: | ---: |
| Train | 1,792,647 | 386,680,799 |
| Development | 10,995 | 2,316,856 |
| Test | 10,995 | 2,315,362 |

The tokenizer has 8,192 entries, with PAD=0, BOS=1 and EOS=2. It was trained on
training text only. Stories are concatenated with BOS/EOS; attention may cross
story boundaries inside a sampled window. The current loader reads all shards
into host RAM as int64; it is not memory-mapped.

| Stage 5 file | Completed updates | Processed positions | Saved development CE |
| --- | ---: | ---: | ---: |
| snapshot_10M.pt | 1,221 | 10,002,432 | 3.164195 |
| snapshot_25M.pt | 3,052 | 25,001,984 | 2.596292 |
| snapshot_50M.pt | 6,104 | 50,003,968 | 2.302775 |
| best.pt | 12,000 | 98,304,000 | 2.136093 |
| latest.pt | 12,000 | 98,304,000 | 2.136093 |
| snapshot_100M.pt | 12,000 | 98,304,000 | 2.136093 |

The last three files contain identical model weights and optimizer update counts.
`latest.pt` and `snapshot_100M.pt` do not establish a final 100M-position state.
The configured 12,207 updates would process **99,999,744** positions, 256 short of
100M. Sampling is with replacement; processed positions do not imply unique data
coverage. Saved pilot checkpoints instead record 1,220 updates, 9,994,240 positions
and CE 3.353233.

## Verification commands

Run these from the repository root in an already configured local environment,
with the required data and model artifacts present:

```bash
# Production-model checks; audit result: 52 passed, 0 failed
python3 stage_5_tinystories/tests/test_stage5.py

# Development evaluation and prompt-conditioned generation
python3 stage_5_tinystories/evaluate.py --checkpoint best_model.pt --split dev_val --n_chunks 500

# Equivalent saved Stage 5 baseline
python3 stage_5_tinystories/evaluate.py --checkpoint stage_5_tinystories/checkpoints/best.pt
```

The evaluator automatically selects CUDA, MPS or CPU. The independent audit
explicitly used CPU and evaluation microbatch 4 instead of the default 32, with
outputs redirected to a temporary directory. The commands above use the script's
defaults and save generated stories under the Stage 5 evaluation-output directory.
The test split has not been used for model scoring; reserve it for final evaluation.

Stage 1 can also train locally with `python3 stage_1_bigram/model.py`. Stage 3's
main training script requires CUDA and raises an error on a Mac/CPU-only system.

## Known gaps before fresh preparation or longer training

- `pipeline/train_tokenizer.py`: `_root` is used before definition (`NameError`).
- `pipeline/build_token_shards.py`: misplaced path helper causes `IndentationError`.
- `pipeline/encode_prompts_mac.py`: misplaced path helper causes `SyntaxError`.
- The Stage 4 PBS script references removed root `gpt.py` and an outdated Conda
  environment. Stage 5's PBS output paths differ from its created log directory;
  PBS path expansion needs cluster verification.
- Walltime protection is disabled by default. A saved optimizer is not a working
  resume path: restart loading and RNG/sampler restoration are not implemented.
- Evaluation/generation share or reset training RNG. Final checkpoint accounting
  and loss metadata need correction before continuation experiments.
- Benchmarks synchronize each step; save probes omit optimizer state. They do not
  provide a boundaries-only throughput benchmark or full-checkpoint overhead.

The 52 passing tests do not cover these preparation startup and cluster execution
paths. Existing data and trained weights passed the independent audit; no retraining
is required solely to correct documentation or historical checkpoint labels.

## Environments

Local audit: Python 3.9.6, PyTorch 2.8.0, Tokenizers 0.22.2, CPU. Tokenizer artifact
metadata also records Tokenizers 0.22.2. Some older script instructions mention
0.20.3; that is not the recorded artifact-building version.

`requirements.txt` currently installs only NumPy; it does not install PyTorch or
Tokenizers or fully recreate the tested local environment. Its commented generic
CUDA-wheel examples are not the verified IITD setup.

The working historical IITD environment is:

```bash
module purge
module load apps/pytorch/1.10.0/gpu/intelpython3.7
```

This supplies Python 3.7.9 and IITD's custom PyTorch 1.10.0a0 build; recorded V100
runs report CUDA 11.0. K40 has also been used successfully. GPU allocation must be
confirmed for each experiment; A100 access is not confirmed. Preserve FP32/manual
attention on this stack. SDPA, FlashAttention, BF16 and `torch.compile` are not
validated features of this environment and must not be enabled automatically.

## Next work

Preserve the accepted Stage 5 model. First repair preparation entry points and
training continuation/accounting, then run a measured quality-improvement experiment
with fixed development prompts and a declared compute budget. Broader pretraining,
SFT, preference learning, reasoning/tools and compression remain later goals.
