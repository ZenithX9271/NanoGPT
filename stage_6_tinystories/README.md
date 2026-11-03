# Stage 6A — TinyStories continued pretraining

Stage 6A continues the verified Stage 5 model for 100,007,936 additional
positions. The run is complete: 12,208 new updates, a final development CE of
2.048690, and a complete 159-sample generation evaluation.

Start with [the final results](docs/RESULTS.md). The Stage 6 source remains here
for reproducibility; the experiment should not be launched again by default.

## Repository map

```text
stage_6_tinystories/
├── configs/       # Continuation, control, and fixed evaluation configuration
├── deployment/    # Exact source/data transfer inventory and checksums
├── docs/          # Results, research report, validation, and recovery notes
├── evaluations/   # Complete matched outputs at +25M, +50M, +75M, and +100M
├── hpc/train.pbs  # Canonical PBS entry point retained for reproducibility
├── recovery/      # CUDA-generation compatibility source and explanation
├── tests/         # Stage 6 and transfer-verification tests
├── tools/         # Transfer inventory preparation and verification
├── train.py       # Train/resume/preflight/benchmark entry point
└── evaluate.py    # Fixed-window scoring and generation entry point
```

Large checkpoints, scheduler logs, and transfer backups are intentionally ignored
by Git. The local `runs/s6a/` directory contains the preserved experiment state;
the public repository is documented by the results and evaluation artifacts.

## Experiment definition

- 6,897,152 parameters: 6 layers, width 256, 8 heads, context 256, dropout 0.2.
- Existing 8,192-token byte-level BPE, tied embeddings, FP32 manual attention.
- Stage 5 weights and AdamW state initialize a new continuation trajectory.
- Learning rate warms from 3e-5 to 1e-4 over 200 updates, then cosine-decays to
  3e-5 across 12,208 Stage 6 updates.
- Data are the existing repository-level `data/shards/v2/` and
  `data/tokenizer/v2/` artifacts; no Stage 6 data copy is required.
- `configs/control_lr.json` is an optional constant-3e-5 control and was not run.

## Results and evidence

- [Final results and artifact identities](docs/RESULTS.md)
- [Milestone comparison through +75M](evaluations/COMPARISON_THROUGH_75M.md)
- [Evaluation artifact index](evaluations/README.md)
- [Detailed research report through +50M](docs/RESEARCH_REPORT_THROUGH_50M.md)
- [Local and HPC validation](docs/VERIFICATION.md)
- [Deployment/transfer procedure](docs/TRANSFER.md)

The lower final development loss establishes improved next-token prediction on
the fixed windows. It does not by itself establish better narrative coherence;
human-rating columns remain intentionally blank.

## Verification

From the repository root:

```bash
python3 -B stage_6_tinystories/tests/test_stage6.py
python3 -B stage_6_tinystories/tests/test_transfer.py
python3 -B stage_6_tinystories/tools/verify_transfer.py --quick
bash -n stage_6_tinystories/hpc/train.pbs
```

Run the full transfer verifier without `--quick` to hash the entire payload. See
[the transfer guide](docs/TRANSFER.md) before regenerating its ledger.
