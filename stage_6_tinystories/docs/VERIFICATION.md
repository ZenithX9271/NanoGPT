# Stage 6A verification — 29 October 2026

This preserves the original local check record. For the 30 October transfer repair
and current status, see the [HPC validation report](HPC_VALIDATION.md) and the update below.

## Executed locally

Environment: macOS arm64, Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, CPU.
No source checkpoint was trained in place; all training smoke outputs were in
temporary directories. No HPC job was submitted.

### Automated suite

`python3 -B stage_6_tinystories/tests/test_stage6.py`

**19 tests passed, 0 failed, 0 skipped** with the real Stage 5 source/checkpoint
available. The suite includes:

- Cross-shard windows, shifted targets, memmap use and corruption detection.
- Required tokenizer/source-checkpoint identity and optimizer hyperparameter checks.
- Weight tying, causality, finite gradients and nonfinite-checkpoint rejection.
- Bit-for-bit uninterrupted/restarted training, both dropout 0.0 and dropout 0.35,
  including extra evaluations between training steps.
- A separate three-process test with dropout 0.35: eight uninterrupted updates
  versus three updates, process exit, checkpoint load, then five updates.
- Independent generation/evaluation RNG and preservation of caller train/eval mode.
- Schedule endpoints, ceiling token budgeting and incompatible-resume rejection.
- Walltime and signal exits, full final accounting, and pending final
  evaluation/snapshot completion on restart without an extra optimizer update.
- Benchmark-only source preservation, full checkpoint probe and run-directory guards.
- Read-only preflight CLI.
- Production Stage 5 strict state loading, exact logits and 6,897,152 parameters.

### Real 6.9M model restart demonstration

Using the pinned real Stage 5 checkpoint, unchanged model configuration and
**dropout 0.2**, three separate CPU processes performed:

1. Four uninterrupted updates.
2. Two updates, followed by process exit.
3. Resume from the second process's `latest.pt` and perform two more updates.

For this bounded smoke check only, batch was 1, context remained 256, and evaluation
used two training/two development windows. The comparison found **bit-identical
model tensors, AdamW state, RNG state, evaluation results and update/token counts**.
Final counters: 12,004 total completed updates / 98,305,024 processed positions.
This verifies the restart path, not story-quality improvement.

### Real artifacts and CLI smoke checks

- Preflight passed with the real baseline, tokenizer, fixed prompts, and SHA256
  checks of all training/development shards. The baseline remained byte-identical
  to SHA256 `2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce`.
- Benchmark-only CLI completed with the real model on CPU, batch 1/context 256,
  one warmup update plus two two-update windows. Its full optimizer/RNG checkpoint
  probe was **95,632,923 bytes**; no `.pt` file remained in the benchmark directory.
  CPU timings are not GPU budget estimates.
- Standalone evaluator completed against the real resumed model: two development
  windows and one prompt × three generation seeds × up to eight generated tokens.
  Evaluation JSON, story JSONL and rating CSV were produced.
- The supplied `tests/check_restart.py` orchestration harness also passed on CPU
  with the real checkpoint, batch 1/context 256/dropout 0.2, including all three
  fresh worker processes. Its GPU/default-batch run remains pending.
- All Stage 6 Python files parse with Python 3.7 grammar; `bash -n` passes for PBS.

## Status recorded on 29 October before the later HPC history was supplied

- IITD Python 3.7.9 / custom PyTorch 1.10.0a0 runtime and CUDA compatibility.
- K40/V100 forward/backward and GPU uninterrupted-versus-restarted equivalence.
  Use the supplied PBS `STAGE6_MODE=restart_check` to perform the fresh-process proof.
- Actual throughput, memory, evaluation/full-save overhead and PBS walltime behaviour.
- Complete fixed-window/159-continuation baseline versus Stage 6 comparison.
- Evidence that additional training improves development loss and story quality.

Do not interpret syntax checks or CPU restart equality as validation on a different
GPU/software stack. The original commands made that validation the first HPC step.

## 30 October update: user-reported HPC validation and local packaging repair

The user supplied a reconstructed execution history for V100 node vsky008:

- Preflight 1125951 passed strict checkpoint/optimizer/data/tokenizer/prompt checks.
- Benchmark 1125955 passed and discarded its temporary updates; reported throughput
  was approximately 70,375 processed positions/s, not a new measurement by this audit.
- Restart check 1125957 passed four uninterrupted versus two + restart + two updates,
  including weights, optimizer, RNG, counters and evaluation. The source was unchanged.

These are user-reported historical results, not new GPU executions or established
quality-training progress. No same-run `runs/s6a/latest.pt` is assumed to exist.
The earlier baseline-missing/pilot-hash errors explain the required exact baseline
copy and checksummed transfer packaging. Normal execution now goes directly from
transfer verification to actual training in `~/zenithX9271/NanoGPT`.

Training root Python files, model/training/evaluation configurations, original
tokenizer, all genuine prepared data and Stage 1–5 artifacts remain unchanged by
the packaging repair. Current locally executed results and clean-copy portability
evidence are in [HPC validation report](HPC_VALIDATION.md). No HPC modification/job submission was
performed as part of the repair.
