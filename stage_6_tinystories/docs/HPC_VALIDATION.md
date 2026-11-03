# HPC transfer validation report — 30 October 2026

Local environment: python=3.9.6, torch=2.8.0, numpy=2.0.2, device=cpu.

| Check | Result | Evidence |
| --- | --- | --- |
| Genuine corpus located | PASS | Existing root data/tinystories/v2 JSONL + data/shards/v2; no regenerated or substitute corpus |
| Complete file SHA256 verification | PASS | 104 transfer entries present; all non-self files hashed |
| Shard integrity | PASS | 41 original shards; exact manifest hashes/byte lengths; all IDs < 8192 |
| Document offsets and BOS/EOS | PASS | All three splits have contiguous offsets and matching story/BOS/EOS counts |
| Split isolation | PASS | Full normalized training vs dev/test and dev vs test fingerprint checks: zero overlap |
| JSONL/tokenizer/shard reencoding | PASS | 8 genuine stories per split reencoded with existing tokenizer; all 24 exactly match shards |
| Raw-source recovery | PASS | Two external symlinks replaced by physical copies; raw SHA256 matches original provenance; originals untouched |
| Baseline copies | PASS | Both required files exactly match 2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce |
| Strict real model/optimizer loading | PASS (CPU) | Pinned 6,897,152-parameter checkpoint, 12,000 completed source updates; optimizer shape/moment/counter validation |
| Real forward and evaluation | PASS (CPU) | (1,256,8192) finite logits; 2 training + 2 development smoke windows; no optimizer update |
| Real generation | PASS (CPU) | Private seeded generator; up to 8 new tokens; PAD/BOS suppressed |
| Real checkpoint save/resume | PASS (CPU) | Temporary zero-update save/reload preserves all model tensors, optimizer state and sampler; stage6_updates stays 0 |
| Stage 6 suite | PASS | 19 tests, 0 failures, 0 skips; dropout restart equivalence on isolated unit fixtures |
| Packaging suite | PASS | 12 tests, 0 failures; missing/corrupt files, unsafe paths, symlinks, wrong baseline and PBS working directory |
| Clean transfer relocation | PASS (CPU) | Physical whitelist copy to a separate path with a space; full checksum + model/evaluation/checkpoint checks, original repository open() access denied |
| Python 3.7 / PBS syntax | PASS (syntax only) | All Stage 6 Python parses with 3.7 grammar; bash -n passes |
| Preservation | PASS | All root training Python/config/prompt bytes, Stage 1–5 source/docs/checkpoints and prepared artifacts preserved |
| Remaining required-file/symlink gaps | NONE | No required file missing; no repository symlink remains outside Git metadata |
| New GPU/PBS tests | NOT RUN | No HPC connection/modification, GPU job, benchmark or job submission in this repair |
| Quality-training progress | UNVERIFIED | No real-corpus optimizer update run locally; no HPC latest.pt assumed |

## Scope and interpretation

Only isolated unit fixtures perform optimizer updates. The genuine TinyStories
model is exercised with forward/evaluation/generation and a zero-update checkpoint
roundtrip; no quality-training run was started. CPU results do not claim new V100
measurements. Small smoke evaluation scores are not model-quality benchmarks.

The training model, optimizer setup, learning-rate schedule, batch/context,
dropout, seed, budget, evaluation/prompt settings and original corpus/tokenizer
were not changed. Training root-code/config identities are preserved. The only
Stage 6 edits concern packaging tools/tests, PBS target/thread/file checks and
documentation. The incorrect pilot baseline is preserved, excluded and not used.

The normal transfer contains prepared JSONL/test recovery assets even though the
optimizer loader needs only train/development shards. Original raw text is now
physical/self-contained locally but omitted from normal HPC transfer. No dataset
regeneration or tokenizer retraining was necessary.

## User-reported previous GPU evidence (not newly reproduced here)

| Job | Reported result | Interpretation |
| --- | --- | --- |
| 1125948 | Missing required Stage 6 baseline | Explains required physical baseline copy |
| 1125950 | Pilot checkpoint SHA256 rejected | Correct guard; pilot must not replace Stage 5 best |
| 1125951 | Preflight passed | Strict model/optimizer/data/tokenizer/prompt identity validated on V100 |
| 1125955 | Benchmark passed, ~70,375 positions/s, ~6.23s evaluation, ~0.418s save, ~2,983 MiB peak allocated | Historical measured workload; updates discarded, no quality progress |
| 1125957 | Four uninterrupted vs two + restart + two matched all states | Reported GPU restart validation only; not quality training |

Reported stack: Python 3.7.9, IITD PyTorch 1.10.0a0+git36449ea, NumPy 1.18.5,
CUDA 11.0, CentOS 7.9, Tesla V100-PCIE-32GB on vsky008, centos=skylake.
No standalone validation jobs are required again in the normal instructions for
this packaging-only repair. Actual allocation and quality-training outcomes still
depend on the subsequent user-run job. A latest file saved at stage6_updates=0
does not prove an optimizer update occurred.

## Reproduce the local non-training checks

```bash
python3 -B stage_6_tinystories/tools/verify_transfer.py --deep-data
python3 -B stage_6_tinystories/tests/test_transfer.py
python3 -B stage_6_tinystories/tests/test_stage6.py
python3 -B stage_6_tinystories/train.py --mode preflight --device cpu --verify-shards --init-from stage_6_tinystories/baseline/stage5_best.pt
bash -n stage_6_tinystories/hpc/train.pbs
```

The deep verifier's optional 24-story reencoding needs local Tokenizers; file
verification and Stage 6 execution do not. On the tested HPC module it can still
perform NumPy structural checks without installing that optional library.
