# Stage 7 V100 benchmark audit — job 1136426

**Audit date:** 6 November 2026  
**Disposition:** **BENCHMARK PASS; PRODUCTION TRAINING NOT YET ADMITTED**

## Scope and identity

The benchmark used one `Tesla V100-PCIE-32GB` on `vsky033` through interactive
PBS job `1136426.pbshpc` (4 CPUs, 16 GB requested host RAM). The observed stack
was Python 3.7.9, PyTorch `1.10.0a0+git36449ea`, NumPy 1.18.5, CUDA build 11.0,
driver 550.90.07 and compute capability 7.0.

The tested model has 122,702,592 unique parameters, 12 layers, width 768, 12
heads, SwiGLU width 2,048, context 1,024 and vocabulary 49,152. All top-level
Stage 7 Python hashes in the benchmark match the local source on the audit date.
The benchmark data identity
`6bc232a27f927555fe8374925a71411a9d4922563b95e676dd302b4486c0c177`
matches `reports/corpus_manifest.json`.

The six files copied from HPC match their recorded HPC SHA256 values. The
runtime and benchmark reports are immutable evidence for this audit; the three
cost JSON files were derived locally from the selected benchmark by `cost.py`.

## Measured results

| Configuration | Positions/s | Mean step | Step CV | Peak allocated | Peak reserved | Successful measured positions |
|---|---:|---:|---:|---:|---:|---:|
| FP32, microbatch 2 × accumulation 8 | 8,833.78 | 1.8547 s | 0.155% | 9.38 GB | 9.59 GB | 327,680/327,680 |
| FP16, microbatch 2 × accumulation 8 | 16,631.58 | 0.9851 s | 0.261% | 8.60 GB | 9.19 GB | 327,680/327,680 |
| **FP16, microbatch 4 × accumulation 4** | **18,832.35** | **0.8700 s** | **0.254%** | **14.89 GB** | **15.88 GB** | **327,680/327,680** |

The selected microbatch-4 FP16 recipe is 2.132× the measured FP32 throughput
and 1.132× the microbatch-2 FP16 throughput, while retaining about 16.9 GB of
physical VRAM headroom on the 32 GB V100. No measured optimizer attempt was
skipped. The full checkpoint was 1,472,537,399 bytes; save and restore took
2.935 and 2.011 seconds on the measured scratch filesystem.

Evaluation is intentionally FP32. Its values are only fixture diagnostics after
25 temporary updates and are not model-quality evidence. The Cosmopedia
development shard has only 741 tokens and cannot provide one 1,024-position
window.

## Measured cost projections

These projections use 18,832.35 successful positions/s, evaluation every 100
updates, checkpointing every 100 updates and the measured save/evaluation costs.
They exclude queue time, startup, restarts and contingency.

| Horizon | Updates | Training | Periodic overhead | Subtotal |
|---|---:|---:|---:|---:|
| 100M positions | 6,104 | 1.475 h | 0.085 h | **1.560 h** |
| 200M positions | 12,208 | 2.950 h | 0.168 h | **3.118 h** |
| 2B positions | 122,071 | 29.500 h | 1.664 h | **31.164 h** |

## Validation completed

- The V100 runtime probe completed a finite production-class CUDA
  forward/backward operation.
- All three full-candidate benchmarks saved and restored optimizer, RNG,
  sampler and scaler state on the intended scratch filesystem.
- The strict local Stage 7 suite passed 30/30 tests with the hash-pinned native
  tokenizer bundle; no native-tokenizer test was skipped.
- Both proposed PBS scripts pass `bash -n`; every Stage 7 JSON file parses.
- The local benchmark artifacts match the six HPC SHA256 values.

## Production admission blockers

1. The packed fixture has only 13,681 training tokens. One update consumes
   16,384 sampled positions, so it cannot support a meaningful quality run.
2. A bounded 500M-token, 80/20 FineWeb-Edu/Cosmopedia candidate and split-runtime
   ingestion implementation now exist locally. They passed fixture tests, but
   the pinned source files have not been downloaded, verified, packed, or
   audited on HPC; the production manifest therefore does not yet exist.
3. The 49,152-token tokenizer remains a provisional selection. Its native bundle
   is now retained and lock-verified under the ignored local
   `stage_7_general_lm/artifacts/native_tokenizer/` directory, but it must still
   be transferred and verified on HPC before corpus packing or decoding claims.
4. Exact FP16 continuation equivalence on the V100 has not been demonstrated.
   Save/restore succeeded, but an uninterrupted path has not yet been compared
   with a split-and-resumed path after the next optimizer update.
5. The main trainer still needs immutable review-checkpoint publication,
   cumulative GPU accounting and a final production launch review.
6. The main token/LR horizon, resource caps and review gate remain unset. The
   benchmark-only candidate correctly keeps `main_run_approved=false`.
7. The 20 development cases remain human-review drafts; the planned independent
   sealed test set has not been registered. This does not block corpus packing,
   but it blocks defensible model selection and final quality claims.

## Decision

Select FP16 with microbatch 4 and accumulation 4 for the next frozen candidate.
Do not start quality training on the fixture. First preserve the tokenizer,
implement and validate bounded production corpus ingestion, harden checkpoint
publication/restart evidence, then freeze a 2B-position LR horizon with an
initial 100M-position review gate. A shorter review gate must remain part of that
unchanged longer schedule rather than completing a separate short cosine run.
