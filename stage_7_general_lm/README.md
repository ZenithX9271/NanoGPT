# Stage 7 — general language modeling and grounded assistant adaptation

This implements the from-scratch language-model track of
[the agreed Stage 7/8 methodology](../STAGE_7_8_EFFECTIVE_EXECUTION_METHODOLOGY.md).
The capability is fixed: explain, summarize and answer using a supplied English
passage. Stage 6 remains frozen. This is a new model/experiment, not its resume.

**Status, 16 November 2026: Stage 7A base pretraining is complete.** The
122,702,592-parameter model reached 2,000,011,264 successful training positions
on the original 2B schedule. Matched 512-window development perplexity fell from
57.7174 at 100M positions to 24.1660 at 2B; final sampled heldout perplexity is
24.1668. Generated explanations still contain errors and repetition. The
source-grounded study-assistant evaluation and adaptation are unfinished.

Start with the [comprehensive audit](docs/STAGE_7A_COMPREHENSIVE_AUDIT.md) and
[final 2B review](reports/TWO_BILLION_FINAL_REVIEW.md). The
[milestone report](MILESTONE_1_REPORT.md),
[V100 benchmark audit](reports/hpc_benchmark_1136426/AUDIT.md), and
[production-corpus plan](PRODUCTION_CORPUS_PLAN.md) preserve earlier decisions;
their pre-launch status statements are historical. HPC scripts reproduce the
recorded workflow and are not instructions to automatically launch another run.

Git includes code, configs, tests, the four-file pinned tokenizer, lightweight
milestone evidence, plots, and the complete final metrics log. Earlier duplicate
metrics logs, checkpoint backups, raw data, packed binary shards, caches and
environments remain outside Git. Corpus manifests record the external artifact
identities. Training and checkpoint-specific audit scripts need those external
files; a clone alone contains neither trained weights nor the production corpus.

## Implementation

- `evaluation/`: 20 development drafts, four disjoint SFT drafts, anchored rubrics,
  and source-group protection. Human review and the remaining development cases
  are pending. No sealed test content is created here.
- `model.py`: fused QKV, pre-RMSNorm, RoPE, SwiGLU, tied embeddings, portable causal
  attention and tested KV caching with explicit context-overflow refusal.
- `train.py`: fixed-stream pretraining, token-weighted accumulation, optimizer/RNG/
  sampler/scaler checkpoints, strict resume guards and temporary benchmark mode.
- `validate_restart.py`: exact production-runtime comparison of an uninterrupted
  second FP16 update with the same update after a checkpoint/process boundary.
- `corpus.py`: bounded input consumption, normalized exact dedup, stable source
  splits, tokenizer comparison and streaming uint16/uint32 packing by source.
- `extract_parquet.py` and `pack_production.py`: the separate Python 3.6
  Parquet-extraction and Python 3.9 tokenizer-packing stages for the bounded
  500M-token production-corpus candidate. Neither utility downloads data.
- `audit_corpus.py`: independent packed-shard, token-bound, ledger, input-hash,
  and split-group-disjointness validation before GPU admission.
- `pretrained.py`: local-cache-only instruct baseline and bounded LoRA pilot,
  native chat templates, response-only labels, explicit truncation refusal.
- `runtime_probe.py`, `cost.py`, `hpc/`: observed capabilities, cost worksheet and
  unsubmitted PBS templates, including separate CPU-only extraction and packing
  jobs. Allocations, A100 access and modern HPC setup remain unknown.

The numerical candidate is 12 layers / width 768 / 12 heads / intermediate 2048 /
context 1024. With the existing 49,152-token tokenizer it has **122,702,592 unique
parameters**. This pinned tokenizer was used throughout the completed production
run. A 32,000 vocabulary would give 109,529,856 parameters for the same decoder,
but that is a different untrained configuration.

## Checks that can run locally

From the repository root, using the observed Python 3.9 / PyTorch 2.8 CPU stack:

```bash
STAGE7_TOKENIZER_DIR="$PWD/stage_7_general_lm/artifacts/native_tokenizer" \
STAGE7_REQUIRE_NATIVE_TOKENIZER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -B -m unittest discover -s stage_7_general_lm/tests -v
python3 -B -m stage_7_general_lm.runtime_probe --report NEW_RUNTIME_JSON
python3 -B -m stage_7_general_lm.evaluation \
  --cases stage_7_general_lm/evaluation/dev_draft.jsonl --report NEW_VALIDATION_JSON
bash -n stage_7_general_lm/hpc/benchmark.pbs.proposed
bash -n stage_7_general_lm/hpc/train.pbs.proposed
```

Tests use the real production model classes at small configurations, plus an
exact candidate parameter-count check. A separate full-parameter CPU numerical
check is recorded in the milestone report. Native-tokenizer integration tests
use an explicitly configured offline bundle, as described below; a skip is not
a native-template or model-load pass.

## Portable offline native-tokenizer regression

The tokenizer fixture is selected with `STAGE7_TOKENIZER_DIR`; there is no
machine-specific temporary-directory fallback. An absolute path is preferred;
relative paths resolve from the repository root, and `~` is expanded.
Use an existing verified bundle in persistent working storage. Do not run the
dataset fetcher merely to obtain this fixture, and do not download model weights.
No tokenizer-only download utility is added by this change.

The four required files are `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` and `config.json`. Their exact sizes and SHA256 values
are recorded in [native_tokenizer.lock.json](configs/native_tokenizer.lock.json),
derived from the retained verified Mac bundle. The lock identifies
`HuggingFaceTB/SmolLM2-1.7B-Instruct` revision
`31b70e2e869a7173562077fd711b654946d38674`; it covers the chat-template metadata
as well as the vocabulary. All four files are verified before loading through
`AutoTokenizer` with `local_files_only=True` and `trust_remote_code=False`.
Socket connection attempts during that load fail the test.

Ordinary execution without a configured fixture explicitly skips the two
native integration methods. The fixture-policy rejection tests still run:

```bash
env -u STAGE7_TOKENIZER_DIR -u STAGE7_REQUIRE_NATIVE_TOKENIZER \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -B -m unittest stage_7_general_lm.tests.test_stage7 stage_8_compression.tests.test_compression
```

To require offline native-template coverage, replace the example path with your
existing verified directory and run from the repository root:

```bash
env STAGE7_TOKENIZER_DIR=/approved/persistent/path/native_tokenizer \
  STAGE7_REQUIRE_NATIVE_TOKENIZER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -B -m unittest -v \
  stage_7_general_lm.tests.test_stage7.Stage7Tests.test_actual_native_chat_template_and_truncation \
  stage_7_general_lm.tests.test_stage7.NativeTokenizerFixtureTests \
  stage_7_general_lm.tests.test_stage7.NativeTokenizerBundleIntegrityTests
```

For the complete strict suite, repeat the process-local variables:

```bash
env STAGE7_TOKENIZER_DIR=/approved/persistent/path/native_tokenizer \
  STAGE7_REQUIRE_NATIVE_TOKENIZER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -B -m unittest stage_7_general_lm.tests.test_stage7 stage_8_compression.tests.test_compression
```

The variables above apply only to the `env`-launched process. No shell startup
file or global environment setting is changed. If `STAGE7_REQUIRE_NATIVE_TOKENIZER=1` is set
without a fixture, the integration tests fail. An explicitly supplied empty,
missing, incomplete or mismatched bundle fails even in ordinary mode. Missing
Transformers dependencies or unsupported native-chat APIs with a configured
valid bundle are errors, not compatibility passes; do not install or change the
HPC environment automatically to hide such failures.

The native check covers all 20 development and four SFT drafts without changing
their content or human-review status. It checks native prefix alignment,
response-only supervision, real EOS preservation when PAD=EOS, native trailing
newline preservation, positional padding masks, exact context fit and overflow
refusal. No model is loaded and no training is performed by these native checks.
Corruption tests modify disposable copies only, never the configured bundle.
The broader pre-existing regression suite separately performs bounded CPU
optimizer fixtures; it does not launch production training or update preserved
checkpoints. Historical 31-test and 24-example reports remain unchanged.

## Reproduce the bounded data/tokenizer fixture

Choose NEW directories in quota-checked working storage. Raw/cache files stay
outside the cloud-synced source tree. The first fetch has an 8 MiB cap and downloads
24 documents plus tokenizer metadata, never model weights. It refuses changed
source heads/responses instead of silently replacing this measured sample.

```bash
S7_WORK=/quota_checked_working_storage/NEW_stage7_sample
python3 -B -m stage_7_general_lm.fetch_sample --output "$S7_WORK"
python3 -B -m stage_7_general_lm.corpus sample \
  --input "$S7_WORK/raw.jsonl" --output "$S7_WORK/sample" \
  --max-documents 24 --max-text-bytes 1048576 --max-read-bytes 4194304 --max-seconds 60 \
  --protected-groups stage_7_general_lm/evaluation/protected_groups.json
python3 -B -m stage_7_general_lm.corpus compare \
  --documents "$S7_WORK/sample/documents.jsonl" \
  --tokenizers data/tokenizer/v2/tokenizer.json "$S7_WORK/general-tokenizer.json" \
  --report "$S7_WORK/tokenizer_comparison.json"
python3 -B -m stage_7_general_lm.corpus pack \
  --documents "$S7_WORK/sample/documents.jsonl" --tokenizer "$S7_WORK/general-tokenizer.json" \
  --bos-id 1 --eos-id 2 --output "$S7_WORK/packed"
```

The BOS/EOS values above come from this pinned native tokenizer/model metadata;
they are not assumed for other tokenizers. Storage dtype uses the actual maximum
ID, including special tokens. The 24 documents are a bounded prefix pool with
seeded ordering, not a representative web sample. Mixture weights, quality review,
near-duplicate/derivative lineage, source rights and a larger corpus remain pending.
Dataset-viewer rows do not independently attest a git revision; the captured raw
response hashes provide content identity, not a complete revision-binding proof.

## Measured V100 benchmark

Interactive PBS job `1136426.pbshpc` measured the full candidate on one V100
32 GB with four CPUs and 16 GB requested host RAM. FP16 microbatch 4 ×
accumulation 4 was selected at 18,832 successful positions/second, with 15.88 GB
peak reserved VRAM and a 1.47 GB full checkpoint. The transferred reports match
their HPC hashes. See the audit for comparisons, cost projections and blockers.

The command below reproduces the bounded benchmark only inside a separately
approved allocation and with new report/probe paths:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -B -m stage_7_general_lm.runtime_probe --device cuda --threads 4 --report NEW_GPU_RUNTIME_JSON
python -B -m stage_7_general_lm.train --mode benchmark \
  --config stage_7_general_lm/reports/hpc_benchmark_1136426/candidate_fp16_mb4_acc4.json \
  --manifest STAGE7_MANIFEST_ON_WORKING_STORAGE \
  --device cuda --threads 4 --warmup 5 --steps 20 \
  --checkpoint-probe-dir CONFIRMED_PARENT_ON_INTENDED_CHECKPOINT_FILESYSTEM --report NEW_GPU_BENCHMARK_JSON
```

The PBS wrapper uses `STAGE7_GPU_HOURS`, `STAGE7_DISK_BYTES`, `STAGE7_MANIFEST` and
`STAGE7_OUTPUT`, plus `STAGE7_PROBE_DIR` on the intended checkpoint filesystem.
The probe records its filesystem device; a fast local `/tmp` measurement must not
be extrapolated to home/Lustre saves. All fields need actual limits/paths. Review the requested
PBS properties against current availability. No A100 or FlashAttention dependency
is assumed. The tiny fixture data can measure compute/checkpoint functionality,
but cannot establish representative large-corpus IO or quality.

Cost a proposed budget from a real candidate GPU report with:

```bash
python3 -B -m stage_7_general_lm.cost --benchmark GPU_BENCHMARK_JSON \
  --tokens 2000000000 --eval-interval 100 --save-interval 100 --report NEW_2B_COST_JSON
```

The estimator rejects CPU throughput. It separates training from periodic
evaluation/save costs. Startup/restarts, margin, queue exposure and the 40%
adaptation/compression reserve must be included before a main-run decision.

## Pretrained reference and SFT pilot

Candidate: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision
`31b70e2e869a7173562077fd711b654946d38674`. The native tokenizer was tested;
full checkpoint loading and adapters were not. PEFT is absent in the observed
environment. Modern dependency/driver/host compatibility is a separate gate.

Once reviewed weights are already cached and explicit memory allowances exist:

```bash
python3 -B -m stage_7_general_lm.pretrained --mode baseline \
  --revision 31b70e2e869a7173562077fd711b654946d38674 \
  --cases stage_7_general_lm/evaluation/dev_draft.jsonl --max-examples 20 \
  --max-weight-bytes CONFIRMED_WEIGHT_BYTES --max-ram-bytes CONFIRMED_RAM_BYTES \
  --device cpu --precision fp32 --output NEW_BASELINE_DIRECTORY
```

For a separately validated adapter runtime, change mode to `sft`, cases to
`evaluation/sft_train_draft.jsonl`, set `--max-examples 4 --max-updates 2`, and use
`--allow-draft-training` only for a clearly labelled draft pilot. The same model,
revision and explicit memory arguments remain required. CUDA FP16 needs its own
numerical/scaler check. LoRA targets currently assume this Llama-style family.

The SFT entry point exports an adapter and diagnostic state; it is explicitly not
a validated adapter-resume implementation. Do not extend it to a long SFT run
before testing adapter/scaler/sampler restart. Native PAD may equal EOS: masking
uses positions, so actual EOS labels remain supervised. Native trailing separators
after EOS are preserved. Prompt/source labels and padding are excluded.

## Main pretraining remains gated

Freeze the final tokenizer, corpus/mixture, batch, precision, LR horizon, evaluation
cadence and resource limits in a NEW configuration. `train.pbs.proposed` requires
that file, a new owned run directory and an explicit cumulative review update.
Its approved walltime must fit the requested four-hour allocation. This draft
trainer still needs cluster validation and launch review before production use.

Preferred plan: benchmark, select an affordable main horizon, then make the first
50–100M successful positions a gate on that unchanged trajectory. Do not finish a
short LR schedule and silently extend it. Successful updates set scheduler
progress; processed positions also count overflow attempts. Checkpoints are only
at complete, unpadded optimizer boundaries and restore optimizer/RNG/sampler/scaler.
Post-training response masking is handled in its separate entry point.

Both pruning and quantization continue independently under
[Stage 8](../stage_8_compression/README.md), targeting November 29. The own-1B
allocation decision follows measured learning curves and resource costs.

### First bounded production run: 10 November 2026

Job `1141598.pbshpc` passed the exact FP16 restart and production-data benchmark
and entered training on a V100. The 122,702,592-parameter model uses the unchanged
2B-position schedule with a first review at 100,007,936 successful positions.
`hpc/validate_and_train_first100m.pbs` combines the gate and first segment in one
four-hour allocation with two CPUs. See the [startup record and monitoring
paths](reports/HPC_RUN_1141598.md). Do not submit a duplicate while this job runs.

The 100M segment subsequently finished successfully and its checkpoint was
SHA256-verified and backed up on the Mac. On 11 November, the user approved
expanded evaluation and continuation. Job `1141809.pbshpc` passed that review
and completed 200,015,872 cumulative positions. Fixed-probe development CE
improved from 4.151479 to 3.869384. The earlier expanded development CE of
4.055558 and decoded completion probes describe the 100M checkpoint. See the
[review and continuation record](reports/CONTINUATION_1141809.md).

The user then approved continuation to 500M cumulative tokens, capped at six
additional V100 hours. The 200M endpoint is SHA256-verified and backed up on
the Mac. Phase 1 job `1142015.pbshpc` runs the 200M review and continues to
350,011,392 positions. Phase 2 job `1142016.pbshpc` has an `afterok` dependency
and continues to 500,006,912 positions. Both preserve the same training state
and 2B schedule. Do not launch a duplicate. See the
[500M continuation record](reports/CONTINUATION_TO_500M.md).

The 500M endpoint subsequently completed successfully, with fixed-probe dev CE
3.631664 and perplexity 37.775606. It is checksum-verified and backed up on the
Mac. On 12 November, the user approved a larger review and continuation to 1B,
capped at ten additional V100 hours. The 500M expanded review passed (dev CE
3.554775, perplexity 34.979933); decoded samples remain repetitive and factually
unreliable. Job `1144922.pbshpc` resumed optimization beyond 500M. Dependent jobs
`1144923.pbshpc`, `1144924.pbshpc` and `1144925.pbshpc` target 750M, 875M and 1B
after the first job's 625M endpoint, with another review at 750M. New logs,
deployment files and evidence are on scratch because the HPC home quota grace
period expired. See the [1B continuation record](reports/CONTINUATION_TO_1B.md)
for live log paths and the pinned deployment. Do not submit duplicate training.

### Authorized 1B to 1.5B continuation (13 November 2026)

The 1B endpoint is backed up and its expanded review passed. The next four
dependent jobs stop at 1,500,004,352 successful positions and run the matched
review there, within ten additional V100 GPU-hours. There is no automatic 2B
continuation. See [the submission and validation record](reports/CONTINUATION_TO_1P5B.md).

### Completed 1.5B endpoint (verified 15 November 2026)

All four continuation jobs completed. The full checkpoint is backed up on the
Mac and checksum verified. Matched expanded dev perplexity improved from 29.90
at 1B to 26.02 at 1.5B; generation remains repetitive and factually unreliable.
See [the final review and backup record](reports/ONE_POINT_FIVE_BILLION_REVIEW.md).
No training beyond 1.5B has been submitted by this workflow.

### Final 1.5B to 2B continuation (15 November 2026)

The user approved the final 500M positions. Four dependent jobs stop at the
original horizon, update 122,071 / 2,000,011,264 positions, within ten additional
V100 GPU-hours including review. The final job exports the completed endpoint,
runs matched development/generation review and a separate final-only heldout
evaluation. See [the final submission and validation record](reports/CONTINUATION_TO_2B.md).

### Original 2B horizon completed (verified 16 November 2026)

All four final jobs completed successfully at 122,071 updates / 2,000,011,264
successful positions. The larger dev perplexity is 24.16603 and sampled heldout
perplexity is 24.16679. Text still has substantial repetition and factual errors.
See [the final numerical, generation and preservation record](reports/TWO_BILLION_FINAL_REVIEW.md).
No further training is scheduled by this workflow.
