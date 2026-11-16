# Stage 7/8 — first bounded implementation milestone

Prepared 4 November 2026. Scope: Section 14 of the supplied execution methodology. This is a tested CPU foundation, not completion of Stage 7 pretraining or an approved HPC launch.

## Outcome and boundaries

Implemented the source-grounded study-assistant evaluation framework, bounded corpus/tokenizer tools, a 122.7M-parameter scratch candidate, training/resume/benchmark infrastructure, pretrained baseline/SFT entry points, and real pruning/quantization fixtures. The scratch and pretrained/compression tracks can proceed independently.

All **31 focused tests passed** in 7.695 seconds. Python 3.7 grammar checks passed for 19 Python files and two embedded Python blocks; `bash -n` passed for both proposed PBS scripts. These grammar checks do not establish that the historical HPC Python 3.7/Torch 1.10 environment supports every dependency.

All **72 existing Stage 6 files** retained their original SHA256 and size. No Stage 6 files were added or removed. No SSH connection, transfer, submission, GPU job, environment installation, or main training run occurred. Existing Stage 6 evaluation outputs were preserved. Bounded CPU loss/decode measurements and optimization/recovery fixtures were confined to derived models under the new implementation, as authorized by the methodology. See [preservation evidence](reports/stage6_preservation.json).

## 1. Evaluation before model selection

The capability is: **explain, summarize, and answer from supplied English text**, cite source sentence identifiers, and acknowledge missing information. It includes object/state tracking and conflicting or unsupported requests.

[SPEC.md](evaluation/SPEC.md) defines anchored 0–4 human rubrics for grounding, task accuracy, completeness, clarity, and state tracking, with explicit unsupported-claim review. Automated exact-answer/citation/termination diagnostics remain separate from these judgments; matching a citation identifier does not prove the cited sentence supports the answer.

[dev_draft.jsonl](evaluation/dev_draft.jsonl) contains 20 authored examples across 10 source groups: QA 6, summarization 4, explanation 3, missing-information 3, and state tracking 4. They are assistant-reviewed drafts; **all require human review**. Four separate source groups in [sft_train_draft.jsonl](evaluation/sft_train_draft.jsonl) provide only a pilot SFT fixture. No human ratings were invented, and neither draft is a scientifically sufficient benchmark.

The proposed complete allocation is 50 development cases and 150 owner-sealed test cases. No sealed test content has been created or inspected. The hash-only registration workflow records an independently prepared file without exposing its contents in reports. Source groups and derivative passages must remain in one partition. Exact normalized text/group checks are implemented; semantic derivative lineage still requires review. Grouped bootstrap support is provided, but 10 draft groups are too few for strong statistical conclusions.

Acceptance thresholds in [evaluation.json](configs/evaluation.json) are **proposals**, not validated gates: task/state/missing-information success ≥0.80, over-refusal ≤0.10, unsupported claims ≤0.05, and median faithfulness ≥3/4. Human review and calibration must precede freezing these thresholds and selecting a recipe. See [draft validation counts](reports/evaluation_validation.json).

## 2. Observed runtime and unresolved compatibility

Observed local environment: macOS 15.5 arm64, Python 3.9.6, Torch 2.8.0, NumPy 2.0.2, Transformers 4.57.6, tokenizers 0.22.2, and psutil 7.1.2. Execution used CPU, one thread. CUDA and MPS were unavailable to this process. PEFT, datasets, and bitsandbytes were not installed. See [runtime](reports/runtime.json) and [package observations](reports/observed_packages.json).

Historical HPC evidence identifies a V100 32GB and an older Python 3.7/custom Torch 1.10/CUDA 11 stack. Current driver, quota, walltime, storage, package availability, and permission for a modern environment remain unknown. No A100 allocation is assumed. Native CPU inference on the existing Stage 6 model and tokenizer formatting for the proposed pretrained baseline were observed; **pretrained 1.7B model loading, LoRA, GPU FP16, GPU exact restart, and accelerated INT4 remain unverified**.

The historical environment cannot be treated as compatible with the local Transformers/PEFT path merely because the new source parses as Python 3.7. A runtime probe and a separate compatibility decision are required before installing or launching anything. Deprecation warnings from the Torch AMP/quantization APIs and the local LibreSSL warning are retained in test output; no compatibility repair was attempted.

## 3. Bounded corpus and tokenizer evidence

The implemented fetch command was exercised on public dataset-viewer responses and tokenizer files only. The production fetch transferred 2,449,515 response-body bytes; an earlier bounded exploratory fetch transferred 2,455,922 bytes. Combined: **4,905,437 bytes**, with no model weights downloaded. Files are in separate temporary working directories; [download.json](reports/download.json) preserves the production fetch metadata.

The sample has **24 documents**, 12 FineWeb-Edu/sample-10BT and 12 Cosmopedia-v2; 94,211 retained UTF-8 text bytes. Seed 1337 assigned 20 train, 2 development, and 2 held-out documents using source groups. This is a shuffled bounded input pool, not a uniform or representative sample of either full corpus. See [sampling metadata](reports/sample.json).

| Domain | Original Stage 6 8K tokenizer | General 49K tokenizer |
|---|---:|---:|
| Cosmopedia-v2 | 13,752 | 8,701 |
| FineWeb-Edu | 15,635 | 10,320 |
| Total, without boundary tokens | 29,387 | 19,021 |

The general tokenizer uses **35.27% fewer tokens** on this small sample, with no round-trip mismatches. Documents exceeding 1,024 tokens decline from 10 to 3. This supports a provisional benchmark choice, not a final tokenizer or corpus-mixture decision. Vocabulary sizes differ substantially, which affects embedding cost and parameter comparisons. No new 32K tokenizer was trained. See [full comparison](reports/tokenizers.json).

The provisional tokenizer is SmolLM2-1.7B-Instruct revision `31b70e2e869a7173562077fd711b654946d38674`, tokenizer SHA256 `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`. Native vocabulary 49,152, BOS 1, EOS/PAD 2. Native template checks passed for all 24 development/SFT fixtures, maximum complete example 200 tokens. The template has content after EOS; tests verify real EOS supervision instead of incorrectly requiring EOS to be the final token. See [template checks](reports/native_template.json).

Dataset metadata heads and raw response SHA256 are recorded. The dataset-viewer response itself is not independently attested to a repository commit. Body hashes pin the received content; metadata heads alone do not prove revision-bound full-corpus reproducibility. Source-level licensing, quality review, semantic deduplication, and synthetic-data lineage remain required before scaling.

The streaming packer produced 19,069 tokens including document boundaries, using uint16 after checking actual token IDs. Train/dev/held-out totals are 13,681 / 4,476 / 912. Shards are partitioned by source and split, and manifest SHA256 values protect each shard. The tests also cover uint32 IDs above 65,535. Cross-document attention is currently allowed and documented; there is no hidden document-reset attention mask or dynamic mixture resampling.

**Small-sample limitation:** the Cosmopedia development slice contains only 741 tokens, and the entire held-out split contains only 912. Neither supports a full 1,024-position scoring window plus next-token target. The evaluator explicitly marks insufficient domain data instead of inventing scores. Aggregate development data supports a window, but this is still a pipeline fixture. See [manifest](reports/corpus_manifest.json).

## 4. Scratch candidate and trainer

[candidate.json](configs/candidate.json) defines 12 layers, width 768, 12 attention heads, SwiGLU width 2,048, 1,024 context, pre-RMSNorm, RoPE, tied embeddings/output weights, and dropout 0.1. At vocabulary 49,152 it has **122,702,592 unique parameters**. The 32K variant would have 109,529,856; it was counted, not trained or selected.

The **full 122.7M production class**, not a mathematical estimate or separate toy implementation, completed a CPU forward/backward pass at batch 1, sequence length 32 with finite gradients on every parameter and zero optimizer updates. This validates parameter shapes and bounded numerical execution. It does not demonstrate full-context V100 memory use or throughput. See [full-candidate CPU proof](reports/full_candidate_cpu.json).

The training implementation includes AdamW, parameter groups, clipping, token-weighted gradient accumulation, deterministic seeds, warmup plus cosine decay, CUDA FP16 scaling, and distinct attempted/processed/successfully updated counters. Skipped AMP updates do not count as completed optimizer updates. FP16 overflow behavior remains a GPU validation item.

Default benchmark proposal: microbatch 2 × accumulation 8 × context 1,024 = **16,384 positions/update**, peak/minimum LR 3e-4/3e-5, warmup 200 updates, weight decay 0.1, betas 0.9/0.95, and clipping 1.0. These are provisional, not a selected training recipe. The token budget is null and main-run approval false. Explicit positive resource allowances and a frozen approved configuration are required for main mode; tests establish that an unapproved main run is blocked before creating output artifacts.

Checkpoints contain model, optimizer, scaler, Python/NumPy/Torch/CUDA/sampler RNG, LR/update position, counters, and configuration/source/data/runtime identities. CPU resume tests verify identical next-step weights, AdamW state, RNG, counters, and evaluation across three fresh Python processes, including dropout. Evaluation uses separate fixed sampling and restores training RNG. Malformed state/configuration is rejected. Cached/uncached inference equivalence and context-limit refusal are also tested.

These are **CPU exact-restart results in the observed environment**. They do not establish bitwise restart across GPU models, drivers, CUDA libraries, or workspace settings. Before main training, the launch review must finish the GPU/driver/runtime fingerprint, test FP16 restart and scaler state on the selected stack, enforce cumulative allocation across jobs, and define immutable review/best-checkpoint preservation. The current main trainer is a guarded prototype, not a replacement for the reviewed Stage 6 production publication protocol.

A small model using the same production class (1,593,504 parameters, context 32) exercised the full benchmark/evaluation/checkpoint/restore CLI. Four measured updates produced 256 processed and 256 successful positions. Its 19,162,461-byte checkpoint was written and restored. **Its throughput must not be extrapolated to the candidate or V100.** The cost tool explicitly rejects CPU benchmark extrapolation. See [CPU pipeline evidence](reports/cpu_pipeline_benchmark.json).

## 5. Pretrained assistant/SFT track

[pretrained.py](pretrained.py) provides a native-chat baseline and a bounded LoRA pilot with pinned revisions, local-cache-only loading, explicit weight/RAM allowances, seeded execution, source/prompt/padding masking, and refusal to silently truncate overflowing examples. The true EOS remains supervised even when PAD=EOS. Native chat-template behavior was checked with the actual tokenizer; model weights were not fetched or loaded.

No baseline assistant scores, LoRA quality improvement, or full pretrained runtime compatibility are claimed. PEFT is unavailable locally. The adapter pilot diagnostic checkpoint contains optimizer/scaler/RNG progress but is explicitly **not a validated resumable SFT run format**. A tested pilot restart and compatible environment are required before longer adaptation.

The 1–3B pretrained track supplies a possible assistant/compression result independently of scratch completion. The 122.7M scratch model is a research vehicle; neither size nor a lower base-model language loss guarantees source-grounded instruction following. SFT and task evaluation remain necessary. Losses from different tokenizers cannot be compared directly with Stage 6 CE.

## 6. Pruning and quantization: actual first results

To avoid waiting for scratch completion, compression fixtures used a read-only copy of the trained Stage 6 +100M model: `stage_6_tinystories/runs/s6a/review_u12208_additional_100007936.pt`, SHA256 `da3d6c48a6292cbb75818a894d11a40f651c45aa9a90455d15e8ed957d5424a4`. The parent and all six original training-source identities were verified. All optimizer updates were confined to derived recovery models.

The fixtures physically shrink FFN tensors; they do not merely apply zero masks. Stage 7 coupled SwiGLU pruning is separately tested. All inference exports omit the same regenerable causal-mask buffers and optimizer state, so the dense baseline receives the same cleanup. Embeddings/tied head remain FP32. Every export was reloaded with identical logits.

See [Stage 8 results](../stage_8_compression/reports/RESULTS.md) for the seven-variant table, raw hashes, and measured protocols. CPU INT8 roughly halves export size with negligible change on the bounded language-loss fixture, but is slower on this machine. Pruning improves these short CPU timings modestly while damaging perplexity substantially. Packed INT4 storage is real, but computation dequantizes to FP32 and is slower; accelerated INT4 is unverified.

These are existing-model engineering fixtures, not results on a trained Stage 7 model, a pretrained 1.7B assistant, or the sealed study-assistant benchmark. No human assistant-quality improvement is inferred from them. Both pruning and quantization now have concrete executable paths before the 29 November deadline; recovery/data/recipe optimization and actual deployment measurements remain work.

## 7. Resources, launch preparation, and next gates

[resources.json](configs/resources.json) deliberately leaves confirmed GPU-hour allowance, maximum walltime, scratch/persistent quotas, new environment permission, and A100 availability unknown. Preserve at least 40% of future budget for adaptation/compression and 10% contingency; scratch allocation is at most 50%. Actual GPU-hour amounts cannot be assigned until the total allocation is known.

The proposed first benchmark is one V100 32GB, 4 CPUs, 20GB host RAM, 15 minutes: **0.25 GPU-hours**, with an explicit 4GiB working-storage allowance. These are requested limits, not granted resources. [benchmark.pbs.proposed](hpc/benchmark.pbs.proposed) requires confirmed allowances, a prepared manifest, new output directory, and checkpoint-probe directory on the intended checkpoint filesystem. `/tmp` save timings cannot predict Lustre overhead. It is not submitted or transferred.

[train.pbs.proposed](hpc/train.pbs.proposed) is a separate, unsubmitted template. The main run remains blocked by its unapproved configuration and unresolved resource/runtime gates. No 2–5B-token run is authorized or costed as affordable.

The eventual estimate must use measured successful training positions/second, development-evaluation time, full checkpoint save/restore time on intended storage, cadence, startup, and contingency. Rough training compute is 6 × parameter count × tokens; that is an order-of-magnitude planning aid, not a walltime measurement. CPU throughput and short-sequence proofs are excluded from GPU allocation decisions.

Next actions, in order:

1. Human-review the 20 draft cases, agree task boundaries/rubrics, and assign an independent sealed-set owner. Expand to the agreed 50/150 only under that workflow.
2. Confirm GPU hours, walltime, storage quotas, and allowed environment; run the target-host runtime probe and approve a compatible scratch/pretrained stack.
3. Review the bounded corpus sample and source rights/lineage; enlarge only within an approved cap. Freeze tokenizer and source-group splits before recipe selection.
4. Review and then run the bounded V100 benchmark, including full-context memory, FP16/scaler restart, evaluation overhead, and checkpoint filesystem overhead. Measure at least two feasible microbatch/accumulation settings under a fixed budget.
5. Use those measurements to decide a small scratch review gate under a fixed eventual LR horizon, and whether 2–5B tokens are affordable. Complete main-run accounting and immutable checkpoint preservation first.
6. Independently approve a compatible cached pretrained baseline and small SFT pilot, then repeat pruning/INT8 and supported INT4 measurements on that model with human-grounded quality review. No A100 dependency.

## 8. Verification, locations, and reproducibility limits

Re-run the focused suite from the repository root:

```sh
python3 -B -m unittest stage_7_general_lm.tests.test_stage7 stage_8_compression.tests.test_compression
```

[tests.txt](reports/tests.txt) contains the actual 31-test output. [syntax.json](reports/syntax.json) records grammar/shell checks. Tests cover the production model, loss masking and bounds, causality/cache equivalence, token-weighted accumulation, sampler/evaluation RNG isolation, fresh-process resume, malformed checkpoints, shard boundaries, high token IDs, bounded-prefix reads, launch gates, native EOS/PAD masking, physical/gated pruning, and INT8/INT4 export reload. No skipped GPU test is presented as passed.

Small JSON/text evidence is retained in `stage_7_general_lm/reports/` and `stage_8_compression/reports/`. The bounded raw data, tokenizer files, packed binary shards, discarded benchmark checkpoints, and derived compression exports remain under `/private/tmp/nanogpt-stage7-*` and `/private/tmp/nanogpt-stage8-20261104-final/`. **Temporary storage is not a durable archive.** Reports record hashes and paths; relocation to confirmed persistent storage requires a later storage/retention decision. Do not represent the JSON-only manifest as if its binary shards were bundled in the repository.

There is no final corpus, selected data mixture, trained Stage 7 model, pretrained assistant score, human-rated benchmark, proven GPU runtime, costed main allocation, or production launch approval at this milestone. The implemented and measured foundation makes those next decisions concrete while preserving Stage 6.
