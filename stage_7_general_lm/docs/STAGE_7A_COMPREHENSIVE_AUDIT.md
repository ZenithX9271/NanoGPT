# Stage 7A comprehensive audit — 16 November 2026

**Assessment: the from-scratch base-model milestone is complete and supported by reproducible evidence. The source-grounded study-assistant milestone is not complete.** We built and trained a 122,702,592-parameter decoder through its original 2B-position schedule, preserved its state across 16 training allocations, and measured substantial improvements in language-model loss. Generated text still contains factual errors, drift and repetition. Numerical validation does not establish assistant quality.

This audit inspected production code, corpus and evaluation contracts, the original GPU restart proof, every segment's checkpoint chain, the full final metrics history, local backups and current HPC artifact hashes. It ran the complete Stage 7 test suite and loaded the real final model for CPU inference. It did not launch training, alter trained weights, change frozen training code or modify Stage 6. The evidence directory is [comprehensive_audit_20261116](../reports/comprehensive_audit_20261116/).

| Workstream | Evidence-based state |
|---|---|
| Stage 7A architecture, data pipeline and training | Completed; checks described below passed |
| Final model and checkpoint preservation | HPC and Mac checkpoint hashes agree; six milestone backups verified |
| Language-model evaluation | Improved on matched development samples; final sampled heldout recorded |
| Source-grounded assistant quality | Unvalidated; human-reviewed evaluation and task adaptation remain |
| Stage 7B external pretrained assistant baseline | Implementation scaffolding exists; completed adaptation/evaluation not demonstrated |
| Stage 8 on this 122.7M model | Pending; existing measured compression fixtures use Stage 6 |

**What was actually built**

The production model is a causal decoder with 12 layers, width 768, 12 attention heads, context 1024, SwiGLU intermediate width 2048, pre-RMSNorm, rotary positions, dropout 0.1 and tied token embedding/output weights. Its vocabulary contains 49,152 IDs. The unique parameter count is **122,702,592**, within the methodology's 100–150M candidate range.

The model weights were trained from random initialization. Reusing the pinned SmolLM2 tokenizer does not make these pretrained SmolLM2 weights. “2B” refers to training positions; the architecture has approximately 123M parameters.

Inspection covered causal masking, padding, RoPE/cache positions, next-token target alignment, accumulation by valid target count, FP32 loss computation, FP16 scaling, clipping, evaluation RNG separation and full-state restart guards. No correctness blocker was found in these inspected paths. This is a code-and-evidence audit, not a formal proof or an exhaustive test of every possible input.

**Data preparation and isolation**

The earlier extraction and packing jobs prepared data; they did not train the model. Extraction read two pinned Parquet files into normalized document records. Packing tokenized them, removed normalized exact duplicates, assigned source groups to splits and wrote binary token streams with a manifest and completion marker.

| Corpus quantity | Verified value |
|---|---:|
| Raw Parquet transfer | 3,329,310,703 bytes |
| Extracted documents | 482,019 |
| Retained packed documents | 477,445 |
| Packed tokens across all splits | 454,195,401 |
| Training tokens | 363,787,512 |
| Development tokens | 44,677,142 |
| Heldout tokens | 45,730,747 |
| FineWeb-Edu tokens, all splits | 354,195,407 |
| Cosmopedia-v2 tokens, all splits | 99,999,994 |

The 500M setting was a maximum corpus cap, not a guaranteed output size. FineWeb-Edu stopped below its 400M cap after the bounded extraction input was consumed. Cosmopedia reached its 100M cap to within six tokens. The retained training mixture is approximately 78% FineWeb-Edu and 22% Cosmopedia by tokens.

The exact-dedup pass removed 264 documents; another 4,310 Cosmopedia documents were excluded by its token cap. The larger `unique_normalized_documents=481755` field counts documents before token-cap exclusions, so it is not the retained-document count.

The recorded corpus audit passed document-ledger reconciliation, token bounds, shard hashes, counts and split group isolation. On 16 November, all six binary shards, three group-index files and the compressed document ledger on HPC were independently rehashed and still matched that audited manifest. The manifest and completion marker agree:

`fa40be03f50c9d7bd4cd3bcf33df1538a55f62e9b7760b845ec17a04fe7ade65`

The tokenizer bundle also passed strict offline checks. Its token-JSON SHA256 is:

`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`

These controls establish the documented exact-content and configured-group separation. They do not establish semantic deduplication, independence of every synthetic seed/derivative document, or absence of all benchmark contamination. The inputs were sequential bounded portions of two pinned files, not representative random samples of the complete source datasets.

The sampler draws random windows with replacement. The final 2B exposure is approximately **5.50 times the training-corpus volume**, rather than 2B distinct tokens or 5.50 guaranteed complete passes through every document. Repeated-data training can remain useful, but its returns depend on data and repetition; more repetitions are not equivalent to acquiring fresh information. See the primary study [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264).

**Training continuity and resource accounting**

The original production GPU gate is now preserved in [original_gpu_gate](../reports/comprehensive_audit_20261116/original_gpu_gate/). On a V100 with the production architecture, uninterrupted and resumed runs produced equal update events and semantic checkpoint state, and identical final checkpoint hashes. The gate candidate and actual training configuration have identical model, training and evaluation dictionaries; their resource/status metadata differ.

Training used V100 FP16, microbatch 4, accumulation 4 and 1024-token contexts: **16,384 target positions per successful optimizer update**. AdamW used betas 0.9/0.95, weight decay 0.1 and gradient clipping at 1.0. The learning-rate schedule warmed up for 2,000 updates and decayed from 0.0003 to 0.00003 over the original 2B horizon.

The fresh audit checked all 122,117 update records in the final full-history log. Attempts are contiguous; successful-update counters advance only on accepted optimizer steps; processed and successful positions reconcile exactly. Every logged LR agrees with the original schedule to numerical precision. The endpoint records for all 16 allocations agree with those log counters, and every continuation names the preceding endpoint's checkpoint hash and counters. All 16 saved allocation exit codes are zero. The current 15 core source files match the original frozen code identity.

| Final accounting | Value |
|---|---:|
| Successful updates | 122,071 |
| Successful training positions | 2,000,011,264 |
| Attempted updates | 122,117 |
| Skipped AMP attempts | 46, or approximately 0.0377% |
| Processed positions including skips | 2,000,764,928 |
| Final LR | 0.00003 |
| Sixteen recorded training allocations | 35.3931 GPU-hours |
| Additional standalone 1B review | 0.0699 GPU-hours |
| Recorded cumulative total | 35.4630 GPU-hours |

The slight overshoot above 2B comes from whole optimizer updates. Occasional AMP skips were accounted for and did not falsely advance successful exposure. The final summary is `completed`; intermediate segment summaries are `review_gate`. The four final allocations used 8.6962 hours against their authorized ten-hour additional budget.

These are allocation-ledger upper bounds for the documented training/review chain, not an account-wide bill. Earlier benchmarks, failed environment jobs, interactive allocations and queue waiting are outside this sum. No claim can be made from these records that a specific fraction of the project's total allocation remains reserved for adaptation/compression. Later wrappers and PBS enforced the authorized segment/cumulative limits; the frozen trainer's positive resource-limit checks alone are not a complete cumulative-budget enforcement mechanism.

The setup delays came from several distinct operational issues: module `PYTHONPATH` replacement hiding Torch, MKL/OpenMP conflicts, missing SSL libraries for preparation Python, offline wheel transfers, terminal/SSH reconnects, and two postprocessing/audit failures. Packing had already produced valid outputs before the missing-`jq` summary command caused exit 127. The group-index ordering check was corrected and its rerun passed without regenerating the corpus. These incidents explain friction; the later artifact and continuity checks support the completed training independently of them.

**How much the model improved**

The following are matched FP32 evaluations: the same seeds, context, 512 windows and 524,288 scored positions per stream. The fresh audit rechecked the protocol fields and the CE-to-perplexity arithmetic from the saved reviews.

| Training exposure | Development perplexity |
|---|---:|
| 100M | 57.7174 |
| 200M | 43.4585 |
| 500M | 34.9799 |
| 750M | 32.0867 |
| 1B | 29.9033 |
| 1.5B | 26.0236 |
| 2B | **24.1660** |

Development perplexity decreased **58.13% from 100M**, **30.91% from 500M**, **19.19% from 1B**, and **7.14% from 1.5B**. Both source domains improved in the final continuation: Cosmopedia dev reached 11.5852 and FineWeb-Edu dev 29.0077. Final matched train perplexity is 21.9307.

The final-only heldout sample has perplexity **24.1668** and CE 3.1849792 over 524,288 positions, using a separate seed. It agrees closely with sampled development loss. This is encouraging evidence about this corpus distribution, not a full heldout sweep or proof against all overfitting. The expanded train/dev CE gap is 0.0971. No multi-seed replication or architecture/data ablation establishes that this recipe is optimal.

The routine 16-window evaluations are a separate protocol: their final dev perplexity is 25.6633. They must not be mixed into the larger-window comparison. Likewise, Stage 6 and Stage 7 perplexities cannot be interpreted as a direct quality ranking across different corpora/tokenizers.

Perplexity measures token prediction under the scored distribution. A 58.13% perplexity reduction is **not** a 58.13% factual-accuracy or assistant-quality improvement. The [existing evaluation figure](../reports/review_2b_20261116/evaluation_500m_2b.png) keeps the two protocols separate.

**Quality limitations and unfinished acceptance criteria**

The saved eight-prompt greedy/sampled generations still include loops, invented quantities, confused scientific explanations and topic drift. The actual 2B checkpoint was also loaded on CPU during this audit. Its photosynthesis continuation claims that chlorophyll helps absorb water and nutrients from air, illustrating the remaining factual problem. This short diagnostic is not an additional scored benchmark.

The model can produce plausible local English and learned subject vocabulary. It has not demonstrated reliable source-grounded answering, faithful summarization, missing-information refusal or entity-state tracking. The next-word objective and this web/synthetic corpus do not by themselves train those behaviors. Supervised demonstrations are a separate adaptation step; instruction-tuning research motivates that separation but does not guarantee that our small model will become factually reliable. See [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).

The project's own [evaluation contract](../configs/evaluation.json) still states `model_selection_ready=false`, proposed/unapproved thresholds, no sealed-test hash and no test owner. There are 20 draft development cases and four draft SFT records, not the agreed 50 human-reviewed development cases and independently sealed 150-case final set. Draft assistant-authored references are not human acceptance. Therefore the numerical review's PASS cannot be used to claim a completed study assistant or superiority to an external model.

**Actual final model, backups and preservation**

The final endpoint is `review_u122071_positions_2000011264.pt`, 1,472,538,359 bytes. Both its current HPC copy and local Mac copy hash to:

`f32b6a493d3aab92c5f3d9a5ff482f856f5a21dccfaa38780c4ba7a1e61a6791`

The CPU smoke check strictly loaded the full state dictionary, counted 122,702,592 unique parameters, confirmed tied embeddings and finite parameters, and checked all 86 Adam states for finite moments and step 122,071. Real-model cached and full-prefix next-token logits agreed within the stated tolerances; maximum absolute difference was 0.0000128746. It generated a short continuation without any optimizer updates. This confirms CPU load/inference portability in the observed Torch 2.8.0 environment; it does not claim cross-runtime bitwise GPU training restart. The original resume guards were not relaxed.

Fresh local backup checks passed for 100M, 200M, 500M, 1B, 1.5B and 2B. The three newer bundles were checked against their full backup manifests (69, 76 and 81 files respectively). For the older three, checkpoint bytes/hashes, the 15 core source identities, serialized config and four tokenizer-lock files were checked (21 files each). All 72 files in the saved Stage 6 preservation baseline remain present and byte-identical.

The Mac copies include corpus metadata, not a durable mirror of the complete raw/packed training corpus. Those files were verified on HPC scratch; their retention and a long-term corpus archive remain separate from checkpoint preservation. Google Drive synchronization completion is not independently verified. Stage 7 and Stage 8 currently appear as untracked Git directories: hashed snapshots support recovery, but a clean source-only versioned release remains to be prepared. Do not commit bulk checkpoint files as source code.

**What should happen next, and why**

1. **Keep the completed 2B model as the immutable dense baseline.** Additional pretraining should be a new costed experiment with an explicit data/schedule hypothesis. Repeating the same corpus merely to reach a larger number is not the next acceptance criterion.
2. **Complete the capability evaluation.** Human-review the 50-case development set, define approved scoring/thresholds and assign an independent owner to seal the 150-case final set. This provides a way to measure the behaviors the project actually promises. Do not use final answers for recipe selection.
3. **Measure the base model and a clearly identified external pretrained baseline on the same permitted task protocol.** Tokenizer-dependent perplexity is not a fair cross-model task comparison. This establishes how much adaptation is needed and a realistic task-quality reference.
4. **Prepare a bounded, separately versioned task-adaptation experiment.** Use curated source-grounded training examples, disjoint source groups, answer-only labels and answerable/unanswerable cases. Keep the original 2B artifact. The existing scratch trainer deliberately rejects masked scratch updates and changed resume identity; simply changing its token budget or reusing its completed optimizer is not a validated SFT route. A separate custom-model adaptation path needs implementation and checks. The external-model adaptation entry point is a distinct experiment. Stop or revise based on development task results rather than assuming SFT fixes knowledge errors.
5. **Apply Stage 8 to a suitable actual dense parent and report measured trade-offs.** Existing compression results are Stage 6 fixtures. They establish some tooling, not pruning/quantization quality or speed on this 122.7M endpoint. In fact, the recorded INT8 fixture was smaller but slower, and pruning materially hurt its small loss probe. Benchmark inference-only exports fairly, preserving tied weights and measuring task quality, loss, size, memory and latency on named hardware. Optimizer removal alone is not model compression. The methodology's November 29 delivery target makes this work urgent alongside task evaluation.

The next valuable result is a defensible task-quality comparison and an actual compression trade-off on the chosen model. We have completed the base-pretraining engineering objective; the remaining work is to establish and improve useful, reliable behavior and deployment properties.

**Audit evidence and limits**

- [44-test log](../reports/comprehensive_audit_20261116/tests.txt): all passed, no skips, strict native tokenizer required; local CPU run.
- [Records and backup integrity](../reports/comprehensive_audit_20261116/records_integrity.json), with [read-only verification script](../reports/comprehensive_audit_20261116/verify_records.py).
- [Current HPC corpus hashes](../reports/comprehensive_audit_20261116/remote_corpus_integrity.json) and [final checkpoint hash](../reports/comprehensive_audit_20261116/remote_final_checkpoint.json).
- [Actual final-model CPU result](../reports/comprehensive_audit_20261116/final_model_cpu_smoke.json), with [inference verification script](../reports/comprehensive_audit_20261116/verify_final_model.py).
- [Matched evaluation history](../reports/comprehensive_audit_20261116/matched_evaluation_history.json), [final review](../reports/TWO_BILLION_FINAL_REVIEW.md), and [qualitative generations](../reports/review_2b_20261116/generation_comparison_1p5b_2b.md).
- Original GPU gate and the recovered 350M endpoint/allocation records are preserved in this audit's evidence directory. The SSH session expired after the artifact checks; a new live queue listing was unavailable. Completion is established by saved final counters, endpoint/review records and current artifact verification, not inferred from a blank queue listing.

No new GPU jobs or model updates were performed for this audit. No assistant-accuracy percentage, optimal-training claim, production-readiness certification or compressed-model speedup is supported by the present evidence.
