# NanoGPT: Stage 7 and Stage 8 execution methodology

Prepared: 3 November 2026, Asia/Kolkata.

## 1. Decision and scope

Adopt the working agent's proposal with the refinements below. Proceed with a bounded implementation milestone, not an unconditional billion-token training launch.

The selected first capability is already settled: **a source-grounded English study assistant**. Do not reopen that decision unless the user changes it.

Maintain two complementary tracks:

| Track | Purpose | Principal claim |
|---|---|---|
| Stage 7A: from-scratch general language model | Develop our own approximately 100–150M-parameter model and a scalable pretraining recipe | We trained this model from random initialization on a documented corpus |
| Stage 7B: pretrained assistant adaptation | Obtain a capable task-specific reference and improve source-grounded behavior | We adapted an explicitly identified external pretrained model |
| Stage 8: compression and inference | Complete actual pruning and quantization experiments before November 30 | Measured quality, size, memory and latency trade-offs on stated hardware |

Stage 7A and 7B share the capability evaluation, but have separate initialization, tokenizer, training histories and budgets. Parallel development does not imply simultaneous GPU allocations.

Our own 1B-parameter model remains a later resource-gated objective. Neither a token count nor a parameter count establishes assistant quality. Compute-optimal scaling research guides resource allocation under its experimental assumptions; it is not a product-quality guarantee. [Chinchilla](https://arxiv.org/abs/2203.15556)

**Both pruning and quantization are required November workstreams.** They must not be postponed until the larger from-scratch model finishes or until A100 access appears. Start their implementation on the existing validated model, then apply the pipeline to the strongest compatible dense baseline available.

## 2. Current project facts and boundaries

Canonical repository:

    /Users/shash/Library/CloudStorage/GoogleDrive-shash9271@gmail.com/My Drive/Currently Doing/NanoGPT

Current Stage 6 authority: [stage_6_tinystories/docs/RESULTS.md](stage_6_tinystories/docs/RESULTS.md).

Stage 6 completed 12,208 continuation updates, 100,007,936 additional positions and 198,311,936 total positions. Its immutable endpoint is:

    stage_6_tinystories/runs/s6a/review_u12208_additional_100007936.pt
    SHA256 da3d6c48a6292cbb75818a894d11a40f651c45aa9a90455d15e8ed957d5424a4

Its comparable final development CE/perplexity are approximately 2.048690 / 7.757731. This establishes improved next-token prediction, not solved narrative coherence.

The repository has since been consolidated. Current complete evaluation directories are evaluations/25m, 50m, 75m and 100m under Stage 6. The November 1 report and old chat paths are historical. Do not recreate removed duplicate evaluation folders, removed redundant snapshots or pre-run archive copies merely to match older messages.

The Stage 6 deployment ledger has already been corrected and moved under stage_6_tinystories/deployment. Do not repeat an obsolete repair from an earlier handoff. Read the current transfer guide when deployment is needed.

Keep Stage 6 code, configurations, tokenizer, trained artifacts and reference evaluations unchanged. Create new stages for new training recipes and compression exports. Do not weaken its identity guards or edit old checkpoint metadata to force compatibility.

## 3. Hardware and resource policy

Verified HPC baseline:

- Tesla V100, 32 GB VRAM.
- Python 3.7.9, custom PyTorch 1.10.0a0 build, CUDA 11.0.
- PBS, historically one GPU with four CPUs and 20 GB host RAM; verify the actual requested resources for each job.
- Working module command: module load apps/pytorch/1.10.0/gpu/intelpython3.7.

A100 availability, GPU count, 40/80 GB capacity and GPU-hour allocation are **unknown**. A newer HPC software environment is also unverified. The plan must work without assuming either.

Deployment targets are both local Mac/CPU and, conditionally, A100. V100 remains the verified GPU for training and measurements until another allocation is observed.

Before a long run, record actual limits for GPU-hours, maximum job walltime, disk/cache usage and checkpoint storage. Until these are known, prepare launch commands and cost estimates; do not substitute an invented allocation.

Suggested planning split once GPU-hours are confirmed: at most 50% for from-scratch pretraining, at least 40% reserved for adaptation/compression, and 10% contingency. Adjust before launch if measurements justify it; do not let scratch training silently consume the compression reserve.

Bulk caches, raw data and large transient artifacts belong in quota-checked working storage, not duplicated repeatedly in the cloud-synced source tree. Scratch storage is not assumed persistent: identify which final assets must be retained before access expires.

## 4. First capability and evaluation contract

The assistant must answer, summarize and explain a supplied English passage, retain its facts, and distinguish missing information from supported inference. Broad pretraining remains general English; this capability anchors post-training and product evaluation.

Example:

    Maya placed her notebook in a drawer. Arun later moved it into his bag.
    Where is the notebook? -> In Arun's bag.
    Why did Arun move it? -> The passage does not say.

### 4.1 Initial evaluation set

Use the proposed 200 cases with source-group-level splits:

| Task | Development | Sealed final test | Total |
|---|---:|---:|---:|
| Questions answered from supplied text | 15 | 45 | 60 |
| Faithful summaries | 10 | 30 | 40 |
| Explanation/simplification | 8 | 22 | 30 |
| Missing/ambiguous information | 7 | 23 | 30 |
| Entities, object state and causal relations | 10 | 30 | 40 |
| Total | 50 | 150 | 200 |

Related passages, questions and paraphrases belong to the same split. Prefer newly authored passages and controlled fact variants for at least part of the evaluation. Include names, locations, temporal updates, negation, numbers and information that is deliberately absent.

Specify references, acceptable answer alternatives, supporting spans/sentence IDs, ambiguity labels, source IDs, intended difficulty and task category. Check that source, template, question and allowed response fit the input budget under every compared tokenizer. Never truncate the answer-bearing fact silently.

Stage 6 is a short-context base model, not a trained assistant. Keep its completion evaluation as a reference; do not treat failure to follow a new chat template as a fair post-training comparison.

### 4.2 Scoring

- QA and state tracking: reference correctness with declared acceptable alternatives.
- Evidence: the cited text must exist and actually support the answer. Citation existence alone is insufficient.
- Missing information: correct uncertainty, while also measuring over-refusal on answerable questions.
- Summaries/explanations: important-point coverage, faithfulness, preservation of qualifiers/negation, and clarity.
- Unsupported assertions: report their frequency separately from fluency.
- Score outside knowledge as unsupported when the task requires evidence from the supplied passage, even if the statement is true elsewhere.

Use anchored 0–4 rubrics where exact scoring is unsuitable. A fluent but factually contradictory response must not receive a high faithfulness score. Define what each score means with examples before model selection.

Use one fixed deterministic decoding protocol for the primary study-assistant comparison. Keep stochastic story diagnostics as a separate, unchanged protocol. Native chat templates and EOS IDs must be documented per model.

Before the first model-selection run, record the actual acceptance thresholds in the evaluation specification. Useful quantities include QA accuracy, state-tracking accuracy, correct uncertainty, over-refusal and unsupported-claim rate. Thresholds are product goals, not predictions of what the model will achieve. Do not invent an aggregate accuracy by averaging unrelated rubric scales.

Report denominators and uncertainty. With only 50 development and 150 test cases, avoid broad superiority claims from a handful of wins. If several cases share a source, account for that grouping in uncertainty estimates.

### 4.3 Sealed test and contamination

Use development data for model/recipe selection. Keep sealed test content and labels out of the training/tuning workflow; an independent owner or controlled final-evaluation process should hold them. Record hashes and a release policy. Once inspected for decisions, a set is no longer an untouched test set.

Exclude evaluation passages and known derivatives from all data controlled by this project: pretraining, SFT, teacher-generated examples, recovery and calibration. A sentence paraphrased from an evaluation passage is not automatically independent.

**We cannot prove that an externally pretrained model never saw a public source passage.** Record this limitation rather than claiming complete contamination elimination. Newly authored passages and counterfactual fact variants help reduce direct memorization concerns.

Human evaluation should hide checkpoint names, randomize paired order, allow ties, and distinguish unsupported claims from stylistic preference. Use independent raters where available. Model-assisted judgments remain labeled as such; blank ratings are not human evidence.

## 5. Runtime compatibility before expensive work

Create one capability report separating observed, failed and unknown items:

- GPU name, count, VRAM and compute capability.
- Driver, active PyTorch/CUDA build, Python, OS and glibc.
- CPU architecture/RAM and available quantization engines on the Mac.
- FP32/manual attention path; FP16 training/scaler support; optional optimized attention.
- Target pretrained checkpoint/tokenizer loading, adapter training and serialization.
- Quantization/pruning method, required kernels and actual eligible devices.

Preserve the Stage 6 environment. Test new dependencies in a separate pinned environment. Do not upgrade the working module, driver or host libraries in place. The CUDA number printed by nvidia-smi is not sufficient evidence of the active framework/toolkit compatibility.

V100 does not provide native BF16 support. Its supported FP16 path may be tested separately; FP32 remains the reference. Standard FlashAttention-2's documented NVIDIA support does not make it a V100 dependency. An available SDPA API also does not prove that a particular flash kernel executed. [FlashAttention documentation](https://github.com/Dao-AILab/flash-attention)

Quantization eligibility is method-specific. Current bitsandbytes documentation lists Python >=3.10, PyTorch >=2.4 and CUDA build requirements starting at 11.8; it lists compute capability 7.5+ for LLM.int8(), but 6.0+ for NF4/FP4. Therefore the currently verified HPC stack is not an automatic supported path, and V100 eligibility for one method does not establish eligibility for another. Recheck the pinned release and host requirements. [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/main/en/installation)

The gate is a small successful load/forward/backward/save/restart demonstration on the intended stack, including adapters or scaler when used. A successful installation is insufficient. Prepare this as one bounded execution, not a series of redundant validation jobs.

If a modern GPU environment is unavailable, keep a portable V100/manual training path and physical-pruning path. Use the verified local CPU stack for INT8 experiments. Do not present CPU quantization as a working CUDA backend.

## 6. Bounded corpus and tokenizer selection

### 6.1 Initial sample

Begin with a reproducible sample, not the full corpus. Proposed first-sample ceilings: 25,000 documents, 1 GiB retained normalized text, 4 GiB network/cache transfer and 30 minutes CPU preprocessing, stopping at the earliest limit. Reduce these if the actual storage quota requires it. Streaming does not by itself guarantee bounded network or cache usage.

Candidate families:

- Educational/general English text.
- Explanatory or textbook-style text.
- A smaller retained story domain.
- Code only when it serves a defined evaluation objective.

FineWeb-Edu and SmolLM-Corpus are useful source candidates. SmolLM-Corpus's FineWeb-Edu-Dedup is a subset of FineWeb-Edu: do not count both as independent mixture components without resolving overlap. Record synthetic-source parentage where available. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)

For each document retain source/revision/ID, license/provenance, normalized text hash, filtering decisions, sampling seed and split. Group known originals and derivatives before splitting. Deduplicate across selected sources and protect evaluation source groups. If derivative lineage is unavailable, state the limitation.

Choose initial mixture weights from the sample and objectives. Record whether weights are document-based or token-based; prefer explicit token sampling weights for the training budget. Do not tune many mixtures before establishing one reliable reference.

### 6.2 Tokenizer decision

Compare the existing 8K tokenizer as a reference, one established general-text tokenizer, and a newly trained approximately 32K tokenizer only if useful. Do not spend the whole first milestone training several tokenizers when an existing one meets the criteria.

Measure tokens per UTF-8 byte/word by domain; fragmentation of names/numbers/technical terms; round trips; context truncation; embedding/output cost; and measured throughput. Train a new tokenizer only on the training split.

The pretrained-model track retains its native tokenizer, special IDs and chat template. Do not transplant Stage 6's PAD/BOS/EOS IDs into it.

Published dataset token counts must be recounted under our selected tokenizer. Select token-storage dtype from the actual ID range, including added special tokens. A tokenizer with IDs above 65535 cannot safely use the existing uint16 representation; use an appropriate wider dtype and test boundary IDs.

## 7. Stage 7A: one from-scratch candidate

Provisional candidate, to freeze after tokenizer and runtime selection:

| Component | Proposal |
|---|---|
| Learned parameters | Approximately 100–150M; count unique parameters after implementation |
| Blocks / width / heads | 12 / 768 / 12 |
| Context | 1024 initially |
| Vocabulary | Candidate around 32K, subject to measured selection |
| Embeddings | Tied |
| Default decoder recipe | Pre-RMSNorm, RoPE, SwiGLU, bias-free attention/MLP projections; tied token embedding/output projection |
| Feed-forward intermediate width | 2048 for the width-768 SwiGLU candidate |
| Initialization | Random initialization, explicitly separate from Stage 6 |

This proposed decoder is approximately 110M parameters with a 32K vocabulary; report the exact unique count from the implemented model. Do not call it 125M simply because GPT-2 used that label. Use this as a documented starting recipe, not a claim of optimality. If compatibility or benchmark evidence requires an alternative, record the replacement before the main run instead of opening a large architecture search. The normalization, positional and gated-MLP choices follow established decoder practice. [LLaMA](https://arxiv.org/abs/2302.13971)

Avoid a simultaneous architecture grid. A smaller control is optional and must have a specific comparison purpose and its own measured budget. A new tokenizer, corpus, architecture and context constitute a combined system change; report that scope honestly.

Implement efficient batched attention projections and portable attention. Avoid duplicating large causal masks per head in the new implementation. Preserve a reference computation and validate causal behavior. Enable newer optimized backends only after capability and numerical checks.

Select effective batch size in tokens; use microbatching/accumulation when required. Design optimizer/LR settings for new pretraining rather than blindly copying Stage 6's continuation LR.

## 8. Benchmark, pilot and training-budget gate

### 8.1 Benchmark-only mode

Benchmark the actual configuration, data path, precision and accumulation settings. Keep temporary updates separate from quality-training artifacts and discard them explicitly.

Use bounded warmup and several measured windows. Synchronize CUDA at timing boundaries; do not infer throughput from unsynchronized single-step clocks. Measure data loading, forward/backward, accumulation, clipping and optimizer work.

Report tokens/s distribution, microbatch/effective batch, GPU allocated/reserved peaks, host memory, fixed evaluation time, full checkpoint size/save/restore time, startup and runtime. Probe FP16 only after correctness checks; benchmark context 2048 only if resources justify it.

For a training-only throughput R:

    estimated walltime = tokens/R + evaluation overhead + checkpoint overhead
                         + startup + walltime margin

For effective throughput already measured with scheduled overhead included, do not add the same overhead again. Report GPU-hours and calendar exposure separately; queue access is not guaranteed. Do not extrapolate Stage 6's small-model throughput directly to 100M or 1B.

### 8.2 Avoid an unnecessary double pilot

Preferred sequence: benchmark first, select an affordable main horizon, then make the first bounded 50–100M positions a review gate on that same main trajectory. This preserves useful training exposure and the unchanged LR horizon.

If an independent short-schedule pilot is needed before a main horizon can be chosen, label it as a separate experiment. Do not silently extend its completed short LR schedule into a 2–5B main run. A shorter smoke test may verify functionality but cannot establish final assistant quality.

A 2–5B-position main run is conditional on measured cost and confirmed hours. Reject a budget that consumes the November compression reserve. Do not use an undertrained pilot's chatbot behavior to conclude that the architecture is incapable.

### 8.3 Main-run contract

Before launch freeze architecture/tokenizer, data revisions/mixture, effective batch, precision, optimizer, LR horizon, seed, evaluation cadence, checkpoint cadence and resource caps.

Review gates do not shorten the LR horizon. Changes in budget, recipe or hardware are documented experiment changes. V100-to-A100 migration must not be described as an independently proven bit-identical trajectory.

Every checkpoint must restore optimizer, RNG, sampler/mixture state, counters and AMP scaler where applicable. Save at optimizer-update boundaries unless partial gradients and accumulation state are explicitly supported and tested.

FP16 overflow may skip an optimizer step. Distinguish processed positions, positions contributing to successful updates, microsteps, completed optimizer updates and scheduler progress. Declare which token counter sets the budget/schedule. Log overflow/skips; do not reuse the Stage 6 identity of processed tokens = successful updates times batch size without verifying its assumptions.

## 9. Evaluation of base and post-trained models

Base-model gates use fixed held-out next-token scoring by source/domain, completion tasks, retained story evaluations and numerical stability. Assistant gates apply after instruction adaptation.

Across tokenizers, raw token CE/perplexity is not directly comparable. If reporting bits per byte, compute summed content NLL divided by log(2) and scored UTF-8 bytes on identical held-out text, with a documented document-reset, special-token, context and stride policy. Different conditioning contexts still need disclosure. Common task outcomes remain essential.

Keep a small stable regression suite for early entity/location errors, longer temporal updates, ambiguity and answerability. Do not repeatedly train against the exact three Stage 6 manual prompts.

Use development evaluations to select models. Evaluate the locked candidates on the sealed final set at the final review, then stop tuning against that set. An unchanged repeated test set is not automatically an untouched test set.

## 10. Stage 7B: pretrained assistant baseline and adaptation

Start with an instruction-tuned checkpoint as the primary product baseline. SmolLM2-1.7B-Instruct is a concrete compatibility candidate, not an assertion of current best performance. Record the exact model revision, configuration, license and tokenizer. [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

Keep a base-checkpoint SFT experiment optional and separately labeled. Do not use its weaker initial instruction following to inflate the apparent improvement over a capable instruct reference.

Sequence:

1. Evaluate the unmodified instruct model using its native chat template.
2. Establish curated source-grounded SFT examples with disjoint source groups.
3. Run a small adapter/SFT pilot, then select a bounded training budget.
4. Evaluate adapted versus unmodified checkpoints on identical development cases.
5. Freeze the stronger validated dense reference for compression.

Use answer-only labels for supervised response training, with padding and prompt/source tokens masked as intended. Validate truncation and special IDs on actual batches. LoRA reduces trainable-state memory; it does not by itself compress the deployed base model.

Include answerable and unanswerable examples, state changes, negation, evidence-grounded explanations and summaries. Teacher outputs require validation against the passage; a confident teacher answer is not ground truth.

If teacher/student tokenizers differ, use sequence-level text supervision or an explicitly designed alignment method. Do not apply naive vocabulary-wise KL between unrelated token IDs.

Avoid materializing several full CPU copies of a 1–3B model on a memory-limited Mac. Account for loading, conversion, quantization and adapter-merge peaks, not only steady-state model weights.

Post-training is a separate lever from model scale. Published InstructGPT results concern post-training and a particular prompt distribution, not a universal ranking by parameter count. [InstructGPT](https://arxiv.org/abs/2203.02155)

## 11. Stage 8: required compression experiments

Start on the existing model as a development fixture while Stage 7 runs. The substantive comparison should use the strongest compatible dense baseline available in time, preferably including the pretrained assistant. Compression aims to retain behavior; it is not expected to repair a weak dense model's factual reasoning.

### 11.1 Minimum experiment matrix

| Variant | Required evidence |
|---|---|
| Uncompressed dense reference | Quality, inference export bytes, memory, latency |
| CPU INT8 PTQ | Actual quantized execution and the same measurements |
| Structured FFN pruning, mild level | Physically smaller matrices; pre/post-recovery measurements |
| Structured FFN pruning, second level | Same checks; retain negative results |
| INT4 study | Packed representation and quality; actual execution path and speed limitations stated |
| Selected pruning + quantization combination | Measured combined trade-off against its dense parent |

Start pretrained-model pruning conservatively, for example 10% and 25% FFN-unit removal. More aggressive rates are optional. Development fixtures can use larger rates to exercise the implementation. These are candidate levels, not promised quality-preserving settings.

Pruning is not optional merely because the first result loses quality. A documented negative result and recovery attempt are valid experimental outcomes; do not invent a successful speedup to satisfy a deadline.

### 11.2 Physical structured pruning

For a two-linear-layer FFN, remove selected first-layer output rows/bias entries and matching second-layer input columns. For a gated MLP such as SwiGLU, remove coupled gate/up rows and corresponding down-projection columns. Inspect the actual native model before selecting groups.

Keep residual width, tokenizer and vocabulary unchanged initially. Handle tied weights deliberately. Do not prune attention heads by changing a configuration number without updating all coupled projections and cache metadata.

Rebuild the compact model and optimizer state appropriately for recovery; old moments of incompatible shapes cannot be loaded blindly. Record the dense parent hash, retained-unit indices, new shapes and actual whole-model parameter reduction. Removing 25% of FFN units does not mean removing 25% of total parameters.

Evaluate immediately after pruning and after bounded recovery/distillation. Tune importance and recovery only using permitted training/development data.

Masked zeros alone do not demonstrate a smaller dense computation. PyTorch's standard pruning reparameterization retains tensor shapes; physical compaction or a suitable sparse execution backend is needed for the corresponding deployment claim. [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)

### 11.3 Quantization

Begin with a working CPU INT8 path and complete-model validation. The earlier Mac probe only established an INT8 Linear operation, not complete-model performance.

Keep embeddings and tied output weights high precision initially if necessary. If conversion breaks a tie or duplicates a weight, account for it explicitly. Quantizing a shared head does not automatically reduce total storage when a high-precision embedding copy is still retained.

Add groupwise INT4 or a calibrated method after INT8 is reliable. Methods that require calibration use training-only samples, never sealed test data. Dynamic INT8 and simple weight-only methods do not all require an offline calibration set.

Save actual packed low-bit weights and required scales/metadata. Quantize-dequantize-to-FP32 followed by ordinary matmul is a numerical reference, not evidence of accelerated low-bit inference. State whether an INT4 implementation is a storage/quality reference or a supported optimized runtime. [GPTQ](https://arxiv.org/abs/2210.17323), [TorchAO inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)

A100 2:4 sparsity is an optional hardware extension only after allocation and kernel support are confirmed. It is not the November baseline dependency. [NVIDIA sparse Tensor Cores](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt)

### 11.4 Quality and success criteria

Declare acceptable degradation on development data before choosing a compressed variant. A possible initial target is <=2% relative perplexity increase, together with no material loss of source-grounded task quality. This is a proposed tolerance to review, not a guarantee or a substitute for the assistant evaluation.

Select a quality–size–speed trade-off using actual results. If compression improves size but slows the target workload, report that outcome. Avoid transferring a large-model paper's claimed accuracy preservation to our model without measurements.

## 12. Inference optimization and fair benchmarking

Implement and test KV caching as a separate inference improvement. Validate cached/uncached logits, causal behavior, EOS handling and native position rules. With Stage 6's learned absolute positions, a sliding window rebases positions; naive cache eviction is not equivalent. Rebuild the window or implement a validated equivalent strategy.

Report CPU and GPU results separately, each against its own comparable dense baseline. Test fixed prompt/output lengths and batch sizes for timing, with warmup and synchronized GPU boundaries. Measure prefill, fixed-length decode, median/p95 latency, throughput and peak memory. Keep quality decoding separate from fixed-work timing; an early EOS must not manufacture a speed advantage.

Measure inference-only exports, runtime RAM/VRAM, unique learned parameters, buffers, and quantization metadata. Stage 6 learned FP32 weights occupy approximately 26.3 MiB; its implementation additionally stores 12 MiB of causal-mask buffers. Dropping optimizer states or regenerating masks must not be misreported as pruning or quantization gains. Apply equivalent export cleanup to all compared formats.

Record model/export hashes, device, library/kernel versions, precision, prompt lengths, output lengths, batch size, cache settings and excluded setup costs. Do not promise 4x/8x whole-model compression or speedups from bit width alone.

## 13. November schedule and fallbacks

Target delivery is November 29, leaving November 30 out of the critical path.

| Dates | Required outcome |
|---|---|
| Nov 3–6 | Capability/rubric/split contract; runtime report; bounded data/tokenizer comparison; early CPU INT8 and physical-pruning fixture |
| Nov 7–12 | Candidate benchmark and affordable training gate; pretrained instruct baseline and small adaptation pilot; continue compression independently |
| Nov 13–20 | Freeze a suitable dense parent; pruning, recovery and INT8/INT4 experiments |
| Nov 21–26 | Selected combined variants, paired quality review and device-specific benchmarks |
| Nov 27–29 | Locked final evaluations, exports, reproducible instructions and result tables |

If scratch training does not fit confirmed hours, reduce or defer that run and report the limitation. Preserve the compression allocation. If the larger model is not demonstrably better in time, use the strongest validated existing/pretrained baseline for Stage 8.

If the 1.7B candidate lacks a supported runtime, verify another suitable checkpoint or adjust the execution environment through a separate approved setup. Do not claim that a tiny fixture satisfies the intended billion-scale adaptation result. A100-specific measurements remain conditional on access.

Do not add RLHF/PPO, agents, retrieval infrastructure or a distributed-training framework to the November critical path without a measured need. The chosen product initially receives its passage directly.

## 14. First bounded implementation milestone

Implement the following before a large training commitment:

1. A concise capability/evaluation specification, schema, split/sealing workflow, rubrics and scoring tests, initially with 10–20 reviewed development examples. Complete the agreed development set before recipe selection and the independent sealed set before final evaluation. The full 150-case sealed set need not delay a throughput benchmark. Keep final test answers out of the tuning workflow.
2. A runtime-probe script and compatibility report schema. Separate locally observed capabilities from unknown HPC capabilities.
3. A bounded corpus sampling manifest and tokenizer comparison tool with explicit limits.
4. One candidate model/trainer plus benchmark-only mode, resume support and focused CPU correctness tests.
5. Pretrained-baseline evaluation and small SFT/adapter entry points with native-tokenizer/template handling.
6. A CPU INT8 entry point and physically compact FFN-pruning fixture, with actual format/shape checks.
7. PBS templates and a resource estimate worksheet with unset allocation fields clearly marked.

Suggested new locations: stage_7_general_lm and stage_8_compression. Follow current repository conventions; do not create redundant frameworks. Reuse tested ideas/helpers where appropriate without forcing new models through frozen Stage 6 initialization or identity rules.

Meaningful tests should exercise the real production classes: causality, padding/label masking, token-ID bounds, gradient accumulation, fresh-process restart with dropout/scaler when used, cached inference, source-group isolation and prune/quant export reload. Do not substitute dummy architectures to pass compatibility tests. Temporary tests must not update preserved trained models.

Bulk downloads, GPU jobs and long runs require concrete resource limits. Prepare the commands now; execute only within the user's authorized workload and available allocation. No new GPU-hour limit is invented by this document.

The milestone report must state files added/changed, checks actually run, their results, observed versus unknown environment details, data sizes, model counts, exact next commands and the proposed measured training budget. Do not mark pending cluster checks as passed.

## 15. Deliverables and claims at completion

Produce:

- A reproducible from-scratch experiment or an explicit resource-limited pilot report.
- An unmodified pretrained assistant reference and any validated adaptation.
- Actual pruning and quantization artifacts, including a combined selected variant.
- Paired quality results and size/memory/latency tables on each tested device.
- Pinned code/config/model/tokenizer/data identities, environment records and exact execution commands.
- Final exports and a clear distinction between training checkpoints, deployment artifacts and external pretrained dependencies.

Report base-model competence, post-training gains, compression degradation and systems speedups separately. A combined design can be useful without establishing which individual change caused its gains.

Only then prepare the own-1B training proposal: measured data/learning curves, source availability, tokenizer, GPU-hours, peak memory, training/distribution strategy, capability targets and compression reserve. It is a later allocation decision, not an automatic next launch.

## 16. Message to give the working agent

> Proceed using this methodology. The first capability is the source-grounded English study assistant. Implement the first bounded milestone in Section 14 and prepare reproducible runtime, benchmark and training commands. Keep Stage 6 unchanged. Start the pruning and quantization fixture alongside the new training pipeline. Return actual CPU checks, observed/unknown compatibility, bounded sample/tokenizer results and a costed next-run proposal. Do not launch the billion-token run or consume an unconfirmed GPU allocation; the main horizon and resource caps must be fixed from measurements first. Keep both pruning and quantization on the November 29 delivery path.

