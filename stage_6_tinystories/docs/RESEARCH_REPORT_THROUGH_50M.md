# NanoGPT Stage 6 TinyStories — complete project and research report

> Historical scope: this report captures the project through +50M. For the
> completed +100M outcome, use [RESULTS.md](RESULTS.md). Artifact paths mentioned
> in the historical narrative may describe their locations at the time; the
> current organized layout is indexed by [the Stage 6 README](../README.md).

Prepared: **1 November 2026**, Asia/Kolkata. This is the state supported by the available artifacts and the HPC terminal evidence supplied in this conversation.

## 1. Where Stage 6 stands today

Stage 6A is a controlled continuation of the approximately 6.9M-parameter TinyStories model trained in Stage 5. It has reached **50,003,968 additional training positions**, halfway through its configured whole-update continuation budget. The model architecture, tokenizer, dataset, and main training configuration remain fixed.

Three different conclusions must be kept distinct:

| Question | Current answer |
|---|---|
| Is the implementation and deployment workflow supported by validation? | Yes. Historical CPU/packaging checks and reported GPU checks support it; later real training and evaluation jobs completed successfully. |
| Is the +50M checkpoint a legitimate continuation of this run? | **Yes, within the documented provenance-audit scope.** Hashes, metadata, source/config/data identities, continuation records, optimizer state, LR progression, and PBS completion agree. |
| Has additional training improved language modeling? | **Yes on the fixed evaluation windows.** Development CE and perplexity improved materially from +25M to +50M. |
| Has it established a broad improvement in storytelling? | **No.** Some paired stories improve, but contradictions, identity drift, prompt loss, loops, and weak endings persist. Human ratings are blank. |
| Has +75M or +100M training been completed? | No such completion is established by the available evidence. +75M is the next proposed review gate; approximately +100M is the configured overall budget. |
| Are the Mac and HPC run directories at the same training point? | **No.** The Mac's run checkpoints/logs stop at +25M. The verified +50M checkpoint is on HPC. |

The correct project status is: **training and provenance verified through +50M; language-modeling improvement measured; storytelling benefit still under review.**

This report consolidates the project history and the work completed together. It adds documentation only. It does not change the training plan, run an experiment, transfer a checkpoint, or edit an existing artifact.

## 2. What Stage 6 is trying to answer

The central experiment asks:

> If the same small GPT receives more TinyStories pretraining under a controlled continuation schedule, do its predictions and generated stories improve enough to justify the extra training?

Stage 6 is **continued next-token pretraining**. It is not supervised instruction fine-tuning, preference optimization, or RLHF. There is no demonstrated architecture expansion or new tokenizer in this run.

The main experiment is named `stage6a_context256_continue`. Its run directory is `stage_6_tinystories/runs/s6a`.

The design holds architecture, vocabulary, corpus, batch/context, and evaluation protocol fixed while increasing training progress. This makes the +25M/+50M comparison interpretable. It is still one training trajectory, not a multi-seed training study or an architecture ablation.

A separate optional configuration, `configs/control_lr.json`, defines `stage6a_context256_constant_lr_control`: the same architecture/data and an approximately 100M-position budget at constant LR `3e-5`, with no warmup. Its existence provides an experimental option; the artifacts inspected here do **not** establish a completed control run. It must not be presented as measured evidence.

## 3. Where the project lives and what is authoritative

### Canonical Mac repository

```text
/Users/shash/Library/CloudStorage/GoogleDrive-shash9271@gmail.com/My Drive/Currently Doing/NanoGPT
```

This is the canonical workspace for the source, local artifacts, and reports in this conversation. Folder visibility in Google Drive was not accepted as evidence of complete contents: the initial audit read files and verified hashes and declared transfer paths.

### HPC project

```text
/home/cse/btech/cs1230342/zenithX9271/NanoGPT
```

The user connects with `ssh cs1230342@hpc.iitd.ac.in` using interactive password authentication. The assistant never requested or received the password.

The actual +50M training state is in the HPC `stage_6_tinystories/runs/s6a` directory. A filename such as `latest.pt` is only meaningful together with its machine, path, hash, and metadata. The Mac and HPC files with that relative name currently contain different training stages.

### Evidence hierarchy used in the audit

1. Actual checkpoint hashes and deserialized metadata, plus supplied scheduler history and run logs.
2. Evaluation JSONs, generated IDs/text, appended metrics, source hashes, and configurations.
3. Historical verification/packaging reports, with their original scope and dates retained.
4. Names, directory labels, and old instructions, which can be suggestive but are not sufficient proof.

The supplied HPC output is user-provided evidence, not a separate direct retrieval by the assistant. Its consistency was checked locally and its attachment was hashed. That limitation is explicitly retained in the provenance conclusion.

## 4. The Stage 5 starting point

Stage 6 initializes from the genuine Stage 5 best checkpoint:

| Property | Verified reference |
|---|---|
| Original file | `stage_5_tinystories/checkpoints/best.pt` |
| Dedicated Stage 6 baseline copy | `stage_6_tinystories/baseline/stage5_best.pt` |
| SHA256 | `2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce` |
| Size | 95,568,257 bytes |
| Completed Stage 5 updates | 12,000 |
| Processed Stage 5 positions | 98,304,000 |
| Historical saved validation CE | Approximately 2.136093 |
| Learned parameters | 6,897,152 |

The baseline weights **and existing AdamW moments** are loaded. Stage 6 then defines a **new explicit RNG trajectory and LR schedule**. It does not claim to reconstruct Stage 5's missing historical RNG state. Exact restarts within Stage 6 are a separate capability, supported by Stage 6 checkpoints storing the necessary state.

### The misleading pilot checkpoint

`stage_6_tinystories/baseline/best.pt` is a preserved, different pilot artifact:

- SHA256: `fa8b534b40c372ab4502b831d841c36fde8fcb70e4b5bb6210e5944eff1d0197`.
- Historical report: step 1220, 9,994,240 processed positions, CE approximately 3.353233.

It is not the selected Stage 5 baseline. The missing-baseline and wrong-pilot problems were addressed in the earlier packaging workflow by establishing the pinned `stage5_best.pt` copy, while preserving the pilot. Renaming a convenient file to the expected name would not have established its identity.

### Which baseline score is comparable?

The best numerical reference for the current experiment is the **zero-Stage-6-update evaluation** recorded by the Stage 6 evaluator:

- Evaluated train CE: **2.100881279**.
- Development CE: **2.151154316**.
- Development perplexity: **8.594773754**.

The checkpoint's historical saved CE near 2.136093 and the legacy Stage 5 generation-output header CE 2.5202 were produced under other recorded evaluation circumstances. They must not replace the fixed-protocol baseline in the Stage 6 comparison.

No complete, same-protocol 159-story Stage 5 generation baseline was found. Numerical comparison to the initialization is possible; an equivalent complete storytelling comparison against Stage 5 is not yet available.

## 5. Architecture: what this model actually is

| Component | Stage 6 value |
|---|---|
| Model type | Decoder-only causal Transformer |
| Unique learned parameters | **6,897,152** |
| Transformer blocks | 6 |
| Embedding width | 256 |
| Attention heads | **8**, each width 32 |
| Context length | 256 tokens |
| Vocabulary | 8192 byte-level BPE tokens |
| Feed-forward width / activation | 1024 / ReLU |
| Normalization | Pre-LayerNorm residual blocks; final LayerNorm |
| Positions | Learned positional embeddings |
| Embedding/output sharing | Tied token embedding and output-projection weights |
| Dropout | 0.2 |
| Training precision | FP32 |
| Attention implementation | Manual scaled dot-product attention with a causal mask |

The earlier task attachment described four heads. The actual Stage 5/Stage 6 configuration and the +50M checkpoint specify **eight**. The report uses the checkpoint-backed value.

The implementation preserves Stage 5 state-dictionary names so the existing checkpoint can load strictly. It does not rely on AMP, SDPA/Flash Attention, or compilation for this legacy HPC stack.

An approximate parameter breakdown, following the verified architecture, is:

| Part | Learned parameters |
|---|---:|
| Token embeddings, also used by output head | 2,097,152 |
| Positional embeddings | 65,536 |
| Six Transformer blocks combined | 4,733,952 |
| Final LayerNorm | 512 |
| Total | **6,897,152** |

The approximately 95.6 MB checkpoint is larger than the bare FP32 learned weights, which occupy approximately 27.6 MB. Checkpoints also store two AdamW moment tensors per parameter, causal-mask buffers, RNG state, and metadata.

## 6. Dataset and tokenizer

Stage 6 reuses the genuine prepared TinyStories v2 pipeline artifacts inherited from the earlier project. The local preparation provenance identifies `roneneldan/TinyStories`; no dataset download, replacement corpus, or tokenizer retraining was performed during these audits.

The project's `v2` directory label refers to its revised data pipeline. It is not evidence that a different external TinyStories corpus version was introduced for Stage 6.

| Split | Stories | Stored token positions | Token shards |
|---|---:|---:|---:|
| Train | 1,792,647 | 386,680,799 | 39 |
| Development | 10,995 | 2,316,856 | 1 |
| Test | 10,995 | 2,315,362 | 1 |
| Total shard files | — | — | **41** |

The stored preparation provenance records 2,119,489 raw training stories, removal of 320,241 within-training duplicates and 6601 cross-split duplicates, leaving 1,792,647 training stories. The 21,990 validation-source stories were divided equally into development and test. Earlier full packaging validation reports zero normalized train/development/test overlap; that is historical validation evidence, not a newly repeated deduplication in this report.

Tokenization/storage:

- Vocabulary size 8192; PAD=0, BOS=1, EOS=2.
- Each story is encoded as `[BOS] story_tokens [EOS]`.
- Token shards use little-endian uint16 storage.
- Stories are concatenated. Causal attention is **not** separately blocked at story boundaries; BOS/EOS provide learned markers.
- Document offsets are preserved for split/story traceability and prompt selection.

The training loader reads root `data/shards/v2`, not `stage_6_tinystories/data`. It memory-maps shards and materializes only batch windows. A window can cross shard boundaries.

The sampler draws random window starts with replacement. Therefore, “processed positions” measures training exposure; it is not a count of unique dataset positions seen, unique stories learned, or a completed full-corpus epoch.

Core identities shared by local references, the verified HPC checkpoint, and evaluation records:

| Asset | SHA256 |
|---|---|
| Shard manifest | `89c9df839572ef6a1890caf4b1144229090b8ced8b4ea2c66ed5dd090e17c9a7` |
| Tokenizer | `99055f35c77392e4df4486685b60fee1de1d95e2c373c26f487cf9944746d038` |
| Evaluation prompts | `f1c83474faa2b56d670055d6da373a1460eccc007db296bcf509f44f5737429c` |

## 7. Training configuration and checkpoint semantics

### Main hyperparameters

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 32 |
| Context | 256 |
| Positions per optimizer update | **8192** |
| Betas | 0.9, 0.95 |
| Weight decay | 0.1 for matrix parameters; 0.0 for the other group |
| Gradient clipping | Global norm 1.0 |
| Seed | 1337 |
| Configured additional-position budget | 100,000,000 nominal |
| LR start / peak / minimum | 0.00003 / 0.0001 / 0.00003 |
| Warmup | 200 Stage 6 updates |
| After warmup | Cosine decay across the original full continuation horizon |
| Training log interval | 50 updates |
| Periodic evaluation / latest-save interval | 500 updates |
| Configured snapshot thresholds | +10M, +25M, +50M, +100M |
| Walltime safety margin | 300 seconds |

The parsed `continue.json` matches the actual +50M checkpoint and all three training invocations. Its SHA256 is `0df15f4064a433c9f6db1678eeefdcdb0edf0de0385c408f5dd3cf1e8223f7b7`.

### Why the milestone counts are slightly above round millions

Updates are indivisible. The code uses a ceiling when converting a token budget into optimizer updates:

```text
additional positions = Stage 6 updates × 8192
total updates        = 12,000 + Stage 6 updates
total positions      = 98,304,000 + additional positions
```

| Gate | Stage 6 updates | Actual additional positions | Total updates | Total positions | Evidence/status |
|---|---:|---:|---:|---:|---|
| +10M | 1221 | 10,002,432 | 13,221 | 108,306,432 | Completed |
| +25M | 3052 | 25,001,984 | 15,052 | 123,305,984 | Completed |
| +50M | 6104 | 50,003,968 | 18,104 | 148,307,968 | Completed and provenance verified |
| +75M | 9156 | 75,005,952 | 21,156 | 173,309,952 | Proposed next gate, not completed in supplied evidence |
| +100M | 12,208 | 100,007,936 | 24,208 | 198,311,936 | Configured full target, not completed |

The +75M gate is a review milestone described by the research task, **not an existing entry in the current snapshot list**. These numbers explain the plan; they do not authorize a job or change any configuration.

Stopping at a review gate does not shorten the original LR schedule. The verified +50M run retained the approximately 100M-position horizon rather than restarting warmup or compressing the schedule to 50M.

### What each checkpoint means

- **`latest.pt`:** current training weights plus optimizer/RNG/sampler state, used for same-run resumption. The trainer does not reload best before saving it.
- **`best.pt`:** lowest fixed-development-CE checkpoint observed so far. It can be earlier than latest.
- **Threshold snapshot:** a preserved training state when additional positions first meet a configured threshold; the actual counter is stored inside.
- **`last_eval`:** explicitly records which update was evaluated. A filename alone cannot establish score/update alignment.

Checkpoints include the model, optimizer, configuration, counters, origin, source/data identities, Python/NumPy/Torch CPU/Torch CUDA RNG, the independent training sampler, evaluation records, saved thresholds, and runtime information.

Exact Stage 6 resumption checks configuration, data, and all six root Python source identities. It loads optimizer moments, validates counters/hyperparameters, restores RNG and sampler state, and continues from the cumulative update count. The training CLI requires same-run `latest.pt` in its existing run directory for a resume.

Checkpoint saving uses a temporary file, flush/fsync, and atomic replacement. Walltime and termination handling try to preserve a resumable state; a storage failure or forced kill still cannot guarantee a final save. Pending final evaluation/snapshot work can be finished on restart without an extra optimizer update.

## 8. What was built and validated before the quality runs

The repository separates responsibilities:

| File/component | Purpose |
|---|---|
| `model.py` | Parameterized Stage 5-compatible architecture and generation |
| `data.py` | Memory-mapped shards, cross-shard windows, tokenizer decoder, asset validation |
| `core.py` | Engine, initialization/resume, optimizer/RNG state, LR, fixed evaluations, checkpoint checks |
| `train.py` | CLI, training loop, logging, deadlines, safe directories, preflight and benchmark modes |
| `evaluate.py` | Fixed CE evaluation, seeded generations, JSONL/CSV outputs, diagnostics |
| `hpc/train.pbs` | GPU allocation/runtime checks, transfer verification, controlled training/evaluation wrapper |
| `tests/test_stage6.py` | Restart/state/data/model/safety behavior |
| `tests/check_restart.py` | Separate-process uninterrupted-versus-resumed comparison |
| `tools/prepare_transfer.py` | Declared transfer payload/inventory/checksum preparation |
| `tools/verify_transfer.py` | Payload, baseline, path, and optional deep-data checks |
| `tests/test_transfer.py` | Missing/corrupt files, symlinks, relocation, and PBS/packaging contracts |

### Historical local validation, 29–30 October

[VERIFICATION.md](VERIFICATION.md) and [HPC_VALIDATION_REPORT.md](HPC_VALIDATION.md) record:

- **19 Stage 6 tests passed**, with real Stage 5 compatibility available.
- **12 packaging tests passed** in the recorded packaging validation.
- Strict source-checkpoint/model/optimizer compatibility and the 6,897,152-parameter architecture.
- Dropout restart equivalence, including fresh-process tests and a bounded real-model CPU demonstration.
- Preservation of training RNG and mode around evaluation/generation.
- Wrong data/configuration/source rejection, nonfinite-state rejection, walltime/signal handling, and safe run directories.
- Shard corruption detection, cross-shard windows, and baseline hash enforcement.
- Python 3.7 grammar and PBS shell syntax checks.
- Real CPU forward/evaluation/generation smoke checks and a zero-update checkpoint roundtrip.

These are dated recorded results. They were **not rerun during the present read-only audits**; some tests intentionally perform temporary optimizer updates and would exceed the audit's scope.

### Packaging repair and self-contained data

The earlier investigation found that the dedicated baseline was missing, a different pilot was present under a tempting name, raw-source symlinks pointed outside the repository, and manual transfer procedures could omit dependencies or retain stale files.

The historical repair established the exact baseline copy, materialized genuine raw-source files while preserving originals, kept the existing tokenizer/corpus/shards, and introduced an explicit **104-path transfer list** with inventory and hashes. It verified relocation to a separate path containing a space, with access to the original repository blocked during the relocation checks.

The recorded deep checks include 41 shard hashes/lengths, token ranges, story offsets and BOS/EOS counts, split isolation, and exact tokenizer reencoding for 24 sampled genuine stories. No synthetic replacement corpus was used.

### Historical user-reported V100 checks

| Job | Recorded historical outcome | What it establishes |
|---|---|---|
| 1125948 | Required baseline missing | Deployment failure to resolve before training |
| 1125950 | Wrong pilot SHA256 rejected | Identity guard prevented selecting the wrong artifact |
| 1125951 | Preflight passed | Reported V100 model/optimizer/data identity compatibility |
| 1125955 | Benchmark passed; approximately 70,375 positions/s | Historical performance estimate; benchmark updates discarded |
| 1125957 | Four uninterrupted versus two + restart + two matched | Reported GPU restart validation, not quality-training progress |

The benchmark report also gives approximately 6.23 seconds per evaluation, 0.418 seconds per save, and 2983 MiB peak allocated GPU memory. These values are historical user-reported measurements, not new benchmarks by the assistant. They are not a throughput guarantee for every subsequent job.

## 9. Actual training history: from initialization to +50M

### 30 October: first quality-training interval, 0 → +10M

The first invocation initialized from the pinned Stage 5 baseline and targeted 1221 Stage 6 updates. It began at **16:54:06 IST** and recorded a `review_gate` exit at **16:59:42 IST**.

The development CE temporarily worsened from the initialization score 2.151154 to 2.167323 at update 500, then improved to 2.159101 at the +10M gate. The +10M gate was still worse than the fixed initialization score. This is why a milestone's existence cannot be treated as proof of improvement.

### 31 October: resumed +10M → +25M

The second invocation records `starting_stage6_updates=1221`, uses `--resume .../latest.pt`, and targets update 3052. It began at **15:20:19 IST** and exited at **15:31:48 IST**, again for `review_gate`.

Development CE fell below the fixed baseline by update 2000 (+16.384M positions) and reached **2.133726095** at +25M. Both complete +25M generation folders reference the corresponding best checkpoint.

### 1 November: resumed +25M → +50M

The third invocation records:

- Resume source: existing `stage_6_tinystories/runs/s6a/latest.pt`.
- Initial Stage 6 counter: **3052**.
- Stop counter: **6104**.
- `init_from=null` and unchanged source/config/data identities.
- Invocation timestamp: **15:24:13 IST**.

The final accounting is **6104 Stage 6 updates**, **18,104 total updates**, and **50,003,968 additional positions**. The trainer exits for **`review_gate`**, not an observed crash or walltime termination.

### Full recorded development trajectory

| Stage 6 updates | Additional positions | Development CE | Development perplexity |
|---:|---:|---:|---:|
| 0 | 0 | 2.151154 | 8.594774 |
| 500 | 4,096,000 | 2.167323 | 8.734866 |
| 1000 | 8,192,000 | 2.162191 | 8.690153 |
| 1221 | 10,002,432 | 2.159101 | 8.663345 |
| 1500 | 12,288,000 | 2.154897 | 8.627005 |
| 2000 | 16,384,000 | 2.145209 | 8.543830 |
| 2500 | 20,480,000 | 2.139795 | 8.497699 |
| 3000 | 24,576,000 | 2.134342 | 8.451486 |
| 3052 | 25,001,984 | 2.133726 | 8.446280 |
| 3500 | 28,672,000 | 2.127759 | 8.396031 |
| 4000 | 32,768,000 | 2.121728 | 8.345546 |
| 4500 | 36,864,000 | 2.113605 | 8.278029 |
| 5000 | 40,960,000 | 2.110323 | 8.250903 |
| 5500 | 45,056,000 | 2.102635 | 8.187719 |
| 6000 | 49,152,000 | **2.094860** | **8.124305** |
| 6104 | 50,003,968 | 2.094938 | 8.124939 |

This is one trajectory with a real initial setback followed by sustained development-score improvement, then a very small final reversal. The observations do not support a claim of monotonic improvement from the first update, nor do they establish a clear CE plateau at +50M.

## 10. Evaluation: how the comparison was controlled

### Fixed next-token prediction scores

| Setting | Value |
|---|---|
| Evaluation seed | 2026 |
| Context / batch | 256 / 16 |
| Development windows | 4096 |
| Development scored positions | 1,048,576 |
| Training windows | 512 |
| Training scored positions | 131,072 |

The evaluator resets its own sampling generator for each split and uses model evaluation mode. Thus the reported train CE is a fixed training-split evaluation score; it is not the noisy dropout-enabled loss printed after individual optimizer updates.

The windows are sampled from concatenated token data and can overlap. Scored positions are not independent samples. Saved aggregate CE values do not provide a sound per-window uncertainty estimate by themselves.

### Story generation

| Setting | Value |
|---|---|
| Development prefixes | 50, each 32 tokens including BOS |
| Manual probes | 3 |
| Generation seeds | 1337, 2027, 3407 |
| Temperature | 0.7 |
| Top-k | 50 |
| Maximum new tokens | 300 |
| Expected stories per checkpoint | **53 × 3 = 159** |

The manual prompts are:

1. “Once upon a time, a little fox found a blue box.”
2. “Lily wanted to help her friend, but she did not know how.”
3. “Tom put his toy under the bed. Later, he looked for it and”

Each sample has explicit prompt IDs and generated IDs, plus decoded text, seed, stopping reason, token count, and a repeated-4-gram diagnostic. Saved EOS is removed from continuation IDs/counts after termination. Generation uses its own seeded generator and temporarily switches to evaluation mode.

### Which artifacts were counted

| Directory | Meaning |
|---|---|
| `runs/s6a_eval25` | Primary complete +25M comparison set: CE, 159 stories, ratings template |
| `runs/s6a_best25_generation` | Duplicate +25M story/ratings bytes; different timing metadata, same checkpoint and outputs |
| `evaluations/s6a_1127757.pbshpc_u3052` | Earlier CE-complete, generation-incomplete attempt; empty story file and header-only CSV |
| `evaluations/50m` | Complete +50M latest-checkpoint evaluation and 159 stories |

The two complete +25M directories are **not independent experiments**. Counting them twice would inflate the evidence. The empty older story file is consistent with an incomplete generation attempt; it does not mean the complete +25M generation outputs were lost.

We verified all **159 unique `(prompt_id, seed)` pairs** at each gate, identical prompt text and seeds, valid generated vocabulary IDs, exact decoding, token counts, CSV/JSONL agreement, and recomputed repetition diagnostics. All 318 continuations were inspected as matched pairs in the qualitative audit.

## 11. Quantitative results and their meaning

| Quantity | Fixed Stage 5 initialization | +25M | +50M latest |
|---|---:|---:|---:|
| Evaluated train CE | 2.100881279 | 2.083490644 | 2.044401698 |
| Development CE | 2.151154316 | 2.133726095 | 2.094938189 |
| Evaluated train perplexity | 8.173369755 | 8.032458485 | 7.724535549 |
| Development perplexity | 8.594773754 | 8.446279895 | 8.124938746 |
| Development CE minus train CE | 0.050273037 | 0.050235452 | 0.050536491 |

From +25M to +50M:

- Development CE improves by **0.038787906 nats/token**, or **1.817848%**.
- Development perplexity improves by **0.321341149**, or **3.804529%**.
- Evaluated train CE improves by **0.039088946**.
- The train–development gap increases by only **0.000301040 nats/token**.

Relative to the fixed Stage 5 initialization, +50M improves development CE by **2.613300%** and perplexity by **5.466520%**.

CE measures how well the model predicts held-out next tokens. Perplexity is `exp(CE)`, so these two improvements are different expressions of the same prediction evidence, not independent tests of quality.

The nearly unchanged train–development gap provides no obvious aggregate overfitting signal on these windows. It cannot rule out memorization or deterioration in particular story categories. The intermediate trajectory supports continued improvement through update 6000; its small final reversal does not by itself diagnose overfitting.

### Why best and latest differ at +50M

The actual best checkpoint is update **6000**, +49,152,000 positions:

- Best development CE: **2.094860245**.
- Latest development CE at update 6104: **2.094938189**.
- Difference: approximately **0.000077944 nats/token**, or 0.00372% of best CE.

The 159-story +50M evaluation used latest. Its storytelling findings cannot be reassigned to best simply because their CE values are close. The tiny CE difference supplies no evidence, on its own, that changing the continuation state or rerolling generations is necessary.

## 12. What the generated stories actually show

The qualitative review is model-assisted and unblinded. It is not a human evaluation and does not produce an invented quantitative human-preference result.

| Dimension | Current evidence |
|---|---|
| Grammar/readability | Both checkpoints often produce readable local sentences. Improvements and regressions coexist. |
| Logical progression | Some +50M episodes are simpler and cleaner; many still contain unsupported transitions. |
| Character identity | Name, pronoun, species, and role changes persist. No reliable overall fix is established. |
| Object/setting consistency | Objects change function or identity and locations drift without explanation. |
| Causality | Problems are frequently resolved by an unrelated arrival, hug, or moral rather than a connected action. |
| Prompt retention | Some topic retention improves; explicit facts can still be lost immediately. |
| Narrative development | Extra length does not reliably create a stronger problem–attempt–resolution arc. |
| Endings | EOS is common, but natural and logically earned endings are not assured. More +50M stories reach the cap. |
| Repetition/nonsense | Mean literal repetition improves slightly, but a worse extreme loop appears. Semantic nonsense can have low literal repetition. |
| TinyStories surface style | Short sentences, simple vocabulary, children/animals, and moral endings remain recognizable. Style is not sufficient evidence of coherence. |

### Representative matched examples

**A relative improvement — `dev_09383`, seed 3407.** Rose explores a garden. The +25M version introduces a confused dog episode and mismatched references. The +50M version follows a simpler butterfly/garden sequence and closes with her enjoyment of the garden. It is cleaner, although the butterfly pursuit is still not fully resolved.

**Better topic retention, still shallow — `dev_10452`, seed 2027.** Tom and Mia start with crayons. +25M repeatedly speculates about treasure. +50M returns to their crayons and ends more directly. Repeated 4-grams decline from approximately 0.1783 to 0.0421, but grammar and unexplained stickers remain weak.

**A serious regression — `dev_01952`, seed 2027.** The prompt concerns a mother and a tough novel. At +50M, “She read the book” repeats, then the story shifts into “The frog was not a frog. It was a frog.” The repetition diagnostic rises from approximately 0.0962 to 0.3318. It still ends with EOS.

**Identity failure — `dev_00405`, seed 1337.** A simple +25M Lucy-and-dog friendship story becomes a +50M continuation in which Lucy says “I am a dog” and is proud of being a good dog, without a transformation setup.

**Prompt-fact failure at both gates — `manual_3`, seed 1337.** Tom's toy was put under the bed. Both checkpoints begin by finding it “under a bush,” without explaining movement. The three seeds at each gate do not establish a coherent under-bed retrieval. The initial error occurs before any context-window truncation.

**More events, worse tracking — `manual_1`, seed 3407.** The +25M fox story has self-dialogue. The +50M story switches among fox, rabbit, and bear, with inconsistent box opening/locking. Additional plot events do not make the story more stable.

**Persistent multi-character weakness — `dev_06045`, seed 3407.** The prompt assigns Tom as seeker and Lily as hider. +25M eventually gives Tom a tail. +50M introduces repeated chairs, changes Tom to Max, confuses dog/cat references, and reaches the cap. All three +50M seeds for this prompt hit 300 tokens.

The model's 256-token context eventually drops the earliest prompt tokens during sufficiently long continuations. This is a plausible limitation for long-range retention, but it cannot explain first-token or immediate prompt-fact failures. Small capacity, context, objective, data, and decoding remain hypotheses to separate through future experiments, not causes identified by this audit.

## 13. Generation diagnostics and human-rating status

| Diagnostic | +25M | +50M latest |
|---|---:|---:|
| Complete stories | 159 | 159 |
| EOS terminations | 154/159, 96.8553% | 151/159, 94.9686% |
| Maximum-token terminations | 5/159, 3.1447% | 8/159, 5.0314% |
| Mean continuation tokens | 148.9308 | 155.1069 |
| Median continuation tokens | 148 | 147 |
| Mean repeated-4-gram fraction | 0.0343226 | 0.0323556 |
| Stories with repetition >0.10 | 8 | 5 |
| Stories with repetition >0.20 | 0 | 1 |
| Worst repetition fraction | 0.178261 | 0.331754 |

Interpretation:

- The small reduction in average literal repetition is positive, but it does not capture causal or identity errors.
- The worse extreme loop shows why averages alone are insufficient.
- Increased mean length combined with slightly lower median length indicates a longer tail, not a general increase in useful development.
- EOS proves that the model sampled a terminator, not that the preceding problem was resolved.
- Four paired stories switch EOS→cap, one switches cap→EOS, four remain capped, and 150 remain EOS-terminated.
- The net increase in cap counts is concentrated in seed 3407; the small mean repetition improvement is not consistent across all three seeds.

The earlier comparison used a paired bootstrap over 53 prompt clusters, preserving each prompt's three seeds: 20,000 resamples with analysis seed 20261101. Exploratory 95% percentile intervals for +50M minus +25M were:

| Diagnostic difference | Estimate | Exploratory interval |
|---|---:|---:|
| Mean repetition | −0.001967 | [−0.008787, +0.005019] |
| Mean length | +6.1761 tokens | [−2.2140, +14.4717] |
| Cap fraction | +0.018868 | [−0.006289, +0.050314] |

All include zero. These are sensitivity checks for this selected prompt set, not evidence about variance across independently trained models. No valid CE significance interval or quantitative human-quality superiority was manufactured from aggregate summaries.

### Human ratings

Both primary `ratings.csv` files contain 159 rows and the same four 0–4 fields:

- Grammar.
- Character consistency.
- Prompt-fact retention.
- Ending quality.

**All rating fields are blank at both gates.** Blank means not rated, not a score of zero. The duplicate +25M CSV adds no observations.

This is the principal remaining quality-evidence gap. A practical next review would hide checkpoint names, randomize left/right placement of each matched pair, show the prompt, allow ties, and use two independent human raters. All 159 pairs are preferable; a preregistered 20-prompt × 3-seed pilot gives 60 pairs if a smaller first step is needed. Severe-loop/cap cases should also be examined separately, without counting that targeted set as a representative preference sample.

## 14. What we did together in this conversation

This section distinguishes work in the current conversation from historical implementation and user-run experiments.

### Step 1: audited the Google Drive workspace

You identified Google Drive as the canonical repository and asked for a read-only comparison with an earlier handoff. The message contained a handoff placeholder rather than the complete handoff, so a literal handoff comparison was unavailable. We reconstructed the situation from the repository's records instead of assuming missing content.

We inventoried source, checkpoints, data, metrics, and evaluation folders; checked all declared transfer paths and the then-current checksum ledger; identified the complete and incomplete evaluation sets; and inspected checkpoint metadata on CPU without running training or generation.

The initial inventory established substantial local completeness, but it did not establish that every future remote run artifact had been copied back.

### Step 2: isolated the `model.py` mismatch

At the first audit, current `model.py` hashed to:

```text
b058a3621e93103100e017742bc15a08cc9a586f750b59df5b93fdc1a42417d3
```

The checkpoints recorded:

```text
7aca47adbae88c80bdc9d0339dc662761d179f51e5e274715dc9ae00e954cbb5
```

The other five source identities matched. Thus the problem was one exact source-file identity, not a general failure of the entire codebase or damaged checkpoint weights.

The transfer ledger expected the patched local version at that point. A ledger matching a packaged source copy does not automatically establish that it matches a checkpoint's training source.

### Step 3: followed your exact-copy recovery lead

You supplied the path:

```text
/tmp/nanogpt-hpc-ready-dfYDEx/stage_6_tinystories/model.py
```

We checked it directly. It existed as a regular 5335-byte file and exactly matched the original checkpoint-recorded `7aca…cbb5` SHA256. The surrounding temporary source tree matched all six checkpoint-recorded Python identities.

The exact source difference was:

```python
# Original checkpoint-compatible generation code
logits[:, [pad_id, bos_id]] = float('-inf')

# Separate generation patch
logits[:, pad_id] = float('-inf')
logits[:, bos_id] = float('-inf')
```

We recommended permanently preserving the exact original before relying on a volatile temporary directory. No recovery copy or restoration was performed during that strictly read-only inspection.

### Step 4: observed the preserved archives and restored active identity

Later workspace inspection found both versions under `recovery_archive`, and active `model.py` now matches the original `7aca…cbb5` identity. The original archive and generation patch both remain available.

That establishes the recovered state. The exact archive/restoration command and actor are not captured in this conversation's audit actions; we should not retroactively claim the read-only assistant performed the restore. The archive manifest describes the patched file as active at archive-creation time, while current bytes show the original is now active.

### Step 5: compared +25M against the newly available +50M evaluation

You supplied a detailed research-review task. We found the +50M evaluation and:

- Matched every prompt/seed pair and checked protocol/data identities.
- Identified byte-identical +25M duplicate outputs and excluded duplicate counting.
- Verified exact decoding, CSV correspondence, token counts, and repetition diagnostics.
- Inspected the 318 generated continuations as matched pairs.
- Calculated CE/perplexity changes, generalization gaps, and diagnostic sensitivity intervals.
- Confirmed that human rating fields were blank.
- Corrected the four-head description to the actual eight-head architecture.

The resulting [comparison report](../evaluations/COMPARISON_THROUGH_75M.md) concluded that prediction metrics improved while broad storytelling improvement was not established. At that time the actual +50M checkpoint and continuation logs were not locally available, so provenance remained conditional.

### Step 6: attempted the approved read-only HPC connection

The assistant found no user SSH config and used the documented host. The first probe encountered sandbox DNS failure. The identical approved read-only probe outside that sandbox reached HPC but failed authentication.

That probe used `BatchMode=yes`, so it could not use the interactive password method you later explained was the working route. The failure was not evidence that your account or HPC run was invalid.

We created [the first HPC provenance report](../evaluations/PROVENANCE_50M.md) with a **PARTIAL** result rather than upgrading unverified remote claims to PASS.

### Step 7: checked access to your authenticated Terminal session

You authenticated through your Mac terminal and asked whether we could reuse it. The computer interface showed Terminal running, but blocked access to the Terminal app. We did not try to bypass that restriction or collect your password.

We switched to commands for you to run in your already authenticated session. You first supplied the latest hash, then checkpoint sizes/timestamps, then all five hashes. When you asked for fewer manual round trips, we prepared one read-only batch.

### Step 8: reviewed your complete HPC audit output

You ran the batch and supplied its terminal output. It contained actual checkpoint metadata, optimizer/RNG summaries, source/config/data hashes, appended invocation/metric records, job logs, and successful PBS-history queries.

We independently parsed those records, checked all counters and optimizer steps, compared identities against local references, and recalculated all 62 logged continuation LRs. We confirmed:

- Resume at 3052, finish at 6104.
- Exact expected +50M checkpoint hash.
- Normal review-gate exit.
- The distinct update-6000 best checkpoint.
- Temporary evaluation patch versus original training source.
- Successful training and evaluation scheduler exits.

The [follow-up provenance report](../evaluations/PROVENANCE_50M_FOLLOWUP.md) records a **PASS within its explicit evidence scope**. The earlier PARTIAL report was preserved as history.

### Step 9: created this consolidated report

This report brings the implementation, historical validation, completed training, evaluation findings, incident resolution, and current operational state together. Existing reports, source, data, checkpoints, and provenance remain separate preserved artifacts.

The assistant's audit work did not submit/cancel jobs, run training/generation, modify optimizer/RNG states, bypass resume guards, or transfer large checkpoints. The actual HPC experiments and audit batch were user-run; the assistant inspected local files and the supplied evidence and wrote the new reports.

## 15. What the final +50M provenance audit established

### Verified checkpoint identities

| HPC checkpoint | Stage 6 update | SHA256 |
|---|---:|---|
| 10M snapshot | 1221 | `430216811c4d1f6c258e609d961bbc5015e1336ea4a10c92bb383800e3639abd` |
| 25M snapshot | 3052 | `50f94bf80a142e58fdd0ef3daa933fe4c015c2353a7156d8be288f95cf8f50a6` |
| `best.pt` | 6000 | `e78edb3b96ef956aae64c269da452bb7af39a9e9534f0b4d9cf307d20ab7c573` |
| `latest.pt` and 50M snapshot | 6104 | `40d145f6739896f923438e54f5a3235a26bfdb0d5cc136bfafb816871658d04e` |

The exact inventory, sizes, timestamps, and log fingerprints are retained in the follow-up report.

### Optimizer/RNG/schedule evidence

- The +50M checkpoint has **208 optimizer states**, all recorded AdamW step counters at **18,104**, no missing states, and no reported nonfinite or mismatched moment pairs.
- The 25M snapshot's optimizer counters are **15,052**; the best-at-6000 counters are **18,000**.
- All expected RNG components are present. CUDA and training-sampler fingerprints change across gates; Python/NumPy/Torch CPU fingerprints remain unchanged, consistent with unused streams in GPU training.
- The matching source's resume path explicitly loads optimizer state and restores all RNG/sampler components before successful startup and invocation recording.
- Every one of **62 logged continuation LRs** agrees with the original schedule: maximum absolute recomputed error **0.0**.

These facts support continuation rather than a fresh optimizer/warmup start. They are code-and-record evidence, not an independent bit-for-bit replay of the complete +25M-to-+50M trajectory. No such training replay was performed in the audit.

### Training and evaluation source identities are different for a documented reason

The verified checkpoint and canonical HPC `model.py` use the original `7aca…cbb5` identity. Evaluation job 1129007 logs a separate temporary source:

```text
/var/tmp/pbs.1129007.pbshpc/nanogpt_eval50m.WWb56V/stage_6_tinystories/model.py
```

Its hash is the generation patch `b058…17d3`, also retained in the HPC recovery archive. This distinguishes a temporary evaluation implementation change from training-source modification. The patch does not provide evidence of an architecture or optimizer change.

### Scheduler evidence

| Job | Purpose | Node | Result |
|---|---|---|---|
| **1128982.pbshpc** | Resume +25M → +50M training | vsky034, V100 | `job_state=F`, **`Exit_status=0`**, 17m27s scheduler walltime; trainer reason `review_gate` |
| **1129007.pbshpc** | Full +50M evaluation | vsky010, V100 | `job_state=F`, **`Exit_status=0`**, 10m43s scheduler walltime; 159/159 generations complete |

The generic CUDA-module warning in the logs is followed by an actual allocated V100 and successful completion. It was not treated as proof of a failed job.

Canonical and checkpoint-recorded identities agree for `__init__.py`, `core.py`, `data.py`, `evaluate.py`, `model.py`, and `train.py`. The actual checkpoint confirms eight heads, six layers, and width 256.

## 16. Current files, outstanding gaps, and stale documentation

### Mac versus HPC

| Artifact | Mac Google Drive copy | Verified HPC state |
|---|---|---|
| `runs/s6a/latest.pt` | +25M, hash `50f94…f50a6` | +50M, hash `40d145…8d04e` |
| `runs/s6a/best.pt` | +25M, hash `d70946…42315` | +49.152M/update 6000, hash `e78edb…7c573` |
| 10M / 25M snapshots | Present | Present; hashes match local copies |
| 50M snapshot | Not found locally | Present; byte-identical to HPC latest |
| Run metrics/summary | Local history ends at +25M | Supplied output establishes history through +50M |
| Full +50M evaluation outputs | Present locally | Matching evaluation JSON and completed job verified |
| Active training `model.py` | Original identity | Original identity |

No checkpoint transfer was performed to erase this distinction. The local run directory is useful preserved history, not the current +50M resume state.

### Historical instructions that must not be read as today's state

| Historical statement | Current evidence |
|---|---|
| Stage 6 README/run instructions describe a +10M default gate and two-hour jobs | The current local PBS script requests **four hours** and defaults to a **3052-update/+25M** stop. The verified +50M invocation explicitly targeted 6104. |
| Old deployment instructions discuss a fresh-start default | Current PBS training branch requires an existing resume checkpoint and refuses a silent Stage 5 fallback. |
| Early reports say no quality-training latest is established | Historically accurate at their audit dates; superseded by later +25M/+50M artifacts and follow-up evidence. |
| Recovery manifest says the generation patch is active | True at its recorded archive-creation state; current active bytes are the original source. |
| Historical transfer ledger expects patched `model.py` | Current original source matches checkpoint training identity. The older ledger is not automatically a current full-workspace attestation. |

These records were preserved, not rewritten to make history appear uniform. They should be reconciled in a separately authorized documentation/deployment maintenance task before relying on old launch examples. This report contains no job-submission instruction and makes no PBS/configuration edit.

### Remaining research/provenance limits

1. No quantitative human story ratings have been entered.
2. No complete matched 159-story Stage 5 generation baseline was found.
3. One training trajectory and three generation seeds do not establish training-run reproducibility or architecture superiority.
4. The full +25M-to-+50M trajectory was not independently replayed bit for bit.
5. The HPC resume invocation records a mutable input path, not an immutable input SHA captured at the instant of resume. The preserved 25M snapshot, matching identities/counters, and execution records support the lineage.
6. The +50M audit verified manifest/tokenizer/prompt identities and recorded quick transfer checks; it did not rehash every remote shard or independently validate every final model tensor with a forward pass.
7. The temporary evaluation source hash is documented, but its complete temporary tree was not independently fingerprinted. The +25M generation-time source environment was not newly reconstructed by that +50M audit.
8. Repeated development review is not an untouched final-test evaluation. No new test-set score was run during these audits.
9. No completed +75M/+100M or constant-LR-control result is established.

None of these limits turns the documented +50M run into a failed run. They define which scientific claims remain unavailable.

## 17. What Stage 6 has taught us so far

### Supported by measured or directly recorded evidence

- The same small model can continue training from the genuine Stage 5 state and survive controlled same-run resumes with consistent accounting.
- The +50M result is traceable to the expected checkpoint, training source/configuration/data, and successful scheduler records.
- More pretraining improved fixed-window next-token prediction, including intermediate evaluations through the last major checkpoint.
- The train–development evaluation gap remains near 0.05 nats/token.
- Some stories become cleaner, but clear regressions and recurring semantic failures remain.
- Slightly lower average repetition can coexist with a worse extreme loop and more capped continuations.

### Interpretations, not established causal results

- Additional training has not obviously exhausted its language-modeling benefit, but a fixed improvement per next 25M tokens should not be assumed.
- Entity tracking and causal continuity appear more resistant than local fluency; this review does not identify why.
- The tiny best/latest score difference is consistent with small evaluation fluctuations and is insufficient alone to justify a schedule change.

### Hypotheses requiring future experiments

- Limited capacity or 256-token context may constrain long-story consistency.
- A different training objective, data strategy, schedule, or architecture might improve particular weaknesses.
- The optional constant-LR control could help separate schedule effects from additional exposure.

Those are possible future studies. No available comparison isolates their effects, and none was implemented during the audit.

## 18. Recommended next phase

### First: make the storytelling decision using existing outputs

Review blinded +25M/+50M pairs with anchored criteria for readability, coherence, entity/object consistency, prompt facts, causality, and ending quality. Record ties and severe failures; keep model-assisted observations distinct from human ratings. This can use the existing 159-pair outputs without generating another sample set.

### Second: decide whether the bounded +75M experiment is worth running

If human review finds no material regression, an unchanged continuation to update **9156** using the verified HPC +50M state is scientifically defensible as a bounded experiment. It should preserve the original LR horizon, optimizer moments, RNG/sampler state, architecture, tokenizer, and evaluation settings.

If humans confirm a material regression, inspect the affected cases and evaluation lineage before changing training. A fresh architecture, LR reset, checkpoint-metadata edit, or guard bypass would change the experiment rather than simply continue it.

### Third: predeclare what will count as useful progress

At +75M, examine:

- Development CE/perplexity against +50M and neighboring scheduled evaluation points.
- Evaluated train–development gap.
- Paired human preference and specific semantic-failure rates.
- Explicit prompt-fact retention, especially the toy-location and multi-character probes.
- EOS/cap behavior, repeated 4-grams, and the worst loop cases.
- Checkpoint/source/config/data lineage, optimizer counters, unchanged schedule, and PBS completion.

The earlier comparison proposed a practical target such as at least **0.01 nats/token** further development CE improvement and no material gap widening (illustratively, no increase beyond **0.005 nats/token**), alongside no material human-quality regression. These are **proposed decision thresholds**, not approved configuration changes, statistically derived guarantees, or expected results.

Retain the existing +25M/+50M artifacts regardless of the decision. Avoid spending more training merely because +75M appears next on the plan, and avoid stopping solely because one final score is marginally worse than the immediately preceding best.

## 19. Evidence map and final status

| Source | Main use |
|---|---|
| [Stage 6 implementation README](../README.md) | Original design and component semantics, with historical deployment details |
| [Stage 6 verification record](VERIFICATION.md) | Historical CPU tests and user-reported V100 validation |
| [HPC validation report](HPC_VALIDATION.md) | Packaging/deep-data/relocation checks and their scope |
| [Missing-file report](DATA_RECOVERY.md) | Baseline/pilot/raw-source/transfer repair history |
| [Source recovery manifest](../recovery/README.md) | Preserved original and generation patch identities |
| [Main configuration](../configs/continue.json) | Current controlled experiment settings |
| [Local metrics](../runs/s6a/metrics.jsonl) | Actual local history through +25M |
| [Paired +25M/+50M research review](../evaluations/COMPARISON_THROUGH_75M.md) | Metrics, all paired outputs, examples, diagnostics, and human-rating limits |
| [Initial HPC provenance report](../evaluations/PROVENANCE_50M.md) | Honest PARTIAL conclusion when SSH authentication blocked access |
| [Verified HPC follow-up](../evaluations/PROVENANCE_50M_FOLLOWUP.md) | User-supplied +50M checkpoint/log/scheduler evidence and scoped PASS |

The follow-up's primary HPC transcript is fingerprinted as `9fe055eb4bc4e15f9e57ec4b609280ec024594f9f88ec61522a32b39e131e25d`. Its reported remote metrics hash is `bb904d41491156943248a33b9107a75bd7c2c5f1c90436f2b2ce253dc9c12db6`.

**Final status as of this report:** Stage 6A has completed and verified the +50M review gate; its fixed-window language-modeling metrics are better than +25M and the comparable Stage 5 initialization; its broad storytelling advantage remains unproven. The next step is a quality decision and, if approved, a controlled +75M continuation—not a repair of the now-verified +50M checkpoint.

Only this new report was created for the current request. Existing reports, source/configuration, data, checkpoints, optimizer/RNG state, and provenance were preserved. No new HPC access, training, generation, tests that train temporary models, checkpoint transfer, job submission/cancellation, or Git mutation was performed to prepare it.
