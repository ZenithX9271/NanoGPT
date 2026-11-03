# Stage 6 +50M provenance: verified follow-up

Audit date: 1 November 2026, Asia/Kolkata.

**Conclusion: PASS for checkpoint identity and documented continuation provenance, based on the supplied HPC terminal output.** The +50M checkpoint matches the expected SHA256; checkpoint metadata, appended training events, current source/configuration identities, and PBS history agree. Training resumed at Stage 6 update 3052, finished at update 6104, and exited normally at the review gate. The training and evaluation PBS jobs both have `Exit_status = 0`.

This follow-up resolves the access-related gaps in [the previous provenance report](PROVENANCE_50M.md) to the extent documented below. That report and [the earlier story comparison](COMPARISON_THROUGH_75M.md) remain unchanged. A provenance PASS does not establish an overall storytelling-quality improvement.

## Evidence and scope

The user ran the read-only batch through their authenticated HPC terminal, shown as `cs1230342@login07`, and supplied its output in:

`/Users/shash/.codex/attachments/fc27d34f-8468-4fa6-a36d-4ee865962bb0/Pasted text.txt`

Attachment SHA256: `9fe055eb4bc4e15f9e57ec4b609280ec024594f9f88ec61522a32b39e131e25d`.

The attachment contains 506 lines / 88,033 bytes, including five file hashes and checkpoint metadata records, five optimizer summaries, five saved-RNG summaries, three invocation records, continuation metrics, selected training/evaluation logs, successful PBS-history responses, and the terminal `AUDIT_FINISHED` marker. No checkpoint inspection, metrics parsing, or log-read error was reported. The command echo near its beginning is visually garbled; the ensuing structured results are present through the successful completion marker.

The assistant inspected and cross-checked this supplied output locally; it did not directly operate the authenticated terminal or make another SSH connection. Statements about HPC contents are therefore verified against the supplied terminal evidence, not a separate direct retrieval. All remote paths below use:

```text
PROJECT = /home/cse/btech/cs1230342/zenithX9271/NanoGPT
RUN     = /home/cse/btech/cs1230342/zenithX9271/NanoGPT/stage_6_tinystories/runs/s6a
```

## 1. Checkpoint identity verification

All five entries were previously reported as regular files. Their exact paths are `RUN/` followed by the filenames in this table; modification times below are converted from the batch's UTC timestamps to IST (+05:30).

| Checkpoint filename | Bytes | Modified, IST | Stage 6 updates | Additional positions | Total positions | Total updates |
|---|---:|---|---:|---:|---:|---:|
| `snapshot_additional_10000000.pt` | 95,591,931 | 30 Oct 2026 16:59:42 | 1221 | 10,002,432 | 108,306,432 | 13,221 |
| `snapshot_additional_25000000.pt` | 95,591,803 | 31 Oct 2026 15:31:48 | 3052 | 25,001,984 | 123,305,984 | 15,052 |
| `best.pt` | 95,591,803 | 1 Nov 2026 15:40:27 | 6000 | 49,152,000 | 147,456,000 | 18,000 |
| `snapshot_additional_50000000.pt` | 95,591,995 | 1 Nov 2026 15:40:58 | 6104 | 50,003,968 | 148,307,968 | 18,104 |
| `latest.pt` | 95,591,995 | 1 Nov 2026 15:40:59 | 6104 | 50,003,968 | 148,307,968 | 18,104 |

| Checkpoint filename | SHA256 |
|---|---|
| `snapshot_additional_10000000.pt` | `430216811c4d1f6c258e609d961bbc5015e1336ea4a10c92bb383800e3639abd` |
| `snapshot_additional_25000000.pt` | `50f94bf80a142e58fdd0ef3daa933fe4c015c2353a7156d8be288f95cf8f50a6` |
| `best.pt` | `e78edb3b96ef956aae64c269da452bb7af39a9e9534f0b4d9cf307d20ab7c573` |
| `snapshot_additional_50000000.pt` | `40d145f6739896f923438e54f5a3235a26bfdb0d5cc136bfafb816871658d04e` |
| `latest.pt` | `40d145f6739896f923438e54f5a3235a26bfdb0d5cc136bfafb816871658d04e` |

The +50M snapshot and latest checkpoint are **byte-identical** and match the expected evaluated identity. The 10M and 25M snapshots match the preserved local checkpoint hashes established in the earlier audit.

Every checkpoint's accounting satisfies:

- `additional_tokens = stage6_updates × 8192`, where 8192 = batch 32 × context 256.
- `tokens = 98,304,000 + additional_tokens`.
- `completed_updates = 12,000 + stage6_updates`.

The 50M snapshot and latest record saved thresholds `[10000000, 25000000, 50000000]`. Their format is `nanogpt_stage6_v1` and their origin identifies the pinned Stage 5 initialization hash `2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce`.

### Which checkpoint was evaluated?

`PROJECT/stage_6_tinystories/evaluations/50m/evaluation.json` names **`RUN/latest.pt`**, with the exact `40d145…8d04e` hash. The HPC and local evaluation JSONs share SHA256:

`701abe2886ad8186087a22f1f0865e6586789af1ac96e16f24443586b09dc343`.

Thus the previously compared 159 stories belong to **update 6104 / +50,003,968 positions**, not `best.pt` at update 6000.

The actual best checkpoint has development CE **2.0948602450080216**; latest has **2.0949381892569363**. The final score is higher by only **0.000077944249 nats/token** (approximately 0.00372%). This small last-point reversal does not alone establish a plateau or overfitting, and it does not establish that the best checkpoint's stories would be better. No generation from that best checkpoint was performed in this audit.

## 2. Training continuation verification

The appended metrics contain three invocations: initial Stage 5 initialization, a resume from Stage 6 update 1221, and the +50M continuation from update 3052. All three record identical configuration, six-file source identities, and data identities. The original `provenance.json` retaining the first invocation is consistent with `train.py`'s write-once provenance behavior; the appended invocation events document subsequent resumes.

The +50M invocation at **1 Nov 2026 15:24:13 IST** records:

| Field | Value |
|---|---|
| `mode` | `train` |
| `arguments.resume` | `stage_6_tinystories/runs/s6a/latest.pt` |
| `arguments.init_from` | `null` |
| `starting_stage6_updates` | **3052** |
| `arguments.stop_after_updates` | **6104** |
| `arguments.config` | `stage_6_tinystories/configs/continue.json` |
| `arguments.walltime_seconds` | 14,400 |

Training log `nanogpt_s6a.o1128982` also prints `resumed_updates: 3052` at line 49. This startup output is produced after the verified code's `Engine.resume()` returns. It is supported by the genuine 25M snapshot, matching configuration/identities, and final accounting. There is no invocation restarting from Stage 5 during this continuation.

The first logged continuation training point is update 3100 and the final one is update 6104. All expected 50-update log points from 3100 through 6100, plus final update 6104, are present once: **62 continuation training entries**. The final checkpoint, summary, metrics exit, and PBS log agree on:

- Stage 6 updates: **6104**.
- Additional positions: **50,003,968**.
- Total positions: **148,307,968**.
- Total updates: **18,104**.
- Exit reason: **`review_gate`**.

The continuation therefore accounts for another **3052 optimizer updates / 25,001,984 positions** beyond the +25M gate. No additional training invocation or interrupted exit is recorded within that interval. This statement concerns the recorded trajectory; it is not a blanket claim about unrelated earlier job attempts.

## 3. Optimizer, RNG, and schedule verification

### Optimizer — PASS on state and execution-record evidence

All five inspected checkpoints have **208 optimizer states**, no missing parameter-state entries, no missing moment tensors, no mismatched first/second-moment pair shapes, and no nonfinite moment values reported by the batch.

| Checkpoint | All recorded AdamW step counters |
|---|---:|
| 10M snapshot | 13,221 |
| 25M snapshot | 15,052 |
| Best at update 6000 | 18,000 |
| Latest / 50M snapshot | 18,104 |

The two parameter groups retain betas `(0.9, 0.95)`, epsilon `1e-8`, `amsgrad=false`, and weight decay `0.1` / `0.0`. Latest's saved group LR is **0.00006592023642149754**.

The source matching the invocation and checkpoint explicitly loads `optimizer.load_state_dict`, validates moments/counters and hyperparameters, and returns from resume before startup/invocation recording. Together with the successful startup at 3052 and accumulated final states, this supports restoration of the existing optimizer rather than a fresh optimizer start. The inspection checks paired moment shapes; it did not independently reconstruct the model to compare each final moment shape against its corresponding parameter.

### RNG — PASS on saved-state and verified resume-path evidence

The 25M and 50M checkpoints contain all five expected RNG components: Python, NumPy, Torch CPU, Torch CUDA, and the independent training sampler. Torch CPU and sampler states are uint8 tensors of length **5056**; one CUDA generator state is saved.

The verified `core.py` resume path restores each component, including `torch.cuda.set_rng_state_all` and the training sampler's state, before returning. The matching source hash, successful resumed startup, and complete saved states provide execution-record support for restoration. No RNG-restoration exception or subsequent training restart is reported.

Selected saved-state fingerprints:

| State | 25M snapshot | Latest / 50M snapshot |
|---|---|---|
| CUDA RNG | `f2dabc19d37beaeebd416096422c72335f2b29c307540ba37b46a090e593f5b7` | `982e67912866361959e121d5f03c0f7eaa3705596207a9104374634449059892` |
| Training sampler | `27c8112262a90099e29b58ffaeedb6578591101fde590685e1ffe16a1f06543a` | `0e4dd8e538db4552c9e169e1d203354f2e06cf91120427530101959834eeb5a0` |

Python, NumPy, and Torch CPU fingerprints remain unchanged across these saved gates, while CUDA and sampler fingerprints change. That is consistent with GPU training and a separate CPU sampling generator; unchanged unused RNG streams do not by themselves indicate a reset.

**Limit:** saved-state presence and changed fingerprints alone cannot prove exact historical RNG continuity. The stronger evidence is the source-verified resume path and its successful execution. An independent bit-for-bit replay of the trajectory was not performed, and no RNG state was restored or advanced by the audit.

### Learning-rate schedule — PASS, independently recalculated

The stored configuration preserves the original **100,000,000 additional-token horizon**, 200-update warmup, start LR `3e-5`, peak `1e-4`, and minimum `3e-5`. The update-6104 stop is a review gate; it does not shorten or reset the LR horizon. Whole-update rounding gives the configured budget of 100,007,936 positions.

All **62** logged continuation LRs were independently recomputed locally from the supplied checkpoint configuration and the code's pre-update counter (`logged update − 1`). **Maximum absolute error: 0.0.** This agrees with the HPC batch's own schedule check.

Selected actual training records:

| Stage 6 update | Minibatch training loss | Learning rate |
|---:|---:|---:|
| 3052, prior gate | 2.28072429 | 0.0000907054056836 |
| 3100 | 2.28381371 | 0.0000904050641499 |
| 3500 | 2.27873898 | 0.0000877510433332 |
| 4000 | 2.23950028 | 0.0000840870642443 |
| 4500 | 2.29998279 | 0.0000800968803337 |
| 5000 | 2.18463016 | 0.0000758486852857 |
| 5500 | 2.22479486 | 0.0000714150822880 |
| 6000 | 2.22284794 | 0.0000668718432171 |
| 6104 | 2.22855520 | 0.0000659202364215 |

The logged continuation minibatch losses range from **2.0508680 to 2.3894897**; they are noisy and do not decrease monotonically. These dropout-enabled minibatch values must not be substituted for the fixed-window evaluation-mode train CE below. Logged preclip gradient norms range from **1.3201251 to 1.5391567**; exceeding the configured clipping threshold of 1.0 before clipping is not itself an error.

## 4. Source, configuration, data, and evaluation-patch identities

### Training identities — PASS

All six actual +50M checkpoint-recorded source identities match the current HPC files, the continuation invocation, and the local source files read in this follow-up:

| Stage 6 source | SHA256 |
|---|---|
| `__init__.py` | `02f86e3862c8672123220007e6831262abf939756c9fd053907981fb7cd96f89` |
| `core.py` | `d339cc0a7bee57197754941734266be25cbaabe75cb1a248fdef1b9394eebe3e` |
| `data.py` | `05acdb60f7c8d5bdf94e40de02ac76569299a455ff0243762ee41be11ebdbfaf` |
| `evaluate.py` | `f6d3e32336aed91a8e247755f1392ad6ba8ff7fff742853ec41d896810b5ddd1` |
| `model.py` | `7aca47adbae88c80bdc9d0339dc662761d179f51e5e274715dc9ae00e954cbb5` |
| `train.py` | `9ccb6273e90522da66f1e64b30bdf6cb67d25483e1041a3b859343f66bb3577a` |

Actual +50M architecture metadata confirms **8 attention heads, 6 layers, width 256, context 256, vocabulary 8192, dropout 0.2, and tied embeddings**. The task's earlier four-head description was incorrect. This is the established approximately 6.9M-parameter architecture; this batch did not independently recount all +50M model parameters.

The current HPC `configs/continue.json` SHA256 is `0df15f4064a433c9f6db1678eeefdcdb0edf0de0385c408f5dd3cf1e8223f7b7`. Its parsed contents match the actual +50M checkpoint, all three invocation configurations, and the local configuration exactly. Source/configuration/data checks in the verified resume implementation remain intact. No invalidating training-source change is evident.

| Data identity | SHA256 matching current HPC file, +50M checkpoint, and evaluation record |
|---|---|
| Manifest | `89c9df839572ef6a1890caf4b1144229090b8ced8b4ea2c66ed5dd090e17c9a7` |
| Tokenizer | `99055f35c77392e4df4486685b60fee1de1d95e2c373c26f487cf9944746d038` |
| Evaluation prompts | `f1c83474faa2b56d670055d6da373a1460eccc007db296bcf509f44f5737429c` |

### Temporary evaluation patch — execution context now documented

Evaluation job **1129007.pbshpc**, log lines 17–20, explicitly distinguishes:

- Canonical repository `model.py`: `7aca47adbae88c80bdc9d0339dc662761d179f51e5e274715dc9ae00e954cbb5`.
- Temporary evaluation source at `/var/tmp/pbs.1129007.pbshpc/nanogpt_eval50m.WWb56V/stage_6_tinystories/model.py`: `b058a3621e93103100e017742bc15a08cc9a586f750b59df5b93fdc1a42417d3`.

The current HPC archive contains `stage_6_tinystories/recovery/generation_compat_model.py` with that same patch hash and `model.py.original` with the original hash. The earlier diff of these source identities changes only PAD/BOS suppression in generation: a combined indexing assignment becomes separate assignments.

The actual training checkpoint records the original source. The evaluation job documents the patched temporary source, and the current canonical HPC file still matches the original. This resolves the previous ambiguity between training identity and evaluation identity; the patch is not evidence that the training model or resume guard was changed.

The temporary directory's current existence and every other file in that temporary tree were not independently inventoried. The patch's exact execution-time identity is supported by the job log, retained archive hash, matching completed evaluation, and PBS success. This is not a full execution-time hash manifest of the temporary tree.

## 5. PBS completion, logs, and intermediate evaluations

### Training job — PASS

- PBS job: **1128982.pbshpc**, name `nanogpt_s6a`.
- Node: **vsky034**, Tesla V100-PCIE-32GB.
- Runtime records: Python 3.7.9, PyTorch `1.10.0a0+git36449ea`, NumPy 1.18.5, CUDA 11.0, FP32/manual causal attention.
- Log start: **1 Nov 2026 15:23:36 IST**; wrapper finished **15:41:00 IST**.
- Trainer exit: `review_gate`, update 6104, runtime **1025.2025 seconds**.
- PBS: **`job_state = F`, `Exit_status = 0`**, walltime **00:17:27**, `obittime` **15:41:03**.

The scheduler result establishes normal job completion in addition to the trainer's own exit record. The root bootstrap log and Stage 6 copied log have the same hash and are duplicate evidence of one job:

```text
PROJECT/nanogpt_s6a.o1128982
PROJECT/stage_6_tinystories/hpc/logs/1128982.pbshpc.log
SHA256 d5dae90c7a0236d3f85b452f2494a0bc1d9369fe5ca25f5ba33ad90b2f295974
```

### Full evaluation job — PASS

- PBS job: **1129007.pbshpc**, name `nanogpt_eval50`.
- Node: **vsky010**, Tesla V100-PCIE-32GB; same recorded software stack.
- Log start: **15:53:31 IST**; evaluation finished **16:04:10 IST**.
- Evaluation: `status=complete`, **159/159** stories, matching +50M checkpoint and metrics.
- PBS: **`job_state = F`, `Exit_status = 0`**, walltime **00:10:43**, `obittime` **16:04:14**.

Its two matching log copies are:

```text
PROJECT/nanogpt_eval50.o1129007
PROJECT/stage_6_tinystories/hpc/logs/1129007.pbshpc.log
SHA256 dcb97732b4a4f9f35de09c2c9e4c51afc97159460db8eb08ceb65d7f4ccbd579
```

Both selected logs contain the generic compiler/CUDA module warning about GPU-node availability. Both subsequently identify an allocated V100 and complete successfully. No traceback or training-failure message is present in the supplied selected excerpts. Earlier small job logs were inventoried but not investigated as part of this continuation's completion check.

The job logs report a **quick** transfer verification with 104 files present and 59 files hashed. This is not a full rehash of every dataset shard.

### Actual intermediate scores

These are recorded measurements, not interpolations:

| Stage 6 update | Additional positions | Evaluated train CE | Development CE | Development perplexity |
|---:|---:|---:|---:|---:|
| 3052, prior gate | 25,001,984 | 2.083490644 | 2.133726095 | 8.446279895 |
| 3500 | 28,672,000 | 2.076661840 | 2.127759073 | 8.396030822 |
| 4000 | 32,768,000 | 2.068476666 | 2.121727955 | 8.345545764 |
| 4500 | 36,864,000 | 2.063532490 | 2.113604868 | 8.278028766 |
| 5000 | 40,960,000 | 2.060667116 | 2.110322647 | 8.250902983 |
| 5500 | 45,056,000 | 2.051644601 | 2.102635359 | 8.187719088 |
| 6000, saved best | 49,152,000 | 2.045248814 | 2.094860245 | 8.124305479 |
| 6104, evaluated latest | 50,003,968 | 2.044401698 | 2.094938189 | 8.124938746 |

Development CE improves at each scheduled point through update 6000, followed by the very small reversal at 6104. This supplies the previously missing evidence that the +25M-to-+50M gain was supported by intermediate measurements. It does not establish that the next interval will improve by the same amount.

The full metrics fingerprint reported by the batch is `bb904d41491156943248a33b9107a75bd7c2c5f1c90436f2b2ce253dc9c12db6`. Its event counts are 3 invocations, 3 exits, 125 training events, 16 evaluations, and 30 checkpoint events. Remote summary SHA256: `cf8ef29411f728a6d8289709f313e4fb938aeefa95ddbcfdae6dc1b1a7edf334`.

## 6. Remaining limits and risks

1. The evidence is user-supplied terminal output. It is internally consistent and its attachment is fingerprinted; the assistant did not independently retrieve the HPC files.
2. The resume invocation records the mutable `latest.pt` path and starting update, not an immutable input-checkpoint SHA256 captured at the instant of resume. The preserved 25M snapshot, prior local latest hash, counter/identity agreement, and verified resume path support the lineage. This is not an atomic cryptographic record of every past file access.
3. Optimizer/RNG restoration is supported by the verified implementation, successful invocation/startup, and saved states. No independent bit-for-bit uninterrupted-versus-resumed replay was performed, as that would exceed this read-only audit.
4. The batch checks final optimizer moment finiteness and pair shapes, but does not independently rebuild the architecture, perform a forward pass, or validate every final model tensor's numerical properties. No such claim is made.
5. Manifest/tokenizer/prompt identities match; every remote shard was not rehashed in this phase. The job's quick verifier is explicitly narrower than full payload verification.
6. The temporary evaluation model hash is documented, but the complete temporary tree and its entire execution-time environment were not independently fingerprinted. The six canonical training source identities do match.
7. The Mac's existing run checkpoint copy remains at +25M from the earlier audit. This follow-up transfers no checkpoint and does not make it a local +50M resume point. A future approved continuation must use the verified +50M state on HPC.
8. Human story ratings remain absent from the earlier comparison. Closing provenance gaps does not eliminate the mixed qualitative outcomes, loops, prompt failures, or the need for blinded judgments.

These limits qualify the scope of the PASS. None of the supplied results establishes corruption, a fresh optimizer restart, an LR reset, a source-identity bypass, an interrupted +50M gate, or a failed evaluation job.

## 7. Conclusion and next decision

| Check | Conclusion |
|---|---|
| Exact evaluated +50M checkpoint hash and accounting | **PASS** |
| +50M snapshot equals latest | **PASS**, identical SHA256 and bytes |
| Resume from recorded +25M state to update 6104 | **PASS on documented lineage** |
| Optimizer restoration / final accumulated state | **PASS on verified code, startup, and saved-state evidence** |
| RNG restoration / saved state | **PASS on verified resume-path and state evidence; independent replay not tested** |
| Original LR schedule retained | **PASS**, 62 logged points, recomputed maximum error 0.0 |
| Canonical source/config/data identities | **PASS** |
| Separate generation-patch identity | **PASS on job-log and retained-hash evidence** |
| Training and evaluation PBS completion | **PASS**, both finished with exit status 0 |
| Overall provenance audit | **PASS within the evidence scope stated above** |

The earlier provenance blocker is cleared. The original research comparison's **1.82% development CE / 3.80% perplexity improvement** can now be associated with the verified +50M continuation, and its intermediate loss trajectory is available. Its conclusion that broad storytelling improvement remains unestablished still stands.

The next useful decision is to review blinded paired human judgments on the already generated +25M/+50M stories. If those show no material quality regression, a bounded +75M experiment using the original schedule and exact +50M optimizer/RNG state is defensible. This audit supplies no reason to reset LR, change architecture, patch training source, or overwrite checkpoint metadata. The marginally better update-6000 CE is insufficient by itself to justify changing the established continuation state.

Only this new follow-up report was created. No existing report, source, configuration, checkpoint, provenance artifact, optimizer/RNG state, HPC file, or Git state was edited; no training, generation, job submission/cancellation, or checkpoint transfer was performed by the assistant. Stop here and await the user's approval for any subsequent action.
