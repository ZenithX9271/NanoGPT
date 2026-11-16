# Stage 7/8 resource plan — corrected evidence

Updated 4 November 2026. **The 32 GB-per-job figure is unverified and excluded from RAM, disk and GPU allocation decisions.** The previous hypothetical 32 GB disk partition is withdrawn.

## Evidence and limits

| Item | Established evidence | Still unknown |
|---|---|---|
| Host RAM | Stage 6 PBS jobs requested 4 GB | Actual peak use, Stage 7 requirement and maximum allowed request |
| GPU | Stage 6 used Tesla V100 GPUs with 32 GB VRAM | Current availability, exclusivity, permitted GPU count and aggregate GPU-hours |
| Storage | Home-directory corpus/checkpoints persisted between jobs | Usable bytes, inode quota, retention/backup guarantees and I/O suitability |
| Walltime | User earlier reported up to 64 hours, potentially more | Administrator-confirmed queue/project walltime and charging rules |

See [resource evidence](configs/resources.user_reported_2026-11-04.json). The historical milestone report, original resource worksheet, candidate configuration and PBS templates are unchanged. No SSH, deployment, transfer, installation, submission or HPC modification occurred.

## Sequence independent of the unverified allocation

1. Continue local bounded evaluation work: human review of the 20 draft cases, rubric calibration, independent sealed-set ownership and small-sample source/license/lineage review. No large corpus or model download is needed.
2. Obtain the exact answers in [the administrator checklist](ADMIN_RESOURCE_CONFIRMATION.md). Confirm usable capacity after preserving every Stage 6 artifact. Home persistence does not establish a quota or backup guarantee.
3. Review a bounded benchmark targeting the historically observed V100 32GB. Existing requests of **20 GB host RAM, 15 minutes and 4 GiB probe workspace are proposals**, not granted resources. Stage 6 requesting 4 GB does not prove the new trainer fits 4 GB. If that is the actual ceiling, measure initialization, evaluation and checkpoint save/load with headroom, then reduce/stage the probe or request a permitted larger allocation.
4. Measure progressively: small batch/sequence first, then the intended 1,024 context and compatible FP16. Measure peak host RAM and VRAM separately, successful-update throughput, evaluation cost and complete checkpoint save/restore on the intended persistent filesystem. Validate optimizer/scaler/RNG/sampler continuation. Use temporary scratch only as a cache until retention is known.
5. Choose a fixed main LR horizon after measurement. **2B is a research option, not a commitment.** A 50–100M review gate can lie within that horizon; do not silently extend a short cosine schedule to 2–5B. GPU restart validation, cumulative resource accounting and immutable checkpoint preservation remain required by the [milestone report](MILESTONE_1_REPORT.md).
6. Develop the independent pretrained assistant/SFT/compression track after confirming runtime compatibility, model staging permission, cache/export disk space and RAM/VRAM. Concurrent development does not establish permission or resources for concurrent GPU jobs. No A100 assumption is needed.

## Storage demand, without an invented quota

Allocate from observed usable byte and inode headroom after preserving Stage 6. No existing artifact cleanup is authorized.

- With the measured uint16-compatible tokenizer, 2B token IDs occupy **4 GB**, and 5B occupy **10 GB**, decimal, before metadata, raw data and temporary shards. uint32 doubles these values. Bounded streaming avoids retaining the full raw corpus, but materialized token shards or a replay/streaming scheme still require reproducible manifests and sufficient confirmed capacity.
- For 122,702,592 parameters, FP32 weights plus two FP32 Adam moments alone occupy approximately **1.472 GB per checkpoint**. Other state/serialization adds overhead. Account for every retained version plus a complete temporary new save, reports and headroom. Atomic publication does not remove temporary-file capacity requirements.
- A nominal 1.7B model needs approximately **3.4 GB** for two-byte weights or **6.8 GB** for FP32 weights alone. Cache duplication, conversion, pruning recovery, adapters/optimizer state and exports need additional measured space. Retaining all variants simultaneously is not assumed.
- Home, project storage, shared scratch and node-local scratch need separate quotas and retention policies. A per-job temporary directory is not a durable continuation archive.

## Runtime demand, without an assumed allocation

One GPU, training only. These are hypothetical successful-position rates, not observed V100 speeds:

| Positions/second | 2B hours | 5B hours |
|---:|---:|---:|
| 1,000 | 555.6 | 1,388.9 |
| 5,000 | 111.1 | 277.8 |
| 10,000 | 55.6 | 138.9 |
| 20,000 | 27.8 | 69.4 |

Evaluation, saves, restores, startup and contingency are additional. A hypothetical single 64-hour job would require at least 8,681 positions/second for 2B, or 21,702 for 5B, before overhead. The 64-hour queue limit still needs confirmation. Multiple jobs require reviewed stopping points and verified continuation; no automatic resubmission is proposed.

Retain the proposed allocation policy once an aggregate budget is known: at most 50% scratch research, at least 40% adaptation/compression and 10% contingency. Main token budget, LR horizon, disk limits and approval remain unset. The next launch decision is a bounded benchmark after resource/runtime review.
