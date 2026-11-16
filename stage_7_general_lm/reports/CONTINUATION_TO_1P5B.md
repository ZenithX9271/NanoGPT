# Stage 7A: authorized continuation from 1B to 1.5B

**Completed and verified on 15 November:** all four phases exited successfully
at 1,500,004,352 positions; the expanded review passed and the Mac checkpoint
backup is checksum verified. See [the final review](ONE_POINT_FIVE_BILLION_REVIEW.md)
for matched scores and generation findings. The entries below describe submission
and admission checks.

On 13 November 2026 the user approved the proposed continuation from the verified
1B checkpoint to a 1.5B review milestone. This adds approximately 500M successful
training positions. It does not authorize automatic continuation to 2B.

## Preserved state and baseline

The exact 122,702,592-parameter model, tokenizer, corpus, serialized config and
original 2B learning-rate horizon remain unchanged. Resume restores optimizer,
AMP scaler, RNG and sampler state. The existing frozen deployment at
`/scratch/cse/btech/cs1230342/nanogpt_stage7/deploy_1b_20261112` is reused; the new
wrapper and plan live outside it. No Stage 6 files were changed.

The starting checkpoint is `review_u61036_positions_1000013824.pt`, SHA256
`dc195f441d332e6b8ae21d505af293452e6a7e2c66e45d8ff7a6cdaf023694da`.
Its Mac backup was rehashed before submission and still matches. The passed 1B
review is pinned by SHA256
`5c6857aeeb86f81a2af0f6d166d59c9ffe05dcdc6136d7066933d07f279a54bd`.
Its expanded dev perplexity is 29.90326804 on the fixed 512-window review set.

The prior allocation baseline is 17.907350713 GPU-hours, including all earlier
training allocations and the standalone 1B review. The derived ledger retains
the source training and review allocation records and their hashes.

## Limits and submitted jobs

Each job requests one V100, two CPUs, 16 GiB RAM and a maximum 02:30:00 walltime.
The four-job hard walltime total is ten additional GPU-hours, including the final
review. The unchanged trainer's existing deadline argument tightens its original
four-hour process allowance to the allocation deadline, with its existing
five-minute checkpoint margin. There is no automatic retry or spending extension.

| Phase | Job | Start update | Target update | Cumulative target positions |
|---|---|---:|---:|---:|
| 1 | `1147658.pbshpc` | 61,036 | 68,665 | 1,125,007,360 |
| 2 | `1147659.pbshpc` | 68,665 | 76,294 | 1,250,000,896 |
| 3 | `1147660.pbshpc` | 76,294 | 83,924 | 1,375,010,816 |
| 4 | `1147661.pbshpc` | 83,924 | 91,553 | 1,500,004,352 |

The rounded final target is the first complete 16,384-position update reaching
1.5B. Each job depends on the preceding job's successful exit. All four were
submitted with user holds, checked for requested resources, output paths and
forward `afterok` dependencies, then released in reverse order. PBS's numeric
environment-variable representation and reverse `beforeok` dependency records
were normalized during verification. User holds have been removed; dependency
holds remain on phases 2–4. Phase 1 was queued at the release check.

An exclusive run lock prevents overlapping writers. Each launch rechecks source,
config, corpus manifest, initial Mac backup receipt, passed baseline review,
previous endpoint checksum, rolling checkpoint checksum and counters. A partial
segment is preserved but exits nonzero, blocking dependent optimization. Completed
endpoints are retained independently of later rolling checkpoint replacements.

The existing run contained 11,796,944,476 unique file bytes before launch, counting
hard links once. The external continuation plan allows eight GiB additional run
storage and caps total run storage at 24 GiB, without changing the frozen config.
Each phase reserves capacity for an endpoint and an atomic-save temporary before
training. Cluster free space was checked; this is not a claim about account quota.

## Automatic review at the final endpoint

After preserving the 1.5B checkpoint, phase 4 strictly reloads it and reproduces
its small FP32 evaluation. It then scores the same 512 windows / 524,288 positions
on training, development and both development domains, and generates the same
eight greedy and seeded sampled probes. It compares against the pinned 1B review;
development CE regression above 0.05 causes failure. This is numerical and
qualitative evidence, not a guarantee of reliable answers. It stops at 1.5B
regardless; there is no 2B job in this chain.

## Validation and evidence

The new PBS wrapper passed `bash -n`; both embedded Python blocks parsed.
The phase targets, contiguity, total allocation cap and final stop were checked.
All six checkpoint-review tests passed locally and in the HPC Python 3.7 /
PyTorch 1.10 runtime. The actual phase-1 identity/backup/review/disk admission
block also passed on HPC before submission. The preflight harness's final hash
print initially shadowed a variable; its reporting line was checked separately
and the wrapper checksum verified before submitting any jobs.

Wrapper SHA256:
`4a7e0cf3cf328cd719a5f33083e79fd5489b3bffc1ca4524bf0faa2256419ae0`.

HPC workflow root:
`/scratch/cse/btech/cs1230342/nanogpt_stage7/continuation_to1p5b_20261113`.
The pinned plan is `continuation_to1p5b_20261113.json`; `launch_artifacts.json`
pins the wrapper and baseline inputs. Logs are `logs/phase_1.log` through
`logs/phase_4.log`. Each active job also tees to
`continuation_<job-number>/job.log` under this root. Reports and submission
receipts are copied locally to `reports/continuation_to1p5b_20261113/`.

At the original submission check the work was queued. The completed phase
allocation ledgers now total 8.859494173 GPU-hours, within the ten-hour limit.
The final expanded dev perplexity is 26.02356 versus 29.90327 at 1B. Training
stopped at the 1.5B review boundary; no 2B continuation was submitted.
