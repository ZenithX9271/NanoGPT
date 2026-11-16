# Stage 7A: final authorized continuation to the original 2B horizon

**Final status (verified 16 November):** all four phases completed successfully
at 2,000,011,264 positions. The final development review and sampled heldout
evaluation passed, and the final Mac checkpoint backup is checksum verified.
See [the final results](TWO_BILLION_FINAL_REVIEW.md).
The entries below describe submission and earlier start checks.

On 15 November 2026 the user approved the final approximately 500M positions,
resuming the backed-up and reviewed 1.5B checkpoint. This completes the original
2B-position training schedule. No training beyond that schedule is authorized
or submitted by this workflow.

## State, source and limits

The starting endpoint is update 91,553 / 1,500,004,352 successful positions:
`review_u91553_positions_1500004352.pt`, SHA256
`d641a56858874f563a55e84b8084aa199f6bef3af9e210d36957f7c72d5da642`.
Its complete Mac backup and receipt were verified before submission. The passed
1.5B expanded review is pinned by SHA256
`f8ffcf3639a11b443222953f6e4bb892ac3d79b361554d7a5e531e7a02e3b881`.
Its larger-review dev perplexity is 26.02355677.

The 122,702,592-parameter model, all fifteen frozen training source identities,
original serialized configuration, corpus, tokenizer, optimizer, scaler and
RNG/sampler state are preserved. The original 2B learning-rate horizon is not
reset or extended. The existing 52-file read-only deployment is reused without
modification. The new diagnostic helper is pinned separately under the new
workflow root, outside the frozen training source identity. Stage 6 is untouched.

Four allocations request one V100, two CPUs and 16 GiB RAM for at most 02:30:00
each. The hard total is ten additional GPU-hours, including the final review
and heldout evaluation. The trainer's existing deadline argument narrows its
four-hour serialized process allowance to the allocation deadline, retaining
the existing five-minute checkpoint margin. Expected compute is about nine
hours; queue delays are additional.

Prior cumulative allocation accounting is 26.766844887 GPU-hours. The external
plan reserves eight GiB additional run storage and caps total run storage at
32 GiB. The verified initial unique file size is 17,695,495,507 bytes, counting
hard links once. Before each phase, capacity for an additional endpoint and an
atomic-save temporary is checked against the limits and filesystem free space.
This does not establish the account's scratch quota.

## Submitted chain

| Phase | Job | Start update | Target update | Cumulative target positions |
|---|---|---:|---:|---:|
| 1 | `1150576.pbshpc` | 91,553 | 99,183 | 1,625,014,272 |
| 2 | `1150577.pbshpc` | 99,183 | 106,812 | 1,750,007,808 |
| 3 | `1150578.pbshpc` | 106,812 | 114,441 | 1,875,001,344 |
| 4 | `1150579.pbshpc` | 114,441 | 122,071 | 2,000,011,264 |

All jobs were submitted with user holds. Resources, project, output paths,
environment values, sandbox and strict forward `afterok` dependencies were
verified before user holds were released in reverse order. Rerunnable is false,
so PBS does not automatically rerun a failed allocation. Dependency holds remain
on phases 2–4. At the release check phase 1 was queued, not yet claimed to be
optimizing. Each launch verifies the baseline backup/review and pinned artifacts,
the previous endpoint and rolling checkpoint checksums, counters and full source,
config and corpus identity. The run lock permits only one writer.

## Original-horizon completion and review

The frozen trainer labels intermediate stops `review_gate`, but labels the
original final horizon `completed`. The export helper now accepts that final
reason only when the explicitly supplied original configuration computes the
same target update. Intermediate exports still require `review_gate`. A partial
allocation is exported for recovery but fails and blocks dependent work.

At full completion, phase 4 preserves
`review_u122071_positions_2000011264.pt`, then strictly restores it for evaluation.
It reproduces the final fixed FP32 probes and scores the same 512 independent
windows / 524,288 positions on train, dev and both dev domains. Development CE
regression above 0.05 against the pinned 1.5B reference fails the numerical review.
The same eight prompts are generated greedily and with seeded temperature-0.8 /
top-k-50 sampling and decoded offline.

The reserved heldout split is scored separately, at seed 20261115, for 512
windows / 524,288 positions. The helper rejects heldout requests before the
original completed horizon. Heldout scores are reported outside the matched
development score dictionary and are not used for development admission.
There is no automatic training beyond 2B. Numerical PASS is not an assurance
of factual correctness or assistant quality.

## Validation and evidence

The wrapper passed `bash -n`; the embedded Python admission and accounting blocks
parsed. The phase boundaries, contiguous chain, rounded full horizon and total
walltime ceiling were checked. All twelve focused tests passed locally and in
the actual HPC Python 3.7 / PyTorch 1.10 runtime, using the newly deployed helper.
They include acceptance of original-horizon completion, rejection of an early
completion claim, preservation of partial checkpoints, protection of immutable
endpoints, matched reference sampling, cached generation and final-only heldout
access. The CPU tests use disposable tiny fixtures, not the production heldout
data. Their fixture thread count was matched to the review runtime so exact
resume checks remain enabled. The actual phase-1 launch admission block passed
on HPC before any GPU job was released.

Wrapper SHA256:
`9a321f22fa5a851c72d0077f1888f89726207386d4481e302060b8af7ed9ea5e`.
New helper SHA256:
`35fe076df4c4e697a6d0a8b4b4fdbfc421ff6c7d86939a169252747fc5957bf0`.

HPC root:
`/scratch/cse/btech/cs1230342/nanogpt_stage7/continuation_to2b_20261115`.
The plan is `continuation_to2b_20261115.json`; `launch_artifacts.json` pins inputs
and the new helper. Live logs are `logs/phase_1.log` through `logs/phase_4.log`,
and `continuation_<job-number>/job.log`. Submission, admission, checksums and
test receipts are saved locally under `reports/continuation_to2b_20261115/`.

## Optimization start verified

Phase 1, `1150576.pbshpc`, started on `vsky004` at approximately 14:28 IST on
15 November. At allocation walltime 00:01:43 it had passed
`CONTINUATION_IDENTITY_BACKUP_REVIEW_AND_BUDGET_PASS`, identified the V100 32 GiB,
and printed resume from update 91,553 to 99,183. The metrics independently showed
update 91,640 / attempt 91,675: 1,501,429,760 successful positions, LR
0.00007057893558, loss 3.026431918 and gradient norm 1.759037852, with a successful
optimizer step and AMP scale 65,536. The skipped-attempt difference remained 35,
unchanged from the initial checkpoint at this check.

This confirms actual optimization, not only submission. Phases 2–4 remained
on their expected dependency holds. No claim is made yet that 1.625B or 2B
has completed.
