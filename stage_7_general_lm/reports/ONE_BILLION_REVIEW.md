# Stage 7A: 1B checkpoint backup and review

**Subsequent status (13 November):** the user authorized continuation to 1.5B.
Four jobs have been submitted; see [the continuation record](CONTINUATION_TO_1P5B.md).
The results below describe the completed 1B endpoint.

## Completed training and verified backup

All four 500M-to-1B continuation phases finished with exit code zero. The final
job, `1144925.pbshpc`, stopped at the requested review boundary: 61,036 successful
updates / 1,000,013,824 successful positions. It did not continue beyond 1B.
The four phases used 8.929044865 GPU-hours out of the authorized ten-hour limit.
All logged update losses were finite. There were 23 skipped AMP attempts out of
61,059 attempts (0.0377%); final optimizer step succeeded.

The full checkpoint, including optimizer, scaler and RNG/sampler state, is copied
to `artifacts/checkpoints/1b_1144925/` on the Mac. Its 1,472,538,167 bytes have SHA256:

`dc195f441d332e6b8ae21d505af293452e6a7e2c66e45d8ff7a6cdaf023694da`

This matches the HPC endpoint record. The backup includes the exact frozen
deployment, training evidence, tokenizer bundle, original configuration, audited
corpus metadata and reference reviews. All 52 deployed file hashes, the fifteen
training source identities, the configuration and corpus manifest were verified.
`mac_backup_verified.json` records the check; `backup_manifest.json` lists file
sizes and hashes. Cloud synchronization completion is not independently verified.

## Fixed evaluation results

These are the same seeded 16 windows / 16,384 positions per stream, evaluated in
FP32 during training. They must not be mixed with the larger review's windows.

| Milestone | Train PPL | Dev PPL | Cosmopedia dev PPL | FineWeb-Edu dev PPL |
|---|---:|---:|---:|---:|
| 100M | 59.67495 | 63.52786 | 32.22266 | 67.32159 |
| 200M | 43.85654 | 47.91284 | 22.53706 | 53.61893 |
| 500M | 34.02517 | 37.77561 | 17.09759 | 42.94451 |
| 750M | — | 34.49046 | — | — |
| 1B | 26.63597 | 31.67967 | 14.36157 | 37.23769 |

Overall fixed-probe dev perplexity fell 16.14% from 500M and 50.13% from 100M.
This measures token prediction on these samples, not answer accuracy.
The full curve is saved in `continuation_to1b_20261112/fixed_evaluation_100m_1b.png`
and PDF. Raw terminal-phase evidence is under the same directory.

## Larger review

Review-only PBS job `1146578.pbshpc` was submitted on 12 November 2026 at 21:18 IST.
It requests one V100, two CPUs and 16 GiB RAM for at most fifteen minutes
(0.25 GPU-hours). It performs no optimization. Its wrapper is
`hpc/review_1b.pbs`, outside the frozen training source identity.

The review strictly restores the 1B checkpoint, reproduces the final fixed
evaluation, then scores 512 independent windows / 524,288 positions on training,
development and both development domains using the earlier review seeds. It
generates the same eight prompts with greedy and seeded temperature-0.8/top-k-50
decoding. Numerical comparisons use both 500M and 750M references. Development
CE regression above 0.05 causes failure. A numerical PASS does not certify
factual correctness or assistant quality.

HPC evidence root:
`/scratch/cse/btech/cs1230342/nanogpt_stage7/review_1b_20261112`

Verified on 13 November: the review finished with PBS exit status zero. PBS
records a start at 23:44 IST on 12 November and walltime 00:04:16. The job's own
allocation ledger records 251.723 seconds / 0.069923 GPU-hours. The initial
delay was queue time, not evaluation time. No further training was performed.
There are no current jobs listed for this account at the verification check.

The strict checkpoint restore and reproduction of the final fixed FP32
evaluation passed. All larger development streams improved against both
references. Each entry below uses the same seed, window count and positions
across milestones; this table is separate from the fixed evaluation above.

| Stream | 500M PPL | 750M PPL | 1B PPL | Reduction vs 500M | Reduction vs 750M |
|---|---:|---:|---:|---:|---:|
| Training | 33.44522 | 30.27984 | 28.11845 | 15.93% | 7.14% |
| Development | 34.97993 | 32.08670 | 29.90327 | 14.51% | 6.80% |
| Cosmopedia development | 17.86507 | 16.13690 | 14.79748 | 17.17% | 8.30% |
| FineWeb-Edu development | 41.10719 | 37.92214 | 35.56641 | 13.48% | 6.21% |

The larger 1B train/dev CE values are 3.336425879 and 3.397967773, a gap of
0.061541894 nats/token. Improving development results support further pretraining;
they do not establish factual accuracy or downstream task performance.

Qualitative inspection of the same eight prompts still finds major weaknesses.
Greedy gravity generation repeats a fluid/tube description; the fraction prompt
never gives a valid comparison method; the spoon prompt says it feels cold
because it is metal instead of explaining heat transfer. Sampled text also
contains factual errors and topic drift. The explanation prompt starts plausibly
with the main idea but then repeats or becomes circular. Generation quality
remains unsuitable for reliable question answering, despite the numerical gains.

Downloaded reports are under `review_1b_20261112/results/`: `review.json`,
`completions.json`, `comparisons.json` and `allocation.json`. Their local SHA256
hashes were checked against the HPC copies. The decoded generations are also
presented in `review_1b_20261112/generation_comparison.md` alongside 500M and 750M.
At the time of this review, training had stopped at 1,000,013,824 positions.
The subsequent authorized 1.5B continuation is recorded in CONTINUATION_TO_1P5B.md;
2B continuation remains outside the submitted chain.
The frozen training deployment and all Stage 6 files remain unchanged.

Validation before submission: the PBS wrapper passed `bash -n`; all six
checkpoint review tests passed locally. The remote wrapper checksum matched
the submitted local file.
