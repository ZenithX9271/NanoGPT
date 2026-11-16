# Stage 7A: 500M review and authorized continuation to 1B

On 12 November 2026 the user approved the proposed 500M backup, larger review,
and continuation to 1B cumulative tokens, with a 750M review milestone. The new
allocation ceiling is ten additional V100 GPU-hours. No training beyond 1B is
authorized by this plan.

## Verified backup and unchanged training identity

The 500M checkpoint is backed up on the Mac in
`artifacts/checkpoints/500m_1142016/`, together with its evidence, frozen source
snapshot, configuration, tokenizer bundle and corpus metadata. The local file
has 1,472,538,167 bytes and SHA256:

`c9eb7c507a874ad7d7db497282121617288f5e5dd50602e9ff78733443d8074e`.

The checksum matches the HPC endpoint record. All 15 frozen training source
hashes, the exact serialized configuration, and tokenizer bundle hashes were
verified. Cloud upload completion is not independently verified.

The 122,702,592-parameter model, token corpus, optimizer/scaler, RNG/sampler
state and 2B learning-rate horizon remain unchanged. Diagnostic helpers and the
new allocation plan are outside the frozen top-level training source identity.

## Scratch deployment and quota

The HPC home quota was over its soft limit with an expired grace period. All
new deployment files, live logs, allocation ledgers and review reports are on
scratch. Existing home files were not removed. An unrelated, busy interactive
allocation `1144498.pbshpc` was left untouched.

- Frozen deployment:
  `/scratch/cse/btech/cs1230342/nanogpt_stage7/deploy_1b_20261112`
- Evidence/log root:
  `/scratch/cse/btech/cs1230342/nanogpt_stage7/continuation_to1b_20261112`
- Training run stays in its original owned directory:
  `/scratch/cse/btech/cs1230342/nanogpt_stage7/train_2b_first100m_1141598`

The deployment's 52 files are hash-pinned in `deployment.json` and made read-only.
Every phase verifies the manifest before training. PBS `sandbox=HOME` avoids
creating another private home directory; the script immediately uses scratch
and writes its live `job.log` there. This behavior follows the
[PBS user guide](https://help.altair.com/2024.1.0/PBS%20Professional/PBSUserGuide2024.1.pdf).
PBS confirmed the requested sandbox and scratch output paths.
The submitted jobs' output paths were then set explicitly with `qalter` to
`logs/phase_<phase>_<job-number>.log`, avoiding directory-only output paths.
The live `continuation_<job-number>/job.log` remains available throughout each job.

## Submitted chain

| Phase | Job | Target updates | Cumulative positions |
|---|---|---:|---:|
| 1 | `1144922.pbshpc` | 38,147 | 625,000,448 |
| 2 | `1144923.pbshpc` | 45,777 | 750,010,368 |
| 3 | `1144924.pbshpc` | 53,406 | 875,003,904 |
| 4 | `1144925.pbshpc` | 61,036 | 1,000,013,824 |

Each allocation requests one V100, two CPUs, 16 GiB RAM and at most 02:30:00.
Their total hard walltime is ten hours. The frozen trainer's existing epoch
argument imposes the earlier deadline, including review/startup and a five-minute
checkpoint margin. All phases use an exclusive run lock. A partial checkpoint
is preserved but causes a nonzero exit, preventing the dependent phase from
starting. Each completed phase exports a new immutable endpoint.

PBS verified phase 1 running on `vsky003` at approximately 11:46 IST on
12 November. Phases 2–4 are held on strict `afterok` dependencies.
Submission and deployment receipts are also saved locally in
`reports/continuation_to1b_20261112/`.

## Reviews and admission checks

Phase 1 reviews the 500M endpoint before optimization, and phase 3 reviews the
750M endpoint before continuing. Each review:

- Strictly resumes the full checkpoint and reproduces its fixed FP32 evaluation
  within CE tolerance 1e-5.
- Scores 512 independent windows / 524,288 positions on train, dev and both dev
  domains, using the same seeds as earlier expanded reviews.
- Generates the same eight prompts with greedy and seeded temperature-0.8,
  top-k-50 decoding, up to 96 new tokens.
- Compares 500M against the 200M expanded review, then 750M against the 500M
  expanded review. Code/config/data/tokenizer identities and window counts,
  positions and seeds must match. Overall and domain dev CE may not regress
  by more than 0.05. Finite-score and dev/train-gap sanity checks also apply.

These checks establish numerical/resume sanity, not factual or assistant quality.
Review and optimization run in separate processes, so decoding cannot advance
the restored training RNG. Six focused tests passed both locally and in HPC's
Python 3.7/PyTorch runtime, including rejection of changed probes and domain
regression. The PBS wrapper passed `bash -n`.

Live phase 1 log:
`/scratch/cse/btech/cs1230342/nanogpt_stage7/continuation_to1b_20261112/continuation_1144922/job.log`.

At allocation walltime 00:00:39, the first job was running and had printed
`CONTINUATION_IDENTITY_BACKUP_AND_BUDGET_PASS`.

The expanded 500M numerical scores have now been observed:

| Stream | CE | Perplexity | Positions |
|---|---:|---:|---:|
| Training | 3.509908749 | 33.44521573 | 524,288 |
| Development | 3.554774559 | 34.97993317 | 524,288 |
| Cosmopedia development | 2.882847243 | 17.86506702 | 524,288 |
| FineWeb-Edu development | 3.716182949 | 41.10718604 | 524,288 |

These are the same expanded windows as the 200M review (dev perplexity
43.45850652), an improvement of approximately 19.51%. They differ from the
small fixed training probes, so 34.98 versus 37.78 is sampling variation rather
than evidence of further optimization at this point.

## Completed review and resumed optimization verified

The full 500M review passed in 145.769 seconds, including exact fixed-probe
reproduction, expanded scores and like-for-like reference comparisons. All
development streams improved relative to the 200M expanded review. The decoded
eight-prompt evidence is saved in `reports/continuation_1144922/completions.json`.
Text remains repetitive and factually unreliable: gravity loops a house analogy,
the fraction explanation gives incorrect procedures, and the spoon explanation
does not explain thermal conductivity. This is not an assistant-quality pass.
Further base pretraining follows the improving validation trajectory.

At phase 1 allocation walltime 00:04:02, optimization had reached update 30,562
from 30,574 attempts: 500,727,808 successful positions and 500,924,416 processed
positions. Latest loss was 3.518370211, gradient norm 1.107210875, LR
0.000264027074875 and AMP scale 65,536. The optimizer step succeeded. The
12-attempt difference was unchanged from the 500M checkpoint, so there were no
new skipped attempts at this check. Phase 1 remained running and phases 2–4
remained held on their dependencies.

This confirms actual continuation; it does not claim that 625M, 750M or 1B has
been reached. The independent interactive job `1144498.pbshpc` remained running
and was not changed.
