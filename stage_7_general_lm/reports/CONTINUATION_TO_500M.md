# Stage 7A: authorized continuation from 200M to 500M tokens

## Final 500M result verified — 11 November 2026

Both continuation phases completed with exit code 0. The final checkpoint is
`review_u30518_positions_500006912.pt`, with 1,472,538,167 bytes and SHA256
`c9eb7c507a874ad7d7db497282121617288f5e5dd50602e9ff78733443d8074e`.
A fresh SHA256 calculation on HPC matched the checkpoint record. All frozen
top-level source hashes matched the phase 2 launch record. The 100M, 200M,
350M and 500M immutable endpoints were present.

Final counters: 30,518 successful optimizer updates from 30,530 attempts;
500,006,912 successful positions and 500,203,520 processed positions. All
recorded update losses were finite. Twelve attempts were skipped cumulatively
by AMP overflow handling (about 0.04%); the final update succeeded.

| Fixed FP32 evaluation | 100M | 200M | 350M | 500M |
|---|---:|---:|---:|---:|
| Training CE | 4.0889 | 3.7809 | 3.6186 | 3.5271 |
| Development CE | 4.1515 | 3.8694 | 3.7138 | 3.6317 |
| Development perplexity | 63.5279 | 47.9128 | 41.0096 | 37.7756 |
| Cosmopedia development perplexity | 32.2227 | 22.5371 | 18.9109 | 17.0976 |
| FineWeb-Edu development perplexity | 67.3216 | 53.6189 | 46.0367 | 42.9445 |

Each stream uses the same seeded 16,384 evaluation positions at every update.
Development perplexity fell 21.16% from 200M and 40.54% from 100M. Development
CE fell 6.14% from 200M and 12.52% from 100M. Domain perplexity reductions
from 200M were 24.14% for Cosmopedia and 19.91% for FineWeb-Edu.

Gains continue but slow: development perplexity fell 7.89% during 350M→500M,
compared with 14.40% during 200M→350M. The final dev/train CE gap is 0.10456.
These probes do not show sustained validation divergence, but their small
sample and repeated use cannot rule out overfitting or establish task accuracy.

The two continuation allocations consumed 5.41059 GPU-hours, within the six
additional hours authorized. All recorded training allocations total 8.90838
GPU-hours. The 2B learning-rate schedule remains unchanged; 500M is a review
stop at approximately 25% of that horizon.

Final metrics, summary, checkpoint metadata and allocation ledger are saved
locally in `reports/continuation_1142016/`, with derived `analysis.json` and
`loss_curve_100m_500m.png`. On 12 November, the full 500M checkpoint was
downloaded and SHA256-verified on the Mac under
`artifacts/checkpoints/500m_1142016/`. Its source/config/tokenizer identity was
also verified. The larger 512-window review and generation probes were then
repeated at 500M in job `1144922.pbshpc`. See
[the 1B continuation record](CONTINUATION_TO_1B.md) for the new review and launch.

Recommended next work: back up 500M, repeat the existing larger review and
generation prompts, then consider continuation to 1B with a 750M review point.
At the measured phase 2 rate, another 500M positions requires approximately
8.6 V100-hours before additional queue delays/review overhead; allow 9–10 hours.
Training beyond the present authorized target has not been submitted.

## Original authorization and execution record

User authorization: 11 November 2026, "ok, lets start training over 500M tokens,
lets do the tasks". The agreed interpretation is **500M cumulative successful
training positions**, not another 500M beyond the existing 200M. The allowance
for this continuation is at most six additional V100 allocation hours.

The same 122,702,592-parameter model, 363,787,512-token training corpus,
tokenizer, optimizer, scaler, sampler/RNG state, and 2B-position learning-rate
horizon are retained. Top-level training sources and the serialized original
configuration remain unchanged. The new approval is enforced by the PBS
wrapper, using two allocations of at most three hours each.

| Phase | Starting update | Ending update | Ending successful positions |
|---|---:|---:|---:|
| 1 | 12,208 | 21,363 | 350,011,392 |
| 2 | 21,363 | 30,518 | 500,006,912 |

Phase 1 repeats the 512-window FP32 review and the same eight generation
prompts at the immutable 200M checkpoint before resuming. Review and training
are separate processes, so review does not alter the resumed training state.
Both phases use the existing five-minute checkpoint margin before PBS walltime.

Phase 2 must be submitted with an `afterok` dependency on phase 1. It verifies
the phase 1 checkpoint and exit ledger before resuming. If a phase stops early,
its partial endpoint is preserved and the job exits nonzero, preventing the
dependent phase from running. No automatic continuation beyond 500M is allowed.

The immutable 100M and 200M endpoints are preserved. Before submission, the
200M checkpoint must have a local Mac backup with its hash matched to the HPC
record. The original training source snapshot, configuration, tokenizer, and
corpus metadata accompany that backup. Cloud upload is not independently
verified.

Wrapper: `hpc/continue_to_500m.pbs`. Diagnostic helper: `hpc/review_segment.py`.
Live training metrics remain in the original scratch run directory:
`/scratch/cse/btech/cs1230342/nanogpt_stage7/train_2b_first100m_1141598`.
Each allocation writes metadata under `reports/continuation_<job-number>/`.

## Submission verified

- Phase 1: `1142015.pbshpc`, observed running on `vsky039` at approximately
  10:04 IST on 11 November 2026.
- Phase 2: `1142016.pbshpc`, held with `afterok:1142015.pbshpc` as intended.
- PBS confirmed one GPU, two CPU cores, 16 GiB host RAM, and a three-hour
  hard walltime for each job.
- Five checkpoint-review tests passed locally and on HPC. The wrapper passed
  `bash -n`; uploaded wrapper/helper SHA256 values matched local files.
- The 200M checkpoint Mac backup SHA256 matches
  `2af4534223727988332bb1ca47dfdafeb97601d088005c4cd7caf311588579ef`.
  Backup directory: `artifacts/checkpoints/200m_1141809/`.
- All 15 top-level training module hashes still match the prior launch record.

Submission receipt: `reports/continuation_to500m_20261111_submission.json`.
Phase 1 PBS log: `nanogpt_s7to500m.o1142015` at the HPC repository root.
Phase 2 PBS log: `nanogpt_s7to500m.o1142016` at the same location.

## 200M review verified

The 200M numerical review passed and reproduced the original fixed evaluation.
On the same 512-window expanded samples as the 100M review:

| Stream | 100M CE | 200M CE | 100M perplexity | 200M perplexity |
|---|---:|---:|---:|---:|
| Training | 4.039650365 | 3.748754540 | 56.80647779 | 42.46815668 |
| Development | 4.055558010 | 3.771806610 | 57.71736088 | 43.45850652 |
| Cosmopedia development | 3.477997822 | 3.137410618 | 32.39479694 | 23.04411952 |
| FineWeb-Edu development | 4.184664488 | 3.923397395 | 65.67146365 | 50.57196619 |

Each stream scored 524,288 positions in FP32. The dev/train CE gap is about
0.02305. This is numerical evidence of improved language modeling, not proof
of factual or instruction-following quality. Raw review evidence is retained
under `reports/continuation_1142015/`.

These are batch jobs and do not depend on keeping the Mac or SSH connection
running.

The review completed in 159.690 seconds. Decoded generation evidence is saved
in `reports/continuation_1142015/completions.json`. Greedy samples still loop
phrases and produce factual errors (for example, the spoon explanation
incorrectly calls a metal spoon plastic). The numerical improvement does not
yet establish reliable explanatory text. These same probes should be compared
again at 500M.

## Optimization beyond 200M confirmed

At allocation walltime 00:04:06, phase 1 remained running and had reached:

- 12,240 successful updates / 12,244 attempts.
- 200,540,160 successful positions / 200,605,696 processed positions.
- Latest training loss 4.019304693, gradient norm 0.867710948.
- Learning rate 0.000295184398, AMP scale 131,072.
- The latest optimizer step succeeded. The four-attempt difference remains
  the previously recorded skipped attempts; none had been added at this check.

The successful strict resume and new optimization steps confirm actual
continuation. This report does not claim that 350M or 500M has been reached.
