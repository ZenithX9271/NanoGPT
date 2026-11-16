# Stage 7: 100M checkpoint review and 200M continuation

Authorized by the user on 11 November 2026. PBS job `1141809.pbshpc` ran on
`vsky011`. Completion was verified directly on HPC on 11 November 2026.

## Verified 200M completion

The allocation ledger records exit code 0. The trainer stopped with reason
`review_gate` at 12,208 successful updates from 12,212 attempts:
200,015,872 successful positions and 200,081,408 processed positions.
This is approximately 10% of the unchanged 2B-position learning-rate horizon.
The model still has 122,702,592 parameters; token progress is a separate count.

The same fixed FP32 evaluation probes (16,384 positions per stream) give:

| Stream | 100M CE | 200M CE | 100M perplexity | 200M perplexity |
|---|---:|---:|---:|---:|
| Training | 4.088912398 | 3.780923963 | 59.67495378 | 43.85654483 |
| Development | 4.151478618 | 3.869383603 | 63.52786435 | 47.91284366 |
| Cosmopedia development | 3.472669810 | 3.115160942 | 32.22265608 | 22.53705722 |
| FineWeb-Edu development | 4.209481031 | 3.981902122 | 67.32159290 | 53.61892706 |

Development perplexity fell 24.58% on these probes. The last eight development
CE values decreased at every recorded evaluation. All recorded update losses
were finite. Four optimizer attempts were skipped cumulatively (one in the
first segment and three in this continuation); the final optimizer step
succeeded and the AMP scale was 131,072.

The immutable endpoint `review_u12208_positions_200015872.pt` exists in the
same scratch run directory, with 1,472,538,167 bytes. A fresh SHA256 calculation
matched the saved checkpoint record. The immutable 100M endpoint also remains.
The 200M checkpoint was subsequently downloaded and SHA256-verified on the Mac
before the 500M continuation was submitted. Its source/configuration snapshot,
tokenizer, corpus metadata and training evidence are backed up under
`artifacts/checkpoints/200m_1141809/`.

This allocation consumed 6,606.054879 seconds (about 1h 50m, including review).
The ledger's cumulative GPU-hours upper bound is 3.497795282, leaving about
30.13 minutes within the existing four-hour ceiling. Automatic continuation
was disabled. A further 100M segment would exceed that remaining budget at
the observed rate.

We are at the 200M review point of Stage 7A base-model pretraining. The larger
512-window review and decoded samples below belong to the 100M checkpoint;
they have not yet been repeated at 200M. The next useful work is to back up
the 200M endpoint and repeat those same diagnostics before choosing a longer
continuation and its resource budget.

## Review outcome and verified resume

The review passed in 155.723 seconds. The original fixed evaluation was
reproduced within CE tolerance 1e-5. Expanded FP32 scores, with 524,288 positions
per stream and seeds 20261111 through 20261114, were:

| Stream | CE | Perplexity |
|---|---:|---:|
| Training | 4.039650365 | 56.80647779 |
| Development | 4.055558010 | 57.71736088 |
| Cosmopedia development | 3.477997822 | 32.39479694 |
| FineWeb-Edu development | 4.184664488 | 65.67146365 |

These estimates use different windows from the original 16-window probes.
Their difference is sampling variation, not further optimization of the 100M
checkpoint. The expanded dev/train CE gap is 0.015907645.

All eight completion prompts were decoded using both protocols. Text remains
weak: greedy output repeatedly loops phrases, and sampled explanations include
clear factual errors and topic drift. For example, the evaporation completion
drifts into earthquakes, and the spoon/conduction completion drifts into
toothbrushes. Numerical review PASS therefore does not establish assistant
quality. Further bounded pretraining was continued on the improving trajectory;
retain these same prompts for an honest 100M/200M comparison.

Raw and decoded evidence are retained locally in
`reports/continuation_1141809/review.json` and `completions.json`.

At allocation walltime 00:04:03, PBS reported the job running. The trainer had
reached update 6,142, attempt 6,143, 100,630,528 successful positions, with loss
4.059335947, finite gradient norm 0.747899354, LR 0.000299208374 and AMP scale
131,072. The one-attempt difference remains the original 100M segment's skipped
attempt. This confirms real continuation beyond the original endpoint.

Final handoff check at allocation walltime 00:05:02: job remained running at
update 6,244 / attempt 6,245, with 102,301,696 successful positions, finite loss
3.901139081 and gradient norm 0.676463604. No additional skipped attempt had
occurred. The 200M endpoint had not yet been reached at handoff.

## Verified 100M endpoint and backup

Job `1141598.pbshpc` finished with exit status zero and reason `review_gate`.
It completed 6,104 successful updates from 6,105 attempts, consuming 100,007,936
successful positions and 100,024,320 processed positions. There was one guarded
AMP overflow attempt, at attempt 4,378; all recorded losses were finite.
PBS walltime was 01:39:49. The allocation ledger recorded 5,986.008137 seconds.

Final fixed-sample development CE/perplexity: 4.151478618 / 63.527864355.
Training CE: 4.088912398; Cosmopedia development CE: 3.472669810;
FineWeb-Edu development CE: 4.209481031. Each original sample scored 16,384
positions. Development CE decreased at all 13 recorded evaluation points.

Endpoint checkpoint:
`review_u6104_positions_100007936.pt`, 1,472,538,167 bytes.

SHA256:
`bd9b7ddabfdf75b4551a3a7da4e16a67fecedead3ac32a11e959449b164ee36a`.

A full copy was downloaded to the Mac and independently SHA256-verified under
`stage_7_general_lm/artifacts/checkpoints/100m_1141598/`. It includes training
evidence, the frozen configuration, a hash-verified snapshot of all training
source modules, and the native tokenizer bundle. Production corpus metadata is
also retained; the large token shards are still on HPC scratch. Cloud upload
completion is not independently verified.

The HPC home filesystem was over its 100 GiB soft quota, with a 110 GiB hard
limit and approximately 3 GiB headroom. This is why the full backup was stored
on the Mac instead of adding another large file to HPC home. No unrelated
user files were removed.

## Review and continuation contract

- Keep all top-level training Python sources and the exact configuration
  `configs/production_2b_first100m.json` unchanged; their identities were verified.
- New diagnostic code is only in `hpc/review_segment.py`, outside the frozen
  top-level source identity.
- Use one V100, two CPU cores, 16 GiB host RAM, and a 02:15 PBS hard limit.
- Previous allocation plus this maximum is 14,086.008137 seconds, below the
  cumulative four-GPU-hour ceiling.
- An exclusive run lock prevents concurrent continuation jobs.
- Verify the Mac-backup receipt and the original immutable checkpoint identity.
- Restore with the existing strict `Engine.resume` checks, including optimizer,
  scaler, RNG, sampler, configuration, runtime and source identity.
- Reproduce the original 16-window evaluation within CE tolerance 1e-5.
- Independently score 512 windows / 524,288 positions for each of training,
  overall development, and both development domains, in FP32.
- Decode eight fixed plain-text prompts using greedy decoding and seeded
  temperature-0.8/top-k-50 sampling, up to 96 new tokens each.
- Numerical admission checks require finite scores, expanded dev CE less than
  the original dev CE plus 0.5, and expanded dev/train gap below 0.5. These are
  broad sanity checks, not product-quality targets. Completion text remains
  qualitative diagnostic evidence.
- Review and training run in separate processes so diagnostics cannot advance
  the restored training RNG or sampler.
- Continue the same 2B-position LR horizon to cumulative update 12,208,
  corresponding to 200,015,872 successful positions.
- Include review/startup in this job's earlier deadline using the existing
  trainer epoch argument, without changing the serialized configuration.
- Preserve the 100M endpoint and publish an exclusive 200M endpoint link,
  checkpoint digest, metrics, summary, and allocation ledger. No auto-continuation.

HPC run:
`/scratch/cse/btech/cs1230342/nanogpt_stage7/train_2b_first100m_1141598`

New evidence: `stage_7_general_lm/reports/continuation_1141809/`.
PBS output: `nanogpt_s7next.o1141809` at the HPC repository root.

## Local verification

Four review-tool tests passed: remainder-batch token accounting and independent
sampling; cached versus full-prefix sampled generation; exclusion from frozen
source identity; and endpoint hard-link preservation across rolling replacement.
The PBS script passed `bash -n`; uploaded script SHA256 values matched locally.
