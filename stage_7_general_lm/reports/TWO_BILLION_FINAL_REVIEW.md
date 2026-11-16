# Stage 7A: completed 2B base pretraining

Verified from HPC on 16 November 2026. The original training horizon is complete:
122,071 successful updates / 2,000,011,264 successful positions. The model has
122,702,592 parameters; 2B refers to training positions, not model parameters or
distinct corpus tokens. The final summary reason is `completed` and final LR
is the original schedule's minimum, 0.00003. No continuation beyond 2B is
scheduled by this workflow.

## Completion and checkpoint

All four final-segment allocation records have exit code zero. Final job
`1150579.pbshpc` ran on `vsky004`; PBS reports exit zero and walltime 02:14:58.
Its recorded start was 20:55:32 IST on 15 November, giving completion at
approximately 23:10:30 IST that night. The final 500M-position segment used
8.696200771 GPU-hours in its allocation ledgers, within the ten-hour hard limit.

The final endpoint is:

`/scratch/cse/btech/cs1230342/nanogpt_stage7/train_2b_first100m_1141598/review_u122071_positions_2000011264.pt`

It has 1,472,538,359 bytes and recorded SHA256:

`f32b6a493d3aab92c5f3d9a5ff482f856f5a21dccfaa38780c4ba7a1e61a6791`

The final numerical review strictly reloaded this endpoint, verified its counters
and finite parameters, reproduced its fixed FP32 evaluation and passed matched
development checks against 1.5B. All logged update losses are finite. There were
46 skipped AMP attempts out of 122,117 attempts (0.0377%); the final optimizer
step succeeded. The reported final gradient norm, 2.03386, is before clipping.
There were 2,000,764,928 processed positions including skipped attempts.

The frozen model code, original serialized config and 2B horizon, corpus,
tokenizer and full optimizer/scaler/RNG/sampler resume identities were preserved.
The separately pinned diagnostic helper handles original-horizon completion
and final-only heldout scoring. The frozen deployment and Stage 6 are unchanged.

## Larger development evaluation

Each value uses the same seeded 512 windows / 524,288 positions per stream,
context 1024, FP32. Comparison code checked the seeds, counts and identities.

| Stream | 1B PPL | 1.5B PPL | 2B PPL | Reduction from 1.5B |
|---|---:|---:|---:|---:|
| Train | 28.11845 | 23.98103 | 21.93068 | 8.55% |
| Dev | 29.90327 | 26.02356 | 24.16603 | 7.14% |
| Cosmopedia dev | 14.79748 | 12.61377 | 11.58518 | 8.15% |
| FineWeb-Edu dev | 35.56641 | 31.13962 | 29.00770 | 6.85% |

All development streams improved. This is evidence of improved token prediction,
not a percentage improvement in factual accuracy or instruction following.

The final small fixed probes use different windows and must be compared
separately. Their dev perplexity is 25.66329 versus 27.80846 at 1.5B. Their
train/Cosmopedia-dev/FineWeb-Edu-dev perplexities are 20.85157 / 11.36688 / 30.93120.
The figure [evaluation_500m_2b.png](review_2b_20261116/evaluation_500m_2b.png) and
its PDF display the fixed and larger evaluations in separate panels.

## Reserved heldout evaluation

The final-only heldout sample has CE 3.184979221 and perplexity 24.16678624,
using 512 windows / 524,288 positions at seed 20261115. This is a sample of the
reserved heldout stream, not an evaluation of every token in that split.
It is reported separately from the four matched train/dev streams and was not
used to admit continuation between milestones. Its agreement with development
is encouraging on these samples; it does not certify factual correctness or
rule out all generalization problems.

## Generated text remains a material limitation

The same eight prompts and greedy/seeded sampled settings were inspected.
Greedy outputs still loop. The fraction prompt provides no valid comparison
method; the metal-spoon prompt says it feels colder because it is metal rather
than explaining heat transfer. The water-cycle prompt confuses evaporation,
condensation and precipitation. Sampled outputs also contain scientific errors,
invented quantities and topic drift. Story continuations can have dialogue and
plausible local phrasing but lose consistency. The model is not a reliable
factual assistant at this endpoint.

Full matched 1.5B/2B generations are in
`review_2b_20261116/generation_comparison_1p5b_2b.md`. Eight qualitative probes
are not an accuracy benchmark. The numerical PASS is not a generation-quality
certification.

## Evidence and final preservation

Downloaded checkpoint, review, completions and allocation JSON hashes match
the HPC copies. Matched comparisons and verified final metrics are saved in
`review_2b_20261116/analysis.json`. Raw evidence is under
`continuation_to2b_20261115/continuation_1150579/`.

The final Mac backup is complete at `artifacts/checkpoints/2b_1150579/`.
It includes the exact frozen deployment, config, tokenizer, audited corpus
metadata, full checkpoint, final metrics and reviews, matched generations, plots
and the new diagnostic helper. The local 1,472,538,359-byte checkpoint matches
the HPC SHA256 above. All frozen deployment and diagnostic hashes were checked.
`mac_backup_verified.json` certifies the local copy; `backup_manifest.json` lists
file sizes and hashes. Cloud synchronization completion is not independently
verified.

The completed artifact is a base language model. Further training, instruction
tuning, data changes and compression require a separate concrete plan; none
have been launched by this status check.
