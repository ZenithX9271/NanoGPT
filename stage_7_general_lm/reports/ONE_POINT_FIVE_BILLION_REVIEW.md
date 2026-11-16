# Stage 7A: verified 1.5B review

**Subsequent status (15 November):** the user approved the final segment to 2B.
Four jobs are submitted; see [the continuation record](CONTINUATION_TO_2B.md).
The results below describe the preserved 1.5B checkpoint.

Verified from HPC on 15 November 2026. The four continuation jobs
`1147658.pbshpc` through `1147661.pbshpc` completed with exit code zero. Training
stopped at the requested boundary: 91,553 successful updates / 1,500,004,352
successful positions. No continuation beyond 1.5B is running or submitted by
this workflow. The model still has 122,702,592 parameters.

## Training and restart integrity

The final checkpoint is `review_u91553_positions_1500004352.pt`, 1,472,538,359
bytes, with recorded SHA256:

`d641a56858874f563a55e84b8084aa199f6bef3af9e210d36957f7c72d5da642`

The automatic review strictly restored the checkpoint, checked counters and
finite parameters, and reproduced its fixed FP32 evaluation within CE tolerance
1e-5. Its numerical status is PASS. The model, frozen source, original 2B learning
rate horizon, serialized configuration, corpus and tokenizer identities match
the 1B review. Model parameters, optimizer, scaler and RNG/sampler state were
preserved throughout continuation.

All logged update losses are finite. There were 35 skipped AMP attempts out of
91,588 attempts (0.0382%). The last optimizer step succeeded. Final LR was
0.00007079881504, loss 3.141850293 and gradient norm 1.814223409. Individual batch
losses are not the basis for the cross-checkpoint quality comparison.

The phase allocation ledgers total 8.859494173 GPU-hours, within the ten-hour
ceiling. The user-provided PBS walltimes sum to 8h 51m 47s. The final review
itself took 151.660 seconds within the last allocation. There are no current
jobs listed at the authenticated verification check.

## Larger evaluation: matched windows and seeds

Each stream has 512 windows / 524,288 positions, context 1024, evaluated in FP32.
The same independent window seeds, source/config/data/tokenizer identities and
decoding settings were checked against earlier review files.

| Stream | 500M PPL | 1B PPL | 1.5B PPL | Reduction from 1B |
|---|---:|---:|---:|---:|
| Train | 33.44522 | 28.11845 | 23.98103 | 14.71% |
| Dev | 34.97993 | 29.90327 | 26.02356 | 12.97% |
| Cosmopedia dev | 17.86507 | 14.79748 | 12.61377 | 14.76% |
| FineWeb-Edu dev | 41.10719 | 35.56641 | 31.13962 | 12.45% |

Overall larger-review dev perplexity fell 25.60% from 500M. The 1.5B train/dev
CE values are 3.177263010 and 3.259002157, a gap of 0.081739148 nats/token.
All development streams improve against the 1B reference. This supports further
pretraining but does not establish factual accuracy or downstream task quality.

The smaller fixed training probes use different windows: their dev perplexity
fell from 31.67967 at 1B to 27.80846 at 1.5B (12.22%). These scores must not be
mixed with the larger-review scores above. The two sets tell the same direction
of improvement with different sampling variation.

The figure [evaluation_500m_1p5b.png](review_1p5b_20261115/evaluation_500m_1p5b.png)
and its PDF show both evaluation sets separately. Exact comparisons are saved in
`review_1p5b_20261115/analysis.json`.

## Generated text: some better openings, major weaknesses remain

The same eight prompts were generated using greedy decoding and seeded
temperature 0.8 / top-k 50, up to 96 new tokens. These are qualitative diagnostics,
not an accuracy benchmark.

At 1.5B, the greedy photosynthesis continuation correctly opens with absorption
of carbon dioxide and release of oxygen, an improvement over its 1B opening.
It then drifts into unrelated explanations. The sampled version incorrectly calls
glucose a kind of grain and gives confused chemistry. The gravity prompt starts
with a plausible ball-on-a-slope image, then repeats. The explanation prompt's
sampled continuation has more structure, but remains circular and awkward.

The fraction prompt still gives no valid comparison procedure; sampled decoding
suggests subtracting zero. The spoon prompt fails to explain thermal conductivity
in both modes and invents terminology. Repetition and topic drift remain common.
The model is not ready to provide reliable factual answers or function as a
finished assistant. A numerical PASS is not a generation-quality certification.

Full matched outputs are saved in
`review_1p5b_20261115/generation_comparison_1b_1p5b.md`.

## Mac backup

The destination is `artifacts/checkpoints/1p5b_1147661/`. It includes the frozen
deployment, original config, tokenizer, audited corpus metadata, final training
evidence and review, and the 1B reference review. All 52 copied deployment hashes
and the new review's source/config/corpus identities have been verified. The
full local checkpoint now matches the endpoint's 1,472,538,359 bytes and SHA256.
`mac_backup_verified.json` certifies the local copy; `backup_manifest.json` lists
file sizes and hashes. The matched generations, analysis and plots are included
in the backup. Cloud synchronization completion is not independently verified.

Downloaded checkpoint/review/completion/allocation/launch/summary JSON hashes
match their HPC copies. Verification receipts are in `review_1p5b_20261115/`.

## Next decision

Completing the original 2B horizon is a reasonable next base-pretraining segment
given the continued improvement across both development domains. This would add
500,006,912 successful positions, stop at update 122,071 / 2,000,011,264 positions,
and take approximately nine V100 GPU-hours at the measured rate, plus queue
time. The existing 2B LR horizon should be preserved. The review-and-backup task submitted no additional GPU jobs; the subsequent
user-authorized 2B chain is recorded in CONTINUATION_TO_2B.md.

The training corpus contains 363,787,512 positions. Nominal sampled volume is
4.12 times that size at 1.5B and 5.50 times at 2B; the sampler draws windows with
replacement, so these are volume equivalents, not literal complete epochs or
unique tokens. Research on repeated data finds diminishing value with increasing
repetition ([Muennighoff et al.](https://arxiv.org/abs/2305.16264)). Our measured
development improvements justify considering the final segment, while these
results do not justify assuming indefinite gains from additional repetition.
After the 2B endpoint, use the reserved heldout evaluation and generation review
before deciding on fresh-data training or instruction tuning.
