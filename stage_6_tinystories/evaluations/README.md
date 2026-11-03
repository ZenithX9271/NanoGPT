# Stage 6A evaluation artifacts

Each milestone directory contains the complete fixed-protocol output triplet:
`evaluation.json`, generated `stories.jsonl`, and the blank `ratings.csv` rubric.

| Directory | Additional positions | Samples | Purpose |
|---|---:|---:|---|
| `25m/` | 25,001,984 | 159 | First retained complete generation set |
| `50m/` | 50,003,968 | 159 | Mid-run comparison |
| `75m/` | 75,005,952 | 159 | Pre-final review gate |
| `100m/` | 100,007,936 | 159 | Final evaluation |

All sets use the same 50 development prefixes, three manual prompts, three seeds,
temperature 0.7, top-k 50, and a 300-token continuation cap. Checkpoint identity
is recorded in each `evaluation.json`; rolling checkpoint pathnames are not stable
identifiers.

`COMPARISON_THROUGH_75M.md` is the detailed controlled comparison. The two
`PROVENANCE_50M*.md` files preserve the original and follow-up +50M audits.
Final +100M metrics and interpretation are in [the results report](../docs/RESULTS.md).
