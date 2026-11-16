# Stage 7A bounded production-corpus plan

**Prepared:** 7 November 2026, Asia/Kolkata  
**Status:** implementation validated locally; source download not yet run

## Decision

The first quality-training corpus candidate is capped at 500,000,000 selected
tokens under the frozen 49,152-entry tokenizer:

| Source | Pinned file | Selected-token cap | Intended share |
|---|---|---:|---:|
| FineWeb-Edu | `sample/10BT/000_00000.parquet` | 400,000,000 | 80% |
| Cosmopedia v2 | `cosmopedia-v2/train-00030-of-00104.parquet` | 100,000,000 | 20% |

The source files and SHA-256 values are frozen in
`configs/production_sources_v1.json`. Their combined network transfer must stay
within 4 GiB. Raw files, extracted JSONL, token streams, and wheel environments
belong under quota-checked HPC scratch, never the cloud-synced repository.

This corpus is intended for the initial 100M-position review gate on an
unchanged longer LR horizon. Approximately 80% of selected tokens should land in
the training split under stable source-group hashing, leaving enough unique
training text to avoid intentional corpus recycling at the review gate. Actual
split counts, not this expectation, decide admission.

## Runtime separation

The IIT Delhi environments cannot safely be merged:

1. `apps/anaconda/3` provides Python 3.6 and PyArrow 0.12.1. It runs
   `extract_parquet.py` only.
2. The isolated scratch Python 3.9 environment provides NumPy 1.26.4 and
   Tokenizers 0.22.2. It runs `pack_production.py` only. Its base Python has a
   broken `_ssl` linkage, so it must remain offline.
3. `apps/pytorch/1.10.0/gpu/intelpython3.7` remains the independent, measured
   V100 training environment. Corpus preparation packages must not be injected
   into it.

## Data contract

Extraction verifies each complete Parquet file before reading, applies NFC and
trim normalization, retains compact provenance, groups FineWeb by URL and
Cosmopedia by prompt where present, and stops at configured document, text,
output-byte, or walltime limits.

Packing then:

- verifies the tokenizer JSON hash and vocabulary;
- performs cross-source exact normalized-text deduplication;
- assigns entire source groups to stable 80/10/10 train/dev/heldout splits;
- enforces per-source selected-token caps;
- writes little-endian token streams and hashes every shard;
- writes a compressed, text-free document/provenance ledger;
- records deterministic hashed group indexes for split-disjointness audits; and
- publishes `COMPLETE.json` only after a successful manifest is written.

The known limitations are explicit: exact deduplication does not detect semantic
paraphrases, Cosmopedia derivative lineage is incomplete, and two selected
files are bounded convenience samples rather than uniform full-dataset samples.

## Admission checks before GPU training

1. Source files match the frozen SHA-256 values.
2. Extraction and packing complete on a CPU PBS allocation, not the login node.
3. Packed train tokens exceed the 100M review exposure with a documented margin;
   dev and heldout each exceed one 1,024-token window per source.
4. Every token shard matches its manifest size and SHA-256.
5. The pinned native tokenizer bundle passes its lock on HPC.
6. A short V100 benchmark is repeated because adding production utilities
   changes the repository-wide source identity used by checkpoint guards.
7. Exact uninterrupted-versus-resumed FP16 next-update equivalence passes.
8. The final 2B-horizon configuration and positive resource limits are frozen;
   only then may `main_run_approved` become true.
