# Stage 6A final status — 3 November 2026

Stage 6A TinyStories continued pretraining is **complete through its configured
+100M additional-position budget**, including final scoring and 159 generated
continuations. This status supersedes completion claims in the November 1
[research report through +50M](RESEARCH_REPORT_THROUGH_50M.md), which remains unchanged
as historical evidence through +50M.

This document describes the existing local artifacts and saved HPC records. It
does not establish live remote filesystem state or authorize another job.

## Final artifact and accounting

Official fixed endpoint:

`stage_6_tinystories/runs/s6a/review_u12208_additional_100007936.pt`

SHA256:

```text
da3d6c48a6292cbb75818a894d11a40f651c45aa9a90455d15e8ed957d5424a4
```

Fresh local inspection on November 3 confirmed the same SHA256 for the immutable
endpoint and `runs/s6a/latest.pt`. The byte-identical +100M snapshot copy was later
removed from the active tree during repository consolidation.
The final checkpoint records:

| Quantity | Value |
|---|---:|
| Stage 6 completed updates | 12,208 |
| Additional processed positions | 100,007,936 |
| Total completed updates, including Stage 5 | 24,208 |
| Total processed positions, including Stage 5 | 198,311,936 |
| Fixed-window train CE / perplexity | 1.9976016469 / 7.3713557679 |
| Fixed-window development CE / perplexity | 2.0486899521 / 7.7577314468 |

The small overshoot follows whole-batch budgeting: ceil(100,000,000 / 8192).
Training positions are sampled positions, not a count of distinct corpus tokens.

Preservation receipt: `runs/s6a/review_u12208_additional_100007936.json`.
Its SHA256 is `58da233693d6983eb83a9da5e3ec8b5debf57ad9d8336337d2e82ff1fe5b077d`.
The matching training log is `hpc/logs/s6a_100m_1130192.pbshpc.log`;
it records `reason=completed`, `REVIEW_100M_PRESERVATION_PASS` and a finish marker.
The pre-run archive identity was captured during the audit and the redundant local
copy was later removed from the active tree during repository consolidation.
Raw final PBS accounting was not independently recovered in the local audit;
the receipt is a validation record, not a substitute for scheduler exit evidence.

## Milestones and complete evaluations

| Milestone | Stage 6 updates | Additional positions | Dev CE | Dev perplexity | Complete evaluation directory |
|---|---:|---:|---:|---:|---|
| +10M | 1,221 | 10,002,432 | 2.159101 | 8.663345 | No matched full story set identified |
| +25M | 3,052 | 25,001,984 | 2.133726 | 8.446280 | `evaluations/25m/` |
| +50M | 6,104 | 50,003,968 | 2.094938 | 8.124939 | `evaluations/50m/` |
| +75M | 9,156 | 75,005,952 | 2.065159 | 7.886555 | `evaluations/75m/` |
| +100M | 12,208 | 100,007,936 | 2.048690 | 7.757731 | `evaluations/100m/` |

Scores are sampled fixed-window CE, not complete-corpus CE. Sources are
`runs/s6a/metrics.jsonl`, checkpoint metadata and each evaluation's `evaluation.json`.
The retained +25M directory is the single canonical copy of that story set; a
duplicate and an incomplete generation attempt were removed from the active tree.

The +75M immutable review is `runs/s6a/review_u9156_additional_75005952.pt`, SHA256
`7bcad3a13ce5e056b11af0617e6890db072a7b4f5eaafb5df9fdc8b47f5e9105`.
The +50M native snapshot retains SHA256
`40d145f6739896f923438e54f5a3235a26bfdb0d5cc136bfafb816871658d04e`.
Historical evaluation paths can name rolling files: their recorded SHA256 is the
identity to follow, rather than the current contents of that mutable pathname.
The exact +25M evaluated best hash `d70946e39078e9474cae34742c8ad7d96d0ba6315a2b849b75aed12dbd742315`
is not among the retained checkpoints; the native +25M snapshot remains preserved.

## Final generation and interpretation

`evaluations/100m/evaluation.json` records:

| Diagnostic | Value |
|---|---:|
| Generated / expected samples | 159 / 159 |
| Generation complete | true |
| EOS fraction | 0.9433962264 (150/159) |
| Mean continuation length, excluding EOS | 146.3584905660 |
| Mean repeated token 4-gram fraction | 0.0335092746 |

Fifty development prefixes and three manual probes use generation seeds
1337/2027/3407, temperature 0.7, top-k 50 and a 300-token continuation cap.
The +100M evaluation log is `hpc/logs/1130319.pbshpc.log`.
Human-rating fields remain blank. Lower development CE establishes better measured
next-token prediction, not a general improvement in narrative coherence. Identity
drift, object-state errors, semantic loops and weak causal links remain material
limitations. Compared with +75M, final EOS/repetition diagnostics are worse; these
diagnostics also do not substitute for human ratings.

## Training source versus evaluation workaround

All six current root Stage 6 Python source hashes match the final checkpoint's
`code_identity`. The unchanged architecture has 6,897,152 parameters, six layers,
eight attention heads, width/context 256, dropout 0.2 and tied embeddings.

| Role | File | SHA256 |
|---|---|---|
| Checkpoint-compatible source | `model.py` | `7aca47adbae88c80bdc9d0339dc662761d179f51e5e274715dc9ae00e954cbb5` |
| Temporary evaluation source | `recovery/generation_compat_model.py` | `b058a3621e93103100e017742bc15a08cc9a586f750b59df5b93fdc1a42417d3` |

The workaround uses separate PAD/BOS logit assignments for the old CUDA runtime;
the evaluation wrappers apply it to a temporary source copy. Do not replace the
active training model to satisfy a historical transfer inventory.

Data remain `data/shards/v2/` and `data/tokenizer/v2/` at repository root.
The manifest, tokenizer and prompt hashes in the final checkpoint agree with
the verified assets. Original LR horizon, source/config identity guards and
optimizer/RNG continuation semantics remain unchanged.

## Transfer-ledger correction and evidence scope

An earlier root transfer inventory expected patched `model.py` (5375 bytes), while
the correct active training file is 5335 bytes. The canonical controls now live in
`deployment/`; see [the transfer guide](TRANSFER.md).

The refreshed transfer scope contains runtime/source/data assets and this status,
not trained Stage 6 checkpoints, evaluation results, PBS logs or pre-run archives.
The single source-only compatibility file is explicitly included. This is not a
complete experiment backup and does not authorize transferring large checkpoints.

The deployment ledger proves consistency of its selected source/data payload; it
is not an experiment backup. Checkpoints, evaluations and scheduler logs remain
separate evidence with their own identities.
