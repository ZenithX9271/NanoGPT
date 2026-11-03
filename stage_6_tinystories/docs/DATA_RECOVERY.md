# Stage 6 missing-file investigation

Date: 30 October 2026. This report describes the local repository, not a live HPC
filesystem inspection. No HPC changes or job submissions were made.

## Findings and repair plan

| Item | Finding | Resolution |
| --- | --- | --- |
| `stage_6_tinystories/data/tinystories/v2/` | This is not a path read by the current code | Use the existing genuine root `data/tinystories/v2/`; no duplicate corpus |
| `stage_6_tinystories/data/shards/v2/` | This is not the configured shard directory | Use existing root `data/shards/v2/`; all 41 shards included in transfer |
| Root train/dev/test JSONL | Found, with intended story counts and provenance | Preserve byte-for-byte and include in transfer |
| Required `baseline/stage5_best.pt` | Missing before this repair | Copy the exact original Stage 5 best and verify the pinned SHA256 |
| Existing `baseline/best.pt` | Wrong artifact: pilot step 1,220, 9,994,240 positions, CE 3.353233 | Preserve locally; exclude from transfer and never select |
| Two raw files | Existing symlinks led to files outside the repository | Verify genuine source hashes and replace links with exact physical local files; external originals preserved |
| Tokenizer/manifest/prompts/offsets | Found | Include and verify exact identity, structure and paths |
| Old PBS Haswell selector | Inconsistent with reported tested V100/Skylake execution | Restore `centos=skylake`, retain resource counts/module and check GPU is V100 |
| Partial transfers | Old manual/ignore-existing steps could omit or retain stale artifacts | Exact file list, sizes and SHA256 ledger; inline missing-file failure |
| `runs/s6a/latest.pt` | No local quality-training latest found; remote existence unverified | Generated runtime output, not an initial dependency. Fresh start unless a genuine same-run checkpoint is established |

## Final status

All required files are physically present and checksum-verified. No required
dataset, shard, tokenizer, checkpoint, configuration, script or manifest remains
missing. Both raw-source links were materialized; no repository symlink remains
outside Git metadata. The clean physical transfer copy passed checks and real
CPU model/evaluation/checkpoint roundtrip with reads of the original repo blocked.

No raw dataset or prepared split needs to be downloaded/regenerated. The existing
genuine source files, original tokenizer and original shards were located. No
required runtime file is intentionally supplied as a symlink or dummy substitute.

## Preserved but excluded artifacts

The wrong pilot copy remains at `stage_6_tinystories/baseline/best.pt`, SHA256
`fa8b534b40c372ab4502b831d841c36fde8fcb70e4b5bb6210e5944eff1d0197`.
It is not renamed to the expected baseline. All Stage 1–5 implementations,
documentation and historical results remain local and unchanged.

Previous GPU jobs 1125951/1125955/1125957 were validation-only. They do not establish
that a Stage 6 quality-training run started or produced a resumable latest.
