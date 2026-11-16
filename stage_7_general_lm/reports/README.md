# Stage 7 evidence index

Current outcome: the 122.7M-parameter base model completed its original 2B
training-position horizon. Start with the
[comprehensive audit](../docs/STAGE_7A_COMPREHENSIVE_AUDIT.md),
[final review](TWO_BILLION_FINAL_REVIEW.md), and
[matched evaluation history](comprehensive_audit_20261116/matched_evaluation_history.json).
Numerical PASS results do not certify factual or study-assistant accuracy.

| Evidence | Location |
|---|---|
| Production corpus manifest, completion marker and audit | [production_corpus_v1](production_corpus_v1/) |
| Original production GPU restart/throughput proof | [original_gpu_gate](comprehensive_audit_20261116/original_gpu_gate/) |
| First 100M checkpoint/allocation/launch records | [run_1141598](run_1141598/) |
| 200M continuation and review of its 100M parent | [continuation_1141809](continuation_1141809/) |
| 350M continuation and review of its 200M parent | [continuation_1142015](continuation_1142015/) |
| 500M endpoint | [continuation_1142016](continuation_1142016/) |
| Continuations to 1B | [continuation_to1b_20261112](continuation_to1b_20261112/) |
| Continuations to 1.5B | [continuation_to1p5b_20261113](continuation_to1p5b_20261113/) |
| Continuations to 2B | [continuation_to2b_20261115](continuation_to2b_20261115/) |
| Complete training metrics, including all earlier updates | [final metrics.jsonl](continuation_to2b_20261115/continuation_1150579/metrics.jsonl) |
| Final matched generations and plots | [review_2b_20261116](review_2b_20261116/) |
| Tests, artifact hashes, continuity and CPU inference checks | [comprehensive_audit_20261116](comprehensive_audit_20261116/) |

Intermediate metrics files are cumulative copies. Git retains the final complete
log rather than all repeated prefixes; local copies remain available outside Git.
Small checkpoint JSON files identify external `.pt` artifacts and do not contain
model weights. The production corpus bins, raw Parquet inputs and checkpoint
backup bundles are also external. Audit scripts that load or rehash those files
require the documented local/HPC artifacts; the source unit tests use temporary
fixtures and the included pinned tokenizer instead.

Historical plans, scheduler snapshots, resource probes and pre-launch reports
retain their original status and dates. They are provenance, not current queue
state or instructions to relaunch completed training. Paths and package versions
record the original environment and may need adjustment on another machine.
