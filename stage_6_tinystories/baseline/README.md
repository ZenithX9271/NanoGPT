The PBS runner uses `stage5_best.pt` here, copied from the verified local
`stage_5_tinystories/checkpoints/best.pt`. Checkpoints are ignored by Git. This
separate HPC copy avoids replacing an existing cluster Stage 5 checkpoint.

Expected SHA256:
`2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce`.

The complete prepared project physically includes both verified copies; transfer
both through the `../deployment/files.txt` list. The old `best.pt` in this
folder is a different pilot artifact (step 1,220); it is preserved locally and
excluded from the transfer. Never rename or select it as the Stage 5 baseline.

Direct local CLI commands default to the original Stage 5 checkpoint path in the
configuration. Both locations must contain the same pinned bytes. Resume uses
Stage 6's `latest.pt` and does not need to reopen this baseline file.
