# Stage 6A deployment bundle

The canonical transfer controls are under `stage_6_tinystories/deployment/`:

- `files.txt` — sorted repository-relative rsync file list.
- `inventory.json` — byte size, SHA256, and category for each payload file.
- `sha256sums.txt` — checksum ledger for the payload and control files.

This is a reproducible source/runtime/data deployment bundle, not an experiment
backup. It excludes Stage 6 checkpoints, run metrics, evaluations, PBS logs,
caches, and local cleanup backups. It includes both verified Stage 5 baseline
locations because the PBS runner uses the Stage 6 copy.

## Verify locally or after transfer

From the repository root:

```bash
python3 -B stage_6_tinystories/tools/verify_transfer.py
```

Use `--quick` to check all sizes while hashing only critical/small files. Use
`--deep-data` for shard ranges, offsets, split isolation, and sampled re-encoding;
that mode requires NumPy and optionally uses `tokenizers` when available.

To transfer exactly the verified bundle:

```bash
rsync -avc --files-from=stage_6_tinystories/deployment/files.txt ./ \
  USER@HOST:~/zenithX9271/NanoGPT/
```

Do not add `--delete` or `--ignore-existing`. The former can remove remote work;
the latter can silently retain stale files.

## Regenerate after an intentional source change

```bash
python3 -B stage_6_tinystories/tools/prepare_transfer.py
python3 -B stage_6_tinystories/tools/verify_transfer.py
```

The generator verifies the genuine dataset/tokenizer identities and both pinned
baseline copies before replacing the controls. It does not submit jobs, train,
download data, or include experiment outputs.

## Pinned identities

| Asset | SHA256 |
|---|---|
| Stage 5 baseline | `2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce` |
| Dataset manifest | `89c9df839572ef6a1890caf4b1144229090b8ced8b4ea2c66ed5dd090e17c9a7` |
| Tokenizer | `99055f35c77392e4df4486685b60fee1de1d95e2c373c26f487cf9944746d038` |

Historical packaging validation and repaired-data findings are recorded in
[HPC_VALIDATION.md](HPC_VALIDATION.md) and [DATA_RECOVERY.md](DATA_RECOVERY.md).
