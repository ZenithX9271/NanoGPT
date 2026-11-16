"""Read-only verification of the saved Stage 7 records; never starts training."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / 'stage_7_general_lm'
OUT = Path(__file__).parent


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


segments = [
    ('100m', 1141598, STAGE / 'artifacts/checkpoints/100m_1141598/evidence'),
    ('200m', 1141809, STAGE / 'artifacts/checkpoints/200m_1141809/evidence'),
    ('350m', 1142015, STAGE / 'reports/continuation_1142015'),
    ('500m', 1142016, STAGE / 'reports/continuation_1142016'),
]
for group, entries in [
    ('to1b_20261112', [('625m',1144922),('750m',1144923),('875m',1144924),('1b',1144925)]),
    ('to1p5b_20261113', [('1p125b',1147658),('1p25b',1147659),('1p375b',1147660),('1p5b',1147661)]),
    ('to2b_20261115', [('1p625b',1150576),('1p75b',1150577),('1p875b',1150578),('2b',1150579)]),
]:
    segments += [(name, job, STAGE / ('reports/continuation_' + group) / ('continuation_' + str(job))) for name,job in entries]

cfg = read(STAGE / 'configs/production_2b_first100m.json')
candidate = read(STAGE / 'configs/production_2b_candidate.json')
assert all(cfg[k] == candidate[k] for k in ['model','training','evaluation'])
identity = read(segments[0][2] / 'launch.json')['code_identity']
assert all(sha(STAGE / name) == digest for name,digest in identity.items())
chain, last, allocation_seconds = [], None, 0
for name,job,p in segments:
    launch = read(p / 'launch.json')
    endpoint = OUT / 'recovered_350m' if job == 1142015 else p
    checkpoint, allocation = [read(endpoint / (n+'.json')) for n in ['checkpoint','allocation']]
    if last is not None:
        assert launch['source_checkpoint']['checkpoint_sha256'] == last['checkpoint_sha256'], name
        assert launch['source_checkpoint']['summary']['counters'] == last['summary']['counters'], name
    if 'code_identity' in launch:
        assert launch['code_identity'] == identity, name
    if 'config_sha256' in launch:
        assert launch['config_sha256'] == sha(STAGE / 'configs/production_2b_first100m.json'), name
    assert allocation['exit_code'] == 0, name
    assert checkpoint['summary']['reason'] == ('completed' if name == '2b' else 'review_gate')
    allocation_seconds += allocation['allocation_seconds']
    chain.append(dict(milestone=name, job_id=job, checkpoint_sha256=checkpoint['checkpoint_sha256'],
                      counters=checkpoint['summary']['counters'], allocation_seconds=allocation['allocation_seconds']))
    last = checkpoint

# The final metrics copy contains the full history, including rejected AMP steps.
t = cfg['training']
positions = t['micro_batch_size'] * t['accumulation_steps'] * cfg['model']['block_size']
horizon = math.ceil(t['token_budget'] / positions)
attempts = updates = skips = evaluations = 0
max_lr_error = 0.0
milestone_counters = {x['counters']['attempts']:x['counters'] for x in chain}
with (segments[-1][2] / 'metrics.jsonl').open() as f:
    for line in f:
        row = json.loads(line)
        if row['event'] == 'evaluation':
            evaluations += 1
            continue
        assert row['event'] == 'update'
        warmup = min(t['warmup_updates'],horizon)
        if updates < warmup:
            expected_lr = t['peak_lr'] * (updates + 1) / warmup
        else:
            progress = min(1., (updates - warmup) / max(1, horizon - warmup - 1))
            expected_lr = t['min_lr'] + .5 * (t['peak_lr'] - t['min_lr']) * (1 + math.cos(math.pi * progress))
        error = abs(row['lr'] - expected_lr)
        max_lr_error = max(max_lr_error,error)
        assert error < 1e-12
        attempts += 1
        if row['optimizer_step_succeeded']:
            updates += 1
            assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
        else:
            skips += 1
        assert row['attempts'] == attempts and row['updates'] == updates
        assert row['processed_positions'] == attempts * positions
        assert row['successful_positions'] == updates * positions
        if attempts in milestone_counters:
            assert all(row[k] == v for k,v in milestone_counters[attempts].items())
assert updates == horizon == 122071 and skips == 46

backups = {}
for folder in sorted((STAGE / 'artifacts/checkpoints').iterdir()):
    manifest = folder / 'backup_manifest.json'
    receipt_path = folder / 'mac_backup_verified.json'
    if not receipt_path.exists():
        continue
    receipt = read(receipt_path)
    if manifest.exists():
        assert sha(manifest) == receipt['backup_manifest_sha256']
        records = read(manifest)['files']
    else:
        checkpoint_path = next(folder.glob('*.pt'))
        records = {checkpoint_path.name:dict(bytes=receipt['checkpoint_bytes'],sha256=receipt['checkpoint_sha256'])}
        source = folder / 'source_snapshot'
        for name,digest in identity.items():
            p = source / name
            records[str(p.relative_to(folder))] = dict(bytes=p.stat().st_size,sha256=digest)
        config_path = folder / 'production_2b_first100m.json'
        records[config_path.name] = dict(bytes=config_path.stat().st_size,sha256=sha(STAGE / 'configs/production_2b_first100m.json'))
        lock = read(STAGE / 'configs/native_tokenizer.lock.json')['files']
        for name,rec in lock.items():
            records['tokenizer/'+name] = dict(bytes=rec.get('bytes',rec.get('size')),sha256=rec['sha256'])
    total = 0
    for name, rec in records.items():
        p = folder / name
        assert p.stat().st_size == rec['bytes'], str(p)
        assert sha(p) == rec['sha256'], str(p)
        total += rec['bytes']
    backups[folder.name] = dict(status='PASS',verified_files=len(records),bytes=total,
                               scope='full backup manifest' if manifest.exists() else 'checkpoint, source identity, config, tokenizer lock',
                               manifest_sha256=sha(manifest) if manifest.exists() else None)

preservation = read(STAGE / 'reports/stage6_preservation.json')['baseline']
changed,missing = [],[]
for name,rec in preservation.items():
    p = ROOT / name
    if not p.is_file():
        missing.append(name)
    elif p.stat().st_size != rec['bytes'] or sha(p) != rec['sha256']:
        changed.append(name)

result = dict(status='PASS',segments=chain,
              training_config_equal_to_original_gpu_gate_candidate=True,
              current_core_code_matches_frozen_identity=True,
              metrics=dict(attempts=attempts,updates=updates,amp_skips=skips,evaluations=evaluations,
                           positions_per_update=positions,successful_positions=updates*positions,
                           processed_positions=attempts*positions,max_lr_error=max_lr_error),
              sixteen_training_allocation_hours=allocation_seconds/3600,
              final_recorded_cumulative_hours=read(segments[-1][2]/'allocation.json')['cumulative_gpu_hours_upper_bound'],
              backups=backups,
              stage6_preservation=dict(baseline_files=len(preservation),changed=changed,missing=missing))
(OUT / 'records_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['segments','backups']},indent=2))
print('Verified backup bundles:', ', '.join(backups))
