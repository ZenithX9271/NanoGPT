#!/usr/bin/env python
"""Short fresh-process restart proof on the allocated GPU (or explicit CPU smoke)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from stage_6_tinystories.core import load_checkpoint, write_json
from stage_6_tinystories.data import read_json, resolve, sha256
from stage_6_tinystories.train import safe_run_dir


def identical(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def main():
    stage = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=str(stage.parent))
    parser.add_argument('--config', default=str(stage/'configs/continue.json'))
    parser.add_argument('--baseline')
    parser.add_argument('--run-dir', default='stage_6_tinystories/runs/restart_check')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, help='Default: real experiment batch 32')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--walltime-seconds', type=float)
    parser.add_argument('--job-start-epoch', type=float)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    cfg = read_json(resolve(root, args.config))
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('Batch size must be positive')
        cfg['training']['batch_size'] = args.batch_size
    out = safe_run_dir(root, args.run_dir)
    baseline = resolve(root, args.baseline or cfg['initial_checkpoint'])
    baseline_hash = sha256(baseline)
    cfg['experiment'] = 'restart_equivalence_validation'
    cfg['training'].update(additional_tokens=8*cfg['training']['batch_size']*cfg['model']['block_size'],
                           eval_interval=2, save_interval=2, log_interval=1, snapshot_tokens=[])
    cfg['evaluation'].update(windows=8, train_windows=8, batch_size=4)
    config_path = out/'restart_config.json'
    write_json(cfg, config_path)
    common = [sys.executable, '-B', '-u', str(stage/'train.py'), '--project-root', str(root),
              '--config', str(config_path), '--device', args.device, '--cpu-threads', str(args.cpu_threads)]
    for option, value in [('--walltime-seconds', args.walltime_seconds), ('--job-start-epoch', args.job_start_epoch)]:
        if value is not None:
            common += [option, str(value)]
    commands = [('full', ['--init-from', str(baseline), '--run-dir', str(out/'full'), '--stop-after-updates', '4']),
                ('split', ['--init-from', str(baseline), '--run-dir', str(out/'split'), '--stop-after-updates', '2']),
                ('resume', ['--resume', str(out/'split/latest.pt'), '--run-dir', str(out/'split'), '--stop-after-updates', '4'])]
    for label, options in commands:
        with open(str(out/(label+'.log')), 'w') as stream:
            subprocess.run(common+options, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(label+' process completed', flush=True)
    a, b = [load_checkpoint(out/name/'latest.pt') for name in ('full', 'split')]
    keys = ['model', 'optimizer', 'rng', 'stage6_updates', 'completed_updates', 'tokens', 'last_eval']
    matched = {key: identical(a[key], b[key]) for key in keys}
    passed = all(matched.values()) and a['stage6_updates'] == 4 and sha256(baseline) == baseline_hash
    report = {'passed': passed, 'comparisons': matched, 'runtime': a['runtime'],
              'batch_size': cfg['training']['batch_size'], 'context': cfg['model']['block_size'],
              'dropout': cfg['model']['dropout'], 'completed_updates': a['completed_updates'],
              'tokens': a['tokens'], 'source_unchanged': sha256(baseline) == baseline_hash,
              'note': '4 uninterrupted updates versus 2 + fresh-process resume + 2; validation, not quality training'}
    write_json(report, out/'restart_check.json')
    print(report, flush=True)
    if not passed:
        raise SystemExit('Restart equivalence failed: return restart_check.json and worker logs before long training')


if __name__ == '__main__':
    main()
