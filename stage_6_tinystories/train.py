#!/usr/bin/env python
"""Stage 6A single-GPU continuation, read-only preflight, or isolated benchmark."""
import os
import time

# Set before importing torch/CUDA, including when using the legacy module.
PROCESS_START = time.monotonic()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import math
import signal
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from stage_6_tinystories.core import (Engine, EvaluationInterrupted, atomic_save,
    configure_device, evaluate, git_identity, host_peak_mib, runtime_info, synchronize, write_json)
from stage_6_tinystories.data import read_json, resolve, validate_assets


class Deadline:
    def __init__(self, walltime, margin, job_start_epoch=None):
        self.signal = None
        self.margin = margin
        elapsed = time.time()-job_start_epoch if job_start_epoch is not None else time.monotonic()-PROCESS_START
        self.end = None if walltime is None else time.monotonic()+walltime-max(0., elapsed)

    def remaining(self):
        return float('inf') if self.end is None else self.end-time.monotonic()

    def stop(self, allowance=0.):
        return self.signal is not None or self.remaining() <= self.margin+allowance

    def receive(self, number, frame):
        self.signal = signal.Signals(number).name


def append_event(directory, value):
    value = dict(value, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with open(str(directory/'metrics.jsonl'), 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def safe_run_dir(root, value, resume=None):
    directory = resolve(root, value).resolve()
    protected_names = ['stage_1_bigram', 'stage_2_gpt_minimal', 'stage_3_gpt_shakespeare',
                       'stage_4_gpt_scaled', 'stage_5_tinystories', 'data', '.git']
    protected = [(root/name).resolve() for name in protected_names]
    if directory == root.resolve() or any(p == directory or p in directory.parents for p in protected):
        raise ValueError('Use a dedicated Stage 6 output directory, outside preserved stages/data and the project root')
    if directory.exists() and any(directory.iterdir()):
        if resume is None or Path(resume).resolve().parent != directory:
            raise ValueError('Output directory is not empty; use another name or resume its checkpoint')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save(engine, directory, name):
    synchronize(engine.device)
    start = time.perf_counter()
    atomic_save(engine.checkpoint(), directory/name)
    synchronize(engine.device)
    seconds = time.perf_counter()-start
    append_event(directory, {'event': 'checkpoint', 'file': name, 'seconds': seconds,
        'bytes': (directory/name).stat().st_size, 'stage6_updates': engine.run_updates,
        'tokens': engine.tokens, 'additional_tokens': engine.additional_tokens,
        'last_eval_updates': None if engine.last_eval is None else engine.last_eval['stage6_updates']})
    return seconds


def train(engine, directory, deadline, stop_after=None):
    cfg = engine.cfg
    safety, tc = cfg['safety'], cfg['training']
    eval_allowance = safety['initial_eval_allowance_seconds']
    save_allowance = safety['initial_save_allowance_seconds']
    last_step_seconds = 0.

    def evaluate_and_save():
        nonlocal eval_allowance, save_allowance
        if deadline.stop(eval_allowance+save_allowance):
            return False
        synchronize(engine.device)
        start = time.perf_counter()
        try:
            result = evaluate(engine, stop=lambda: deadline.stop(save_allowance))
        except EvaluationInterrupted:
            return False
        synchronize(engine.device)
        elapsed = time.perf_counter()-start
        eval_allowance = max(eval_allowance, elapsed*1.5)
        engine.last_eval = result
        improved = engine.best_eval is None or result['dev_val']['ce'] < engine.best_eval['dev_val']['ce']
        if improved:
            engine.best_eval = result
            save_allowance = max(save_allowance, save(engine, directory, 'best.pt')*1.5)
        append_event(directory, dict(event='evaluation', seconds=elapsed, improved=improved, **result))
        print('EVAL new_updates={} additional_tokens={:,} dev_ce={:.6f} ppl={:.3f} time={:.2f}s{}'.format(
            engine.run_updates, engine.additional_tokens, result['dev_val']['ce'],
            result['dev_val']['perplexity'], elapsed, ' BEST' if improved else ''), flush=True)
        return True

    reason = 'completed'
    if engine.last_eval is None:
        if not evaluate_and_save():
            reason = deadline.signal or 'walltime_before_or_during_evaluation'
        save_allowance = max(save_allowance, save(engine, directory, 'latest.pt')*1.5)

    # A previous walltime exit may have completed the final update but interrupted
    # its evaluation/snapshot. Finish that work on restart WITHOUT an extra update.
    pending = [t for t in tc['snapshot_tokens'] if t <= engine.additional_tokens and t not in engine.saved_snapshots]
    at_boundary = engine.run_updates >= engine.total_run_updates or (
        stop_after is not None and engine.run_updates >= stop_after)
    if reason == 'completed' and (pending or at_boundary):
        if engine.last_eval is None or engine.last_eval['stage6_updates'] != engine.run_updates:
            if not evaluate_and_save():
                reason = deadline.signal or 'walltime_before_or_during_evaluation'
        if reason == 'completed':
            for threshold in pending:
                engine.saved_snapshots.append(threshold)
                save_allowance = max(save_allowance, save(
                    engine, directory, 'snapshot_additional_{}.pt'.format(threshold))*1.5)

    while engine.run_updates < engine.total_run_updates and reason == 'completed':
        if stop_after is not None and engine.run_updates >= stop_after:
            reason = 'review_gate'
            break
        if deadline.stop(save_allowance+last_step_seconds):
            reason = deadline.signal or 'walltime_margin'
            break
        start = time.perf_counter()
        loss, norm = engine.update()
        last_step_seconds = time.perf_counter()-start
        u = engine.run_updates
        is_final = u == engine.total_run_updates or (stop_after is not None and u >= stop_after)
        targets = [t for t in tc['snapshot_tokens'] if t <= engine.additional_tokens and t not in engine.saved_snapshots]
        # Bounded synchronization for finite checks/logs, rather than every benchmark step.
        if u % tc['log_interval'] == 0 or targets or is_final or u % tc['save_interval'] == 0 or u % tc['eval_interval'] == 0:
            loss_value, norm_value = loss.item(), norm.item()
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise FloatingPointError('Nonfinite loss/gradient: last valid checkpoint preserved')
            append_event(directory, {'event': 'training', 'stage6_updates': u,
                'completed_updates': engine.base_updates+u, 'additional_tokens': engine.additional_tokens,
                'tokens': engine.tokens, 'loss': loss_value, 'grad_norm_before_clip': norm_value,
                'lr': engine.optimizer.param_groups[0]['lr']})
            print('UPDATE {} total_updates={} new_tokens={:,} loss={:.4f} grad={:.3f} lr={:.3g}'.format(
                u, engine.base_updates+u, engine.additional_tokens, loss_value, norm_value,
                engine.optimizer.param_groups[0]['lr']), flush=True)
        if u % tc['eval_interval'] == 0 or targets or is_final:
            if not evaluate_and_save():
                reason = deadline.signal or 'walltime_before_or_during_evaluation'
                break
        for threshold in targets:
            engine.saved_snapshots.append(threshold)
            save_allowance = max(save_allowance, save(
                engine, directory, 'snapshot_additional_{}.pt'.format(threshold))*1.5)
        if u % tc['save_interval'] == 0:
            save_allowance = max(save_allowance, save(engine, directory, 'latest.pt')*1.5)

    # Save current weights/optimizer/RNG before any other model is loaded. No best reload.
    save(engine, directory, 'latest.pt')
    report = {'event': 'exit', 'reason': reason, 'stage6_updates': engine.run_updates,
              'completed_updates': engine.base_updates+engine.run_updates,
              'additional_tokens': engine.additional_tokens, 'tokens': engine.tokens,
              'runtime_seconds': time.monotonic()-PROCESS_START, 'host_peak_mib': host_peak_mib(),
              'last_eval': engine.last_eval, 'best_eval': engine.best_eval,
              'remaining_walltime_seconds': None if deadline.end is None else deadline.remaining()}
    append_event(directory, report)
    write_json(report, directory/'summary.json')
    print(json.dumps(report, indent=2), flush=True)
    return report


def benchmark(engine, directory, deadline):
    bc = engine.cfg['benchmark']
    minimum_extra = engine.cfg['safety']['initial_save_allowance_seconds']
    start_updates = engine.run_updates
    if start_updates+bc['warmup_updates']+bc['windows']*bc['updates_per_window'] > engine.total_run_updates:
        raise ValueError('Benchmark exceeds remaining configured schedule; use a fresh baseline/config')
    for _ in range(bc['warmup_updates']):
        if deadline.stop(minimum_extra):
            raise RuntimeError('Insufficient walltime for benchmark; no trained checkpoint written')
        engine.update()
    synchronize(engine.device)
    if engine.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(engine.device)
    windows, latencies = [], []
    for w in range(bc['windows']):
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(bc['updates_per_window'])] if engine.device.type == 'cuda' else []
        synchronize(engine.device)
        begin = time.perf_counter()
        losses = []
        for j in range(bc['updates_per_window']):
            if deadline.stop(minimum_extra):
                raise RuntimeError('Benchmark stopped for walltime; partial windows are not reported')
            if events:
                events[j][0].record()
            loss, norm = engine.update()
            losses.append(loss)
            if events:
                events[j][1].record()
        synchronize(engine.device)  # one synchronization at this window's end
        elapsed = time.perf_counter()-begin
        if not torch.isfinite(torch.stack(losses)).all().item() or not torch.isfinite(norm).item():
            raise FloatingPointError('Nonfinite benchmark workload')
        if events:
            latencies.extend(a.elapsed_time(b) for a, b in events)
        tokens = bc['updates_per_window']*engine.tokens_per_update
        row = {'window': w+1, 'seconds': elapsed, 'tokens': tokens, 'tokens_per_second': tokens/elapsed}
        windows.append(row)
        print('BENCH window={}/{} tokens/s={:,.0f} seconds={:.3f}'.format(
            w+1, bc['windows'], row['tokens_per_second'], elapsed), flush=True)
    if deadline.stop(engine.cfg['safety']['initial_eval_allowance_seconds']+minimum_extra):
        raise RuntimeError('Insufficient time to measure evaluation/full-save overhead')
    synchronize(engine.device)
    begin = time.perf_counter()
    evaluation = evaluate(engine, stop=lambda: deadline.stop(minimum_extra))
    synchronize(engine.device)
    eval_seconds = time.perf_counter()-begin
    probe = directory/'full_checkpoint_probe.pt'
    begin = time.perf_counter()
    atomic_save(engine.checkpoint(), probe)
    synchronize(engine.device)
    save_seconds = time.perf_counter()-begin
    probe_bytes = probe.stat().st_size
    probe.unlink()
    total_tokens = sum(x['tokens'] for x in windows)
    weighted_tps = total_tokens/sum(x['seconds'] for x in windows)
    rates = [x['tokens_per_second'] for x in windows]
    remaining_updates = engine.total_run_updates-start_updates
    # Budget describes a NEW training invocation; benchmark updates are discarded.
    estimate = (remaining_updates*engine.tokens_per_update/weighted_tps
                + (math.ceil(remaining_updates/engine.cfg['training']['eval_interval'])+1)*eval_seconds
                + (math.ceil(remaining_updates/engine.cfg['training']['save_interval'])
                   + 2*len(engine.cfg['training']['snapshot_tokens'])+3)*save_seconds)
    memory = {'host_peak_mib': host_peak_mib()}
    if engine.device.type == 'cuda':
        memory.update(gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(engine.device)/1024**2,
                      gpu_peak_reserved_mib=torch.cuda.max_memory_reserved(engine.device)/1024**2)
    try:
        memory['nvidia_smi_snapshot'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except (OSError, subprocess.SubprocessError):
        memory['nvidia_smi_snapshot'] = None
    report = {'mode': 'benchmark_only_updates_discarded', 'runtime': runtime_info(engine.device),
              'warmup_updates': bc['warmup_updates'], 'windows': windows,
              'tokens_per_second': weighted_tps, 'min_tokens_per_second': min(rates),
              'max_tokens_per_second': max(rates), 'range_over_mean': (max(rates)-min(rates))/statistics.mean(rates),
              'timing': 'window wall clock; CUDA sync only at boundaries; event pairs read after window',
              'step_cuda_median_ms': statistics.median(latencies) if latencies else None,
              'step_cuda_p95_ms': sorted(latencies)[min(len(latencies)-1, math.ceil(.95*len(latencies))-1)] if latencies else None,
              'evaluation_seconds': eval_seconds, 'evaluation': evaluation,
              'full_checkpoint_seconds': save_seconds, 'full_checkpoint_bytes': probe_bytes,
              'memory': memory, 'runtime_seconds': time.monotonic()-PROCESS_START,
              'estimated_new_training_seconds_excluding_startup_and_margin': estimate,
              'budget_note': 'Planning estimate from this configuration, not an affordability guarantee; include startup, variance and PBS margin.',
              'origin': engine.origin, 'data_identity': engine.identity}
    write_json(report, directory/'benchmark.json')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).parent/'configs/continue.json'))
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--mode', choices=['preflight', 'benchmark', 'train'], default='train')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--init-from', help='Stage 5 checkpoint; default is in config')
    source.add_argument('--resume', help='Stage 6 checkpoint, exact restart on same software/hardware')
    parser.add_argument('--run-dir', default='stage_6_tinystories/runs/s6a')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--verify-shards', action='store_true', help='Read and SHA256-check all train/dev shards once')
    parser.add_argument('--stop-after-updates', type=int, help='Stop at this Stage 6 cumulative update count; schedule remains unchanged')
    parser.add_argument('--walltime-seconds', type=float)
    parser.add_argument('--job-start-epoch', type=float)
    args = parser.parse_args()
    if args.cpu_threads < 1 or (args.stop_after_updates is not None and args.stop_after_updates < 1):
        parser.error('Thread/stop counts must be positive')
    if args.walltime_seconds is not None and args.walltime_seconds <= 0:
        parser.error('Walltime must be positive')
    root = Path(args.project_root).resolve()
    cfg = read_json(resolve(root, args.config))
    deadline = Deadline(args.walltime_seconds, cfg['safety']['margin_seconds'], args.job_start_epoch)
    signal.signal(signal.SIGTERM, deadline.receive)
    signal.signal(signal.SIGINT, deadline.receive)
    device = configure_device(args.device, args.cpu_threads)
    datasets, identity, prompts = validate_assets(root, cfg, args.verify_shards)
    engine = Engine(cfg, device, datasets, identity)
    source_path = resolve(root, args.resume or args.init_from or cfg['initial_checkpoint'])
    prior_runtime = engine.resume(source_path) if args.resume else None
    if not args.resume:
        engine.start_from_stage5(source_path)
    print(json.dumps({'environment': runtime_info(device), 'origin': engine.origin,
        'parameters': sum(p.numel() for p in engine.model.parameters()), 'data_identity': identity,
        'planned_new_updates': engine.total_run_updates,
        'planned_new_tokens': engine.total_run_updates*engine.tokens_per_update,
        'resumed_updates': engine.run_updates, 'prior_runtime': prior_runtime}, indent=2), flush=True)
    if args.mode == 'preflight':
        print('PREFLIGHT PASS: strict model/optimizer load; data identity; no update or output file written.', flush=True)
        return
    directory = safe_run_dir(root, args.run_dir, source_path if args.resume else None)
    if args.mode == 'train' and args.resume and (directory != source_path.resolve().parent or source_path.name != 'latest.pt'):
        raise ValueError('Resume latest.pt in its existing run directory; transfer the entire run folder between machines')
    invocation = {'mode': args.mode, 'arguments': vars(args), 'config': cfg, 'git': git_identity(root),
                  'code_identity': engine.code, 'data_identity': identity, 'runtime': runtime_info(device),
                  'origin': engine.origin, 'starting_stage6_updates': engine.run_updates}
    path = directory/'provenance.json'
    if not path.exists():
        write_json(invocation, path)
    append_event(directory, dict(event='invocation', **invocation))
    if args.mode == 'benchmark':
        benchmark(engine, directory, deadline)
    else:
        train(engine, directory, deadline, args.stop_after_updates)


if __name__ == '__main__':
    main()
