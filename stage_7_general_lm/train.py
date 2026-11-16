"""Scratch training/benchmark entry point. Main launch requires explicit limits.

Budget/schedule count positions contributing to successful optimizer updates.
Processed positions include AMP-overflow attempts; every checkpoint is at an
optimizer boundary with no outstanding accumulated gradients.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import torch

from .common import (atomic_checkpoint, configure, load_checkpoint, read_json,
                     restore_rng, rng_state, runtime, seed_all, source_identity, write_json)
from .data import TokenStream
from .model import LanguageModel


class Engine:
    def __init__(self, cfg, train_data, dev_data, device='cpu'):
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(device)
        self.training_data, self.dev_data = train_data, dev_data
        t = cfg['training']
        if not isinstance(t['token_budget'], int) or t['token_budget'] <= 0:
            raise ValueError('A positive, fixed token_budget is required')
        if min(t['micro_batch_size'], t['accumulation_steps']) <= 0:
            raise ValueError('Positive batch and accumulation required')
        if t['precision'] not in ('fp32', 'fp16') or (t['precision'] == 'fp16' and self.device.type != 'cuda'):
            raise ValueError('FP16/scaler path is CUDA-only; use FP32 CPU reference')
        if cfg['model']['vocab_size'] != train_data.manifest['vocab_size']:
            raise ValueError('Model/tokenizer vocabulary mismatch')
        if train_data.identity != dev_data.identity:
            raise ValueError('Train/development manifests must match')
        self.code, self.data_identity = source_identity(), train_data.identity
        seed_all(t['seed'])
        self.sampler = torch.Generator().manual_seed(t['seed'] + 1)
        self.model = LanguageModel(cfg['model']).to(self.device)
        params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW([
            {'params': [p for p in params if p.ndim >= 2], 'weight_decay': t['weight_decay']},
            {'params': [p for p in params if p.ndim < 2], 'weight_decay': 0.0}],
            lr=t['peak_lr'], betas=tuple(t['betas']))
        self.scaler = torch.cuda.amp.GradScaler(enabled=t['precision'] == 'fp16')
        self.updates = self.attempts = self.processed_positions = self.successful_positions = 0
        self.positions_per_update = t['micro_batch_size'] * t['accumulation_steps'] * cfg['model']['block_size']
        self.horizon = math.ceil(t['token_budget'] / self.positions_per_update)

    def lr(self):
        t = self.cfg['training']
        warmup = min(t['warmup_updates'], self.horizon)
        if self.updates < warmup:
            return t['peak_lr'] * (self.updates + 1) / warmup
        progress = min(1., (self.updates - warmup) / max(1, self.horizon - warmup - 1))
        return t['min_lr'] + .5 * (t['peak_lr'] - t['min_lr']) * (1 + math.cos(math.pi * progress))

    def update(self, batches=None):
        t = self.cfg['training']
        if self.updates >= self.horizon:
            raise ValueError('Original schedule is complete')
        if batches is None:
            batches = [self.training_data.batch(t['micro_batch_size'], self.cfg['model']['block_size'],
                       self.sampler, self.device) for _ in range(t['accumulation_steps'])]
        if len(batches) != t['accumulation_steps']:
            raise ValueError('Expected one batch per accumulation microstep')
        valid = sum(int(y.ne(-100).sum()) for _, y in batches)
        if not valid:
            raise ValueError('No supervised positions')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        learning_rate = self.lr()
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate
        loss_value = 0.
        for x, y in batches:
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                _, summed = self.model(x, y, reduction='sum')
                loss = summed / valid
            loss_value += float(loss.detach())
            self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t['grad_clip'])
        if not self.scaler.is_enabled() and (not math.isfinite(loss_value) or not torch.isfinite(norm)):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite FP32 loss/gradients; no checkpoint published')
        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        successful = self.scaler.get_scale() >= scale
        self.attempts += 1
        self.processed_positions += sum(x.numel() for x, _ in batches)
        if successful:
            self.updates += 1
            self.successful_positions += valid
        self.optimizer.zero_grad(set_to_none=True)
        return dict(event='update', updates=self.updates, attempts=self.attempts,
                    processed_positions=self.processed_positions,
                    successful_positions=self.successful_positions, lr=learning_rate,
                    loss=loss_value if math.isfinite(loss_value) else None,
                    grad_norm=float(norm) if torch.isfinite(norm) else None,
                    optimizer_step_succeeded=successful, amp_scale=self.scaler.get_scale())

    @torch.no_grad()
    def evaluate(self):
        old = self.model.training
        self.model.eval()
        ec = self.cfg['evaluation']
        result = {}
        try:
            domains = sorted({r['source'] for r in self.dev_data.manifest['splits']['dev']['shards'] if r.get('source')})
            splits = [('train', self.training_data), ('dev', self.dev_data)]
            splits += [('dev_domain:' + name, TokenStream(self.dev_data.path, 'dev', verify=False, domain=name)) for name in domains]
            for name, data in splits:
                if data.size <= self.cfg['model']['block_size']:
                    result[name] = dict(status='insufficient_tokens', available_tokens=data.size)
                    continue
                generator = torch.Generator().manual_seed(ec['seed'] + (name == 'train'))
                total, count = 0., 0
                for start in range(0, ec['windows'], ec['batch_size']):
                    n = min(ec['batch_size'], ec['windows'] - start)
                    x, y = data.batch(n, self.cfg['model']['block_size'], generator, self.device)
                    _, loss = self.model(x, y, reduction='sum')
                    total += float(loss)
                    count += y.numel()
                result[name] = dict(ce=total / count, perplexity=math.exp(min(700., total / count)), positions=count)
        finally:
            self.model.train(old)
        return dict(event='evaluation', updates=self.updates, scores=result, precision='fp32')

    def state(self):
        if self.processed_positions != self.attempts * self.positions_per_update or self.successful_positions != self.updates * self.positions_per_update:
            raise ValueError('Scratch checkpoint requires full unpadded update boundaries; masked SFT uses its separate entry point')
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('Checkpoint only at an optimizer boundary')
        if not all(torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise FloatingPointError('Nonfinite model cannot be saved')
        return dict(format='stage7_resume_v1', config=self.cfg, code_identity=self.code,
                    data_identity=self.data_identity, runtime=runtime(self.device),
                    model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    scaler=self.scaler.state_dict(), rng=rng_state(self.sampler),
                    counters={k: getattr(self, k) for k in ('updates', 'attempts', 'processed_positions', 'successful_positions')},
                    counter_policy='schedule: successful optimizer updates; budget: successful supervised positions')

    def save(self, path):
        atomic_checkpoint(path, self.state())

    def resume(self, path):
        c = load_checkpoint(path)
        if c.get('format') != 'stage7_resume_v1' or c['config'] != self.cfg or c['code_identity'] != self.code or c['data_identity'] != self.data_identity:
            raise ValueError('Exact-resume config/source/data guard failed')
        if c['runtime'] != runtime(self.device):
            raise ValueError('Runtime differs; migration needs a documented new experiment')
        if not torch.equal(c['model']['embedding.weight'], c['model']['lm_head.weight']):
            raise ValueError('Inconsistent tied weights')
        counters = c['counters']
        if not 0 <= counters['updates'] <= self.horizon or counters['attempts'] < counters['updates'] or counters['processed_positions'] < counters['successful_positions']:
            raise ValueError('Invalid checkpoint accounting')
        if counters['successful_positions'] != counters['updates'] * self.positions_per_update:
            raise ValueError('Scratch stream requires unpadded, fixed-size update accounting')
        if counters['processed_positions'] != counters['attempts'] * self.positions_per_update:
            raise ValueError('Processed-position counter does not match attempted microbatches')
        self.model.load_state_dict(c['model'], strict=True)
        self.optimizer.load_state_dict(c['optimizer'])
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state.get(p, {})
                if counters['updates']:
                    step = state.get('step', -1)
                    if int(step.item() if torch.is_tensor(step) else step) != counters['updates']:
                        raise ValueError('Optimizer step does not match counters')
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if state[key].shape != p.shape or not torch.isfinite(state[key]).all():
                            raise ValueError('Invalid optimizer moments')
        self.scaler.load_state_dict(c['scaler'])
        for key, value in counters.items():
            setattr(self, key, value)
        restore_rng(c['rng'], self.sampler)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--mode', choices=['benchmark', 'train'], required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--run-dir')
    p.add_argument('--resume')
    p.add_argument('--review-updates', type=int)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--report', help='New benchmark JSON; temporary optimizer updates are discarded')
    p.add_argument('--checkpoint-probe-dir', help='Existing quota-checked directory on the intended checkpoint filesystem')
    p.add_argument('--job-start-epoch', type=float)
    args = p.parse_args()
    if not 1 <= args.steps <= 100 or not 0 <= args.warmup <= 50:
        p.error('Benchmark limited to 100 measured / 50 warmup updates')
    cfg = read_json(args.config)
    if args.mode == 'benchmark':
        if args.resume or args.run_dir:
            p.error('Benchmark cannot use an existing quality run')
        t = cfg['training']
        t['token_budget'] = max(2, args.steps + args.warmup + 1) * t['micro_batch_size'] * t['accumulation_steps'] * cfg['model']['block_size']
        cfg['benchmark_schedule_placeholder'] = True
        if args.device.startswith('cuda') and not args.checkpoint_probe_dir:
            p.error('GPU benchmark requires --checkpoint-probe-dir on the intended checkpoint filesystem')
        if args.checkpoint_probe_dir:
            probe = Path(args.checkpoint_probe_dir).resolve()
            if not probe.is_dir() or any(part.startswith('stage_') and part.split('_')[1] in ('1','2','3','4','5','6') for part in probe.parts):
                p.error('Probe directory must exist outside frozen stages')
    else:
        limits = cfg['resource_limits']
        if not limits.get('main_run_approved') or any(not limits.get(k) or limits[k] <= 0 for k in ('gpu_hours', 'maximum_job_seconds', 'disk_bytes')):
            p.error('Freeze token horizon and positive approved resource limits before training')
        if not args.run_dir:
            p.error('--run-dir required')
    device = configure(args.device, args.threads)
    started = time.perf_counter()
    data, dev = TokenStream(args.manifest), TokenStream(args.manifest, 'dev')
    engine = Engine(cfg, data, dev, device)
    if args.mode == 'benchmark':
        for _ in range(args.warmup):
            engine.update()
        timings = []
        measured_start_processed = engine.processed_positions
        measured_start_successful = engine.successful_positions
        measured_start_updates = engine.updates
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.steps):
            synchronize(device)
            start = time.perf_counter()
            engine.update()
            synchronize(device)
            timings.append(time.perf_counter() - start)
        synchronize(device)
        start = time.perf_counter()
        evaluation = engine.evaluate()
        synchronize(device)
        eval_seconds = time.perf_counter() - start
        with tempfile.TemporaryDirectory(prefix='stage7_benchmark_', dir=args.checkpoint_probe_dir) as temporary:
            probe_parent = str(Path(temporary).parent.resolve())
            probe_device = os.stat(temporary).st_dev
            checkpoint = Path(temporary) / 'probe.pt'
            start = time.perf_counter()
            engine.save(checkpoint)
            save_seconds = time.perf_counter() - start
            checkpoint_bytes = checkpoint.stat().st_size
            start = time.perf_counter()
            engine.resume(checkpoint)
            restore_seconds = time.perf_counter() - start
        report = dict(mode='benchmark_only', quality_training=False, runtime=runtime(device), config=cfg,
                      code_identity=engine.code, data_identity=engine.data_identity,
                      parameters=sum(p.numel() for p in engine.model.parameters()),
                      tokens_per_update=engine.positions_per_update, step_seconds=timings,
                      measured_processed_positions=engine.processed_positions - measured_start_processed,
                      measured_successful_positions=engine.successful_positions - measured_start_successful,
                      measured_optimizer_updates=engine.updates - measured_start_updates,
                      training_tokens_per_second=(engine.successful_positions - measured_start_successful) / sum(timings),
                      processed_tokens_per_second=(engine.processed_positions - measured_start_processed) / sum(timings),
                      evaluation=evaluation, evaluation_seconds=eval_seconds, checkpoint_bytes=checkpoint_bytes,
                      save_seconds=save_seconds, restore_seconds=restore_seconds,
                      checkpoint_probe_parent=probe_parent, checkpoint_probe_device=probe_device,
                      total_seconds=time.perf_counter() - started,
                      gpu_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                      gpu_peak_reserved=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return
    directory = Path(args.run_dir).resolve()
    if any(part.startswith('stage_') and part.split('_')[1] in ('1', '2', '3', '4', '5', '6') for part in directory.parts) or directory == Path.cwd() or 'data' in directory.parts:
        p.error('Use a new Stage 7 run directory, never frozen stages/data')
    marker = directory / 'stage7_owned.json'
    if args.resume:
        if not marker.is_file() or Path(args.resume).resolve().parent != directory:
            p.error('Resume only in its owned run directory')
        engine.resume(args.resume)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(marker, dict(format='stage7_owned_v1', config=cfg, data_identity=data.identity))
    deadline = (args.job_start_epoch or time.time()) + cfg['resource_limits']['maximum_job_seconds'] - 300
    interrupted = []
    signal.signal(signal.SIGTERM, lambda *_: interrupted.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: interrupted.append('SIGINT'))
    target = min(engine.horizon, args.review_updates or engine.horizon)
    if target < engine.updates:
        p.error('Review target precedes current checkpoint')
    reason = 'completed' if target == engine.horizon else 'review_gate'
    while engine.updates < target:
        if interrupted or time.time() >= deadline:
            reason = interrupted[0] if interrupted else 'walltime'
            break
        event = engine.update()
        with (directory / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
        if event['optimizer_step_succeeded']:
            if engine.updates % cfg['training']['eval_interval'] == 0 or engine.updates == target:
                evaluation = engine.evaluate()
                with (directory / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(evaluation, allow_nan=False) + '\n')
            if engine.updates % cfg['training']['save_interval'] == 0:
                engine.save(directory / 'latest.pt')
    engine.save(directory / 'latest.pt')
    write_json(directory / 'summary.json', dict(reason=reason, counters=engine.state()['counters']), exclusive=False)


if __name__ == '__main__':
    main()
