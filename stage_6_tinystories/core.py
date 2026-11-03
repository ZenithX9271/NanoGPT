"""Continuation state, independent samplers, fixed evaluation and full checkpoints."""
import copy
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import sha256
from .model import GPTLanguageModel

FORMAT = 'nanogpt_stage6_v1'


def load_checkpoint(path):
    # Stage 5 and Stage 6 contain tensors/basic containers, not pickled model objects.
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:  # IITD PyTorch 1.10 has no weights_only argument.
        return torch.load(str(path), map_location='cpu')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device(name, threads=4):
    torch.set_num_threads(threads)
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: obtain a GPU allocation and load the IITD module')
    device = torch.device('cuda:0' if name == 'cuda' else name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def runtime_info(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': str(np.__version__), 'cuda': torch.version.cuda,
            'platform': platform.platform(), 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'threads': torch.get_num_threads(), 'precision': 'float32',
            'attention': 'manual causal',
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def host_peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024.0*1024.0 if sys.platform == 'darwin' else 1024.0)


def code_identity():
    folder = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(folder.glob('*.py'))}


def git_identity(root):
    def call(args):
        try:
            return subprocess.check_output(['git']+args, cwd=str(root),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable'
    return {'head': call(['rev-parse', 'HEAD']), 'status': call(['status', '--short'])}


def rng_state(generator):
    ns = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'train_sampler': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('Exact restart needs the same visible CUDA device count')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    generator.set_state(state['train_sampler'])


def make_optimizer(model, cfg):
    # Same deduplicated group membership and order as Stage 5.
    params = list(model.parameters())
    return torch.optim.AdamW([
        {'params': [p for p in params if p.dim() >= 2], 'weight_decay': cfg['weight_decay']},
        {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}],
        lr=cfg['start_lr'], betas=tuple(cfg['betas']))


def validate_optimizer(optimizer, expected_updates):
    steps = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            state = optimizer.state.get(param, {})
            if not state:
                if expected_updates:
                    raise ValueError('Missing optimizer moments')
                continue
            for key in ('exp_avg', 'exp_avg_sq'):
                if state[key].shape != param.shape or not torch.isfinite(state[key]).all().item():
                    raise ValueError('Invalid optimizer moment: ' + key)
            value = state['step']
            steps.add(int(value.item() if torch.is_tensor(value) else value))
    if expected_updates and steps != {expected_updates}:
        raise ValueError('Optimizer completed-update counters disagree: ' + str(steps))


def assert_tied_state(state):
    if not torch.equal(state['token_embedding_table.weight'], state['lm_head.weight']):
        raise ValueError('Checkpoint embedding and output weights are not tied consistently')


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        with open(str(temporary), 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(value, path):
    with open(str(path), 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(cfg):
    m, t, e, b = (cfg[k] for k in ('model', 'training', 'evaluation', 'benchmark'))
    if not m['weight_tying'] or m['n_embd'] % m['n_head'] or not 0 <= m['dropout'] < 1:
        raise ValueError('Invalid model configuration')
    for section, names in [(m, ('vocab_size', 'n_layer', 'n_embd', 'n_head', 'block_size')),
                           (t, ('additional_tokens', 'batch_size', 'eval_interval', 'save_interval', 'log_interval')),
                           (e, ('windows', 'batch_size')), (b, ('windows', 'updates_per_window'))]:
        if any(not isinstance(section[n], int) or section[n] < 1 for n in names):
            raise ValueError('Counts and intervals must be positive integers')
    if t['warmup_updates'] < 0 or b['warmup_updates'] < 0 or e['train_windows'] < 0:
        raise ValueError('Warmup/window counts cannot be negative')
    if not 0 < t['start_lr'] <= t['peak_lr'] or not 0 < t['min_lr'] <= t['peak_lr']:
        raise ValueError('Invalid learning rate range')
    if t['grad_clip'] <= 0 or cfg['safety']['margin_seconds'] < 0:
        raise ValueError('Invalid clipping or safety margin')
    ids = [cfg['data'][n+'_id'] for n in ('pad', 'bos', 'eos')]
    if len(set(ids)) != 3 or any(i < 0 or i >= m['vocab_size'] for i in ids):
        raise ValueError('Invalid special IDs')


class Engine:
    def __init__(self, cfg, device, datasets, identity):
        validate_config(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.device, self.datasets = device, datasets
        self.identity = dict(identity)
        self.code = code_identity()
        seed_all(cfg['training']['seed'])
        self.model = GPTLanguageModel(cfg['model']).to(device=device, dtype=torch.float32)
        self.optimizer = make_optimizer(self.model, cfg['training'])
        self.sampler = torch.Generator().manual_seed(cfg['training']['seed']+1)
        self.tokens_per_update = cfg['training']['batch_size']*cfg['model']['block_size']
        self.total_run_updates = (cfg['training']['additional_tokens']+self.tokens_per_update-1)//self.tokens_per_update
        self.run_updates = 0
        self.base_updates = 0
        self.base_tokens = 0
        self.best_eval, self.last_eval = None, None
        self.saved_snapshots = []
        self.origin = None

    def check_optimizer_settings(self):
        for index, group in enumerate(self.optimizer.param_groups):
            expected_decay = self.cfg['training']['weight_decay'] if index == 0 else 0.
            if tuple(group['betas']) != tuple(self.cfg['training']['betas']) or group['weight_decay'] != expected_decay:
                raise ValueError('Loaded optimizer hyperparameters disagree with config')

    def start_from_stage5(self, path):
        source_hash = sha256(path)
        if self.cfg.get('initial_checkpoint_sha256', source_hash) != source_hash:
            raise ValueError('Initial checkpoint SHA256 differs from the pinned Stage 5 baseline')
        ckpt = load_checkpoint(path)
        if ckpt.get('format') == FORMAT:
            raise ValueError('Use --resume for a Stage 6 checkpoint')
        for key, value in self.cfg['model'].items():
            if ckpt['config'].get(key) != value:
                raise ValueError('Stage 6A preserves Stage 5 architecture; mismatch: ' + key)
        for key in ('bos_id', 'eos_id'):
            if ckpt['config'].get(key) != self.cfg['data'][key]:
                raise ValueError('Checkpoint special-ID mismatch')
        assert_tied_state(ckpt['model'])
        for tensor in ckpt['model'].values():
            if torch.is_floating_point(tensor) and (tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item()):
                raise ValueError('Source weights must be finite FP32 tensors')
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.base_updates = int(ckpt['step'])
        self.base_tokens = int(ckpt['tokens'])
        validate_optimizer(self.optimizer, self.base_updates)
        self.check_optimizer_settings()
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('FP32 parameters required')
        self.origin = {'path': str(Path(path).resolve()), 'sha256': source_hash,
                       'completed_updates': self.base_updates, 'tokens': self.base_tokens,
                       'saved_val_loss': float(ckpt['val_loss']),
                       'initialisation': 'Stage 5 weights + AdamW moments; new RNG and LR schedule'}
        # Define a fresh, explicit Stage 6 random trajectory after model construction.
        seed_all(self.cfg['training']['seed'])
        self.model.train()

    def resume(self, path):
        ckpt = load_checkpoint(path)
        if ckpt.get('format') != FORMAT:
            raise ValueError('Exact Stage 6 resume requires a Stage 6 checkpoint')
        if ckpt['config'] != self.cfg or ckpt['data_identity'] != self.identity:
            raise ValueError('Resume configuration/data identity changed; create an explicit new experiment')
        if ckpt['code_identity'] != self.code:
            raise ValueError('Training code changed since checkpoint; exact resume refused')
        assert_tied_state(ckpt['model'])
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.run_updates = ckpt['stage6_updates']
        self.base_updates = ckpt['base_updates']
        self.base_tokens = ckpt['base_tokens']
        if not 0 <= self.run_updates <= self.total_run_updates:
            raise ValueError('Completed updates outside configured schedule')
        if ckpt['completed_updates'] != self.base_updates+self.run_updates or ckpt['tokens'] != self.tokens:
            raise ValueError('Inconsistent checkpoint update/token accounting')
        validate_optimizer(self.optimizer, self.base_updates+self.run_updates)
        self.check_optimizer_settings()
        self.best_eval, self.last_eval = ckpt['best_eval'], ckpt['last_eval']
        self.saved_snapshots, self.origin = ckpt['saved_snapshots'], ckpt['origin']
        restore_rng(ckpt['rng'], self.sampler)
        self.model.train()
        return ckpt['runtime']

    @property
    def additional_tokens(self):
        return self.run_updates*self.tokens_per_update

    @property
    def tokens(self):
        return self.base_tokens+self.additional_tokens

    def lr(self):
        t = self.cfg['training']
        warmup = min(t['warmup_updates'], self.total_run_updates)
        if self.run_updates < warmup:
            return t['start_lr']+(t['peak_lr']-t['start_lr'])*(self.run_updates+1)/warmup
        progress = min(1., (self.run_updates-warmup)/max(1, self.total_run_updates-warmup-1))
        return t['min_lr']+0.5*(t['peak_lr']-t['min_lr'])*(1+math.cos(math.pi*progress))

    def update(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr()
        x, y = self.datasets['train'].batch(self.cfg['training']['batch_size'],
                                          self.cfg['model']['block_size'], self.sampler, self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['training']['grad_clip'])
        self.optimizer.step()
        self.run_updates += 1
        # No .item() or explicit synchronization in the measured update path.
        return loss.detach(), norm.detach()

    def checkpoint(self):
        tensors = list(self.model.parameters())
        tensors.extend(value for state in self.optimizer.state.values() for key, value in state.items()
                       if key in ('exp_avg', 'exp_avg_sq'))
        if not torch.stack([torch.isfinite(t).all() for t in tensors]).all().item():
            raise FloatingPointError('Refusing to overwrite a checkpoint with nonfinite weights/moments')
        return {'format': FORMAT, 'config': copy.deepcopy(self.cfg),
                'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'base_updates': self.base_updates, 'stage6_updates': self.run_updates,
                'completed_updates': self.base_updates+self.run_updates,
                'base_tokens': self.base_tokens, 'additional_tokens': self.additional_tokens,
                'tokens': self.tokens, 'last_eval': self.last_eval, 'best_eval': self.best_eval,
                'saved_snapshots': list(self.saved_snapshots), 'origin': self.origin,
                'rng': rng_state(self.sampler), 'data_identity': self.identity,
                'code_identity': self.code, 'runtime': runtime_info(self.device)}


class EvaluationInterrupted(Exception):
    pass


@torch.no_grad()
def evaluate(engine, stop=None):
    was_training = engine.model.training
    engine.model.eval()
    ec = engine.cfg['evaluation']
    result = {'stage6_updates': engine.run_updates, 'tokens': engine.tokens,
              'additional_tokens': engine.additional_tokens, 'seed': ec['seed'],
              'context': engine.cfg['model']['block_size'], 'batch_size': ec['batch_size']}
    try:
        for split, windows in [('train', ec['train_windows']), ('dev_val', ec['windows'])]:
            if not windows:
                continue
            generator = torch.Generator().manual_seed(ec['seed']+(1 if split == 'train' else 0))
            total_loss, count = 0., 0
            for begin in range(0, windows, ec['batch_size']):
                if stop is not None and stop():
                    raise EvaluationInterrupted()
                n = min(ec['batch_size'], windows-begin)
                x, y = engine.datasets[split].batch(n, engine.cfg['model']['block_size'], generator, engine.device)
                _, loss = engine.model(x, y)
                tokens = n*engine.cfg['model']['block_size']
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite evaluation loss')
                total_loss += value*tokens
                count += tokens
            result[split] = {'ce': total_loss/count, 'scored_tokens': count,
                             'perplexity': math.exp(min(700., total_loss/count))}
    finally:
        engine.model.train(was_training)
    return result
