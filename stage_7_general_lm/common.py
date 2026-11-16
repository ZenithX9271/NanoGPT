"""Small shared IO helpers. No Stage 6 source is edited or checkpoint resumed."""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, exclusive=True):
    path = Path(path)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def atomic_checkpoint(path, value):
    """Replace ONLY an owned rolling checkpoint; review exports use exclusive names."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
        if hasattr(os, 'O_DIRECTORY'):
            descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_checkpoint(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:  # Trusted project checkpoint on the old cluster runtime.
        return torch.load(str(path), map_location='cpu')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(device, threads=1):
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if device.startswith('cuda') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before starting the CUDA process')
    return torch.device(device)


def runtime(device='cpu'):
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__),
                platform=platform.platform(), device=str(device), threads=torch.get_num_threads(),
                cuda_build=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None)


def source_identity():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def rng_state(sampler):
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                cpu=torch.get_rng_state(), sampler=sampler.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, sampler):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['cpu'])
    sampler.set_state(state['sampler'])
    if len(state['cuda']) != torch.cuda.device_count():
        raise ValueError('CUDA device count differs from saved RNG state')
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
