"""Bounded runtime probe. A CPU result never certifies a GPU or external model."""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import subprocess
from pathlib import Path

import torch

from .common import configure, runtime, write_json
from .model import LanguageModel


def probe(device='cpu', threads=1):
    device = configure(device, threads)
    result = dict(observed=runtime(device), python_executable=sys.executable,
                  machine=platform.machine(), libc=platform.libc_ver(),
                  optional_modules={m: importlib.util.find_spec(m) is not None for m in ('transformers', 'peft', 'datasets', 'tokenizers', 'bitsandbytes')},
                  quantized_engines=torch.backends.quantized.supported_engines,
                  selected_quantized_engine=torch.backends.quantized.engine,
                  cuda_device_count=torch.cuda.device_count(),
                  gpu_allocation_observed=os.environ.get('PBS_JOBID'),
                  disk_free_bytes=shutil.disk_usage(Path.cwd()).free,
                  unknown=['HPC driver/OS/quotas when probing locally', 'A100 availability and GPU-hour allocation', 'External 1.7B checkpoint load/adapters', 'CPU or GPU INT4 acceleration'])
    try:
        import psutil
        result['observed']['host_ram_bytes'] = psutil.virtual_memory().total
        result['observed']['available_host_ram_bytes'] = psutil.virtual_memory().available
    except ImportError:
        result['observed']['host_ram_bytes'] = None
        result['unknown'].append('Host RAM: optional psutil unavailable')
    result['failed'] = []
    if shutil.which('nvidia-smi'):
        query = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        result['observed']['nvidia_smi_query'] = query.stdout.strip()
        if query.returncode:
            result['failed'].append(dict(check='nvidia-smi', reason=query.stderr.strip()))
    if device.type == 'cuda':
        result['observed']['compute_capability'] = list(torch.cuda.get_device_capability(device))
        result['observed']['vram_bytes'] = torch.cuda.get_device_properties(device).total_memory
    cfg = dict(vocab_size=64, n_layer=2, n_embd=32, n_head=4, block_size=32, intermediate_size=64, dropout=.1)
    model = LanguageModel(cfg).to(device)
    ids = torch.arange(8, device=device).reshape(1, 8)
    _, loss = model(ids, (ids + 1) % 64)
    loss.backward()
    result['production_class_small_config_forward_backward'] = bool(torch.isfinite(loss))
    result['probe_is_not_candidate_throughput'] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    report = probe(args.device, args.threads)
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
