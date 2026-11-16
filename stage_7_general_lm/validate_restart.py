"""Exact uninterrupted-versus-resumed next-update validation on real runtime/data."""
import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

import torch

from .common import (configure, load_checkpoint, read_json, runtime, sha,
                     write_json)
from .data import TokenStream
from .train import Engine


def assert_equal(left, right, path='root'):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError('Tensor differs at ' + path)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError('Dictionary keys differ at ' + path)
        for key in left:
            assert_equal(left[key], right[key], path + '.' + str(key))
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise ValueError('Sequence differs at ' + path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, path + '[' + str(index) + ']')
    elif left != right:
        raise ValueError('Value differs at ' + path)


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate(config_path, manifest_path, work_dir, device='cuda', threads=4):
    started = time.monotonic()
    config_path, manifest_path = Path(config_path), Path(manifest_path)
    cfg = read_json(config_path)
    device = configure(device, threads)
    train_data = TokenStream(manifest_path)
    dev_data = TokenStream(manifest_path, 'dev')
    work_dir = Path(work_dir).resolve()
    if not work_dir.is_dir():
        raise ValueError('Existing quota-checked work directory required')

    with tempfile.TemporaryDirectory(prefix='stage7_restart_', dir=str(work_dir)) as temporary:
        temporary = Path(temporary)
        uninterrupted_path = temporary / 'uninterrupted_u2.pt'
        boundary_path = temporary / 'boundary_u1.pt'
        resumed_path = temporary / 'resumed_u2.pt'

        uninterrupted = Engine(cfg, train_data, dev_data, device)
        uninterrupted_first = uninterrupted.update()
        uninterrupted_second = uninterrupted.update()
        if not (uninterrupted_first['optimizer_step_succeeded'] and
                uninterrupted_second['optimizer_step_succeeded']):
            raise ValueError('Restart validation requires successful FP16 optimizer updates')
        uninterrupted.save(uninterrupted_path)
        uninterrupted_size = uninterrupted_path.stat().st_size
        del uninterrupted
        release()

        boundary = Engine(cfg, train_data, dev_data, device)
        boundary_first = boundary.update()
        assert_equal(uninterrupted_first, boundary_first, 'first_update_event')
        boundary.save(boundary_path)
        boundary_size = boundary_path.stat().st_size
        del boundary
        release()

        resumed = Engine(cfg, train_data, dev_data, device)
        resumed.resume(boundary_path)
        resumed_second = resumed.update()
        if not resumed_second['optimizer_step_succeeded']:
            raise ValueError('Resumed FP16 optimizer update was skipped')
        assert_equal(uninterrupted_second, resumed_second, 'second_update_event')
        resumed.save(resumed_path)
        resumed_size = resumed_path.stat().st_size
        del resumed
        release()

        uninterrupted_state = load_checkpoint(uninterrupted_path)
        resumed_state = load_checkpoint(resumed_path)
        assert_equal(uninterrupted_state, resumed_state, 'checkpoint')
        report = {
            'format': 'stage7_exact_restart_validation_v1',
            'status': 'PASS',
            'semantic_checkpoint_equality': True,
            'first_update_event_equality': True,
            'next_update_event_equality': True,
            'config_sha256': sha(config_path),
            'manifest_sha256': sha(manifest_path),
            'code_identity': uninterrupted_state['code_identity'],
            'runtime': runtime(device),
            'positions_per_update': (cfg['training']['micro_batch_size'] *
                                     cfg['training']['accumulation_steps'] *
                                     cfg['model']['block_size']),
            'uninterrupted_checkpoint_bytes': uninterrupted_size,
            'boundary_checkpoint_bytes': boundary_size,
            'resumed_checkpoint_bytes': resumed_size,
            'uninterrupted_checkpoint_sha256': sha(uninterrupted_path),
            'boundary_checkpoint_sha256': sha(boundary_path),
            'resumed_checkpoint_sha256': sha(resumed_path),
            'first_update': uninterrupted_first,
            'second_update': uninterrupted_second,
            'seconds': time.monotonic() - started,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    result = validate(args.config, args.manifest, args.work_dir,
                      args.device, args.threads)
    write_json(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
