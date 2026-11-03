"""Focused CPU tests; no cluster allocation or training on preserved weights.

Includes fresh-process uninterrupted/restarted equivalence with dropout enabled.
Run: python3 -B stage_6_tinystories/tests/test_stage6.py
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE.parent))
import numpy as np
import torch
from stage_6_tinystories.core import (Engine, atomic_save, evaluate, load_checkpoint,
    restore_rng, rng_state, seed_all)
from stage_6_tinystories.data import TokenShards, read_json, sha256, validate_assets
from stage_6_tinystories.model import GPTLanguageModel
from stage_6_tinystories.train import Deadline, benchmark, safe_run_dir, train


def assert_nested_equal(case, a, b):
    if torch.is_tensor(a):
        case.assertTrue(torch.equal(a, b), 'Tensor mismatch')
    elif isinstance(a, dict):
        case.assertEqual(set(a), set(b))
        for key in a:
            assert_nested_equal(case, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested_equal(case, x, y)
    else:
        case.assertEqual(a, b)


class Stage6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shards = self.root/'data'
        self.shards.mkdir()
        self.tokens = ((np.arange(256)*7) % 29 + 3).astype('<u2')
        tokenizer = self.root/'tokenizer.json'
        tokenizer.write_text(json.dumps({'model': {'vocab': {chr(65+i): i for i in range(32)}}}))
        manifest = {'version': 'v2', 'dtype': 'uint16', 'byte_order': 'little-endian',
                    'vocab_size': 32, 'special_ids': {'pad': 0, 'bos': 1, 'eos': 2},
                    'tokenizer_sha256': sha256(tokenizer), 'splits': {}}
        for split in ('train', 'dev_val'):
            entries = []
            boundaries = [0, 19, 82, 182, 256] if split == 'train' else [0, 256]
            for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                path = self.shards/('{}_{}.bin'.format(split, i))
                self.tokens[a:b].tofile(str(path))
                entries.append({'file': path.name, 'tokens': b-a, 'sha256': sha256(path)})
            manifest['splits'][split] = {'token_count': 256, 'shards': entries}
        (self.shards/'manifest.json').write_text(json.dumps(manifest))
        (self.root/'prompts.json').write_text(json.dumps({
            'tokenizer_sha256': manifest['tokenizer_sha256'],
            'prompts': [{'id': 'manual', 'kind': 'manual_probe', 'ids': [1, 4, 5]}]}))
        self.cfg = {'experiment': 'test', 'model': {'vocab_size': 32, 'n_layer': 2,
            'n_embd': 16, 'n_head': 4, 'block_size': 8, 'dropout': .35, 'weight_tying': True},
            'data': {'shard_dir': 'data', 'tokenizer': 'tokenizer.json', 'pad_id': 0, 'bos_id': 1, 'eos_id': 2},
            'initial_checkpoint': 'stage5.pt',
            'training': {'additional_tokens': 192, 'batch_size': 2, 'seed': 1337,
                'start_lr': 3e-5, 'peak_lr': 1e-4, 'min_lr': 3e-5, 'warmup_updates': 2,
                'weight_decay': .1, 'betas': [.9, .95], 'grad_clip': 1., 'eval_interval': 3,
                'save_interval': 2, 'log_interval': 2, 'snapshot_tokens': [32, 80, 192]},
            'evaluation': {'seed': 2026, 'windows': 6, 'train_windows': 4, 'batch_size': 2,
                'prompts': 'prompts.json', 'generation_seeds': [1337], 'max_new_tokens': 5,
                'temperature': .7, 'top_k': 10},
            'benchmark': {'warmup_updates': 1, 'windows': 2, 'updates_per_window': 2},
            'safety': {'margin_seconds': 0, 'initial_eval_allowance_seconds': 0,
                       'initial_save_allowance_seconds': 0}}
        self.datasets, self.identity, _ = validate_assets(self.root, self.cfg, True)
        self.source = self.root/'stage5.pt'
        self.make_source()

    def make_source(self):
        source = Engine(self.cfg, torch.device('cpu'), self.datasets, self.identity)
        for _ in range(3):
            source.update()
        atomic_save({'step': 3, 'tokens': 48, 'val_loss': 3.4,
            'config': dict(self.cfg['model'], bos_id=1, eos_id=2),
            'model': source.model.state_dict(), 'optimizer': source.optimizer.state_dict()}, self.source)
        self.config_path = self.root/'config.json'
        self.config_path.write_text(json.dumps(self.cfg))

    def engine(self):
        engine = Engine(self.cfg, torch.device('cpu'), self.datasets, self.identity)
        engine.start_from_stage5(self.source)
        return engine

    def test_cross_shard_and_last_legal_window(self):
        ds = self.datasets['train']
        self.assertTrue(np.array_equal(ds.read(15, 100), self.tokens[15:115]))
        self.assertTrue(np.array_equal(ds.read(247, 9), self.tokens[247:256]))
        x, y = ds.batch(8, 8, torch.Generator().manual_seed(1), torch.device('cpu'))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertTrue(all(isinstance(a, np.memmap) for a in ds.arrays))

    def test_corrupt_shard_is_rejected(self):
        path = self.shards/'train_0.bin'
        with open(str(path), 'r+b') as f:
            f.write(b'\x00\x00')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            TokenShards(self.shards, 'train', True)

    def test_tokenizer_identity_is_required(self):
        (self.root/'tokenizer.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Tokenizer SHA256'):
            validate_assets(self.root, self.cfg)

    def test_tying_causality_and_finite_gradients(self):
        model = self.engine().model.eval()
        self.assertIs(model.lm_head.weight, model.token_embedding_table.weight)
        x = torch.tensor([[3, 4, 5, 6, 7, 8]])
        other = x.clone()
        other[:, 3:] = 11
        self.assertTrue(torch.equal(model(x)[0][:, :3], model(other)[0][:, :3]))
        _, loss = model(x, other)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_uninterrupted_equals_restarted_with_and_without_dropout(self):
        for dropout in (0., .35):
            with self.subTest(dropout=dropout):
                self.cfg['model']['dropout'] = dropout
                self.make_source()
                full = self.engine()
                full_losses = [full.update()[0].item() for _ in range(8)]
                expected = copy.deepcopy(full.checkpoint())
                interrupted = self.engine()
                losses = []
                for _ in range(3):
                    losses.append(interrupted.update()[0].item())
                    evaluate(interrupted)  # Extra evaluations must not change training RNG.
                restart = self.root/'restart.pt'
                atomic_save(interrupted.checkpoint(), restart)
                restored = Engine(self.cfg, torch.device('cpu'), self.datasets, self.identity)
                restored.resume(restart)
                for _ in range(5):
                    losses.append(restored.update()[0].item())
                    evaluate(restored)
                actual = restored.checkpoint()
                self.assertEqual(full_losses, losses)
                for key in ('model', 'optimizer', 'rng', 'completed_updates', 'tokens', 'additional_tokens'):
                    assert_nested_equal(self, expected[key], actual[key])

    def test_generation_and_evaluation_preserve_rng_and_mode(self):
        engine = self.engine()
        before = rng_state(engine.sampler)
        a = evaluate(engine)
        b = evaluate(engine)
        self.assertEqual(a, b)
        generator = torch.Generator().manual_seed(5)
        ids, reason = engine.model.generate(torch.tensor([[1, 5, 6]]), 8, generator,
                                            top_k=10, eos_id=None)
        self.assertFalse(any(t in (0, 1) for t in ids[0, 3:].tolist()))
        self.assertTrue(engine.model.training)
        engine.model.eval()
        engine.model.generate(torch.tensor([[1, 5]]), 2, torch.Generator().manual_seed(1))
        self.assertFalse(engine.model.training)
        assert_nested_equal(self, before, rng_state(engine.sampler))

    def test_lr_endpoints_and_integer_budget(self):
        engine = self.engine()
        engine.run_updates = 1
        self.assertEqual(engine.lr(), 1e-4)
        engine.run_updates = engine.total_run_updates-1
        self.assertAlmostEqual(engine.lr(), 3e-5)
        self.cfg['training']['additional_tokens'] = 193
        self.assertEqual(self.engine().total_run_updates, 13)

    def test_resume_rejects_changed_config_or_data(self):
        engine = self.engine()
        engine.update()
        path = self.root/'resume.pt'
        atomic_save(engine.checkpoint(), path)
        cfg = copy.deepcopy(self.cfg)
        cfg['training']['peak_lr'] = .0002
        other = Engine(cfg, torch.device('cpu'), self.datasets, self.identity)
        with self.assertRaisesRegex(ValueError, 'configuration/data'):
            other.resume(path)
        other = Engine(self.cfg, torch.device('cpu'), self.datasets, {'different': 'data'})
        with self.assertRaisesRegex(ValueError, 'configuration/data'):
            other.resume(path)

    def test_source_architecture_mismatch_rejected(self):
        self.cfg['model']['n_head'] = 2
        with self.assertRaisesRegex(ValueError, 'architecture'):
            self.engine()

    def test_source_hash_and_optimizer_settings_are_enforced(self):
        self.cfg['initial_checkpoint_sha256'] = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.engine()
        del self.cfg['initial_checkpoint_sha256']
        self.cfg['training']['betas'] = [.8, .95]
        with self.assertRaisesRegex(ValueError, 'hyperparameters'):
            self.engine()

    def test_nonfinite_checkpoint_is_rejected(self):
        engine = self.engine()
        with torch.no_grad():
            next(engine.model.parameters()).flatten()[0] = float('nan')
        with self.assertRaises(FloatingPointError):
            engine.checkpoint()

    def test_walltime_exit_is_resumable_without_updates(self):
        out = safe_run_dir(self.root, 'walltime')
        report = train(self.engine(), out, Deadline(.000001, 1))
        self.assertEqual(report['stage6_updates'], 0)
        self.assertIn('walltime', report['reason'])
        restored = Engine(self.cfg, torch.device('cpu'), self.datasets, self.identity)
        restored.resume(out/'latest.pt')
        self.assertEqual(restored.run_updates, 0)
        self.assertIsNone(restored.last_eval)

    def test_signal_exit_and_snapshot_accounting(self):
        engine = self.engine()
        deadline = Deadline(None, 0)
        deadline.signal = 'SIGTERM'
        out = safe_run_dir(self.root, 'signal')
        report = train(engine, out, deadline)
        self.assertEqual(report['reason'], 'SIGTERM')
        out2 = safe_run_dir(self.root, 'final')
        report = train(self.engine(), out2, Deadline(None, 0))
        ckpt = load_checkpoint(out2/'latest.pt')
        self.assertEqual(ckpt['stage6_updates'], 12)
        self.assertEqual(ckpt['completed_updates'], 15)
        self.assertEqual(ckpt['additional_tokens'], 192)
        self.assertEqual(ckpt['tokens'], 240)
        self.assertEqual(ckpt['last_eval']['stage6_updates'], 12)
        self.assertTrue((out2/'snapshot_additional_192.pt').exists())
        self.assertEqual(report['reason'], 'completed')

    def test_benchmark_only_and_full_checkpoint_probe(self):
        source_hash = sha256(self.source)
        out = safe_run_dir(self.root, 'benchmark')
        report = benchmark(self.engine(), out, Deadline(None, 0))
        self.assertEqual(len(report['windows']), 2)
        self.assertGreater(report['full_checkpoint_bytes'], 0)
        self.assertGreater(report['tokens_per_second'], 0)
        self.assertFalse(list(out.glob('*.pt')))
        self.assertEqual(sha256(self.source), source_hash)

    def test_restart_finishes_final_evaluation_and_snapshots_without_extra_update(self):
        engine = self.engine()
        engine.last_eval = evaluate(engine)
        engine.best_eval = engine.last_eval
        for _ in range(engine.total_run_updates):
            engine.update()
        out = safe_run_dir(self.root, 'pending_final')
        atomic_save(engine.checkpoint(), out/'latest.pt')
        restored = Engine(self.cfg, torch.device('cpu'), self.datasets, self.identity)
        restored.resume(out/'latest.pt')
        before = copy.deepcopy(restored.model.state_dict())
        train(restored, out, Deadline(None, 0))
        final = load_checkpoint(out/'latest.pt')
        self.assertEqual(final['stage6_updates'], 12)
        self.assertEqual(final['last_eval']['stage6_updates'], 12)
        assert_nested_equal(self, before, final['model'])
        self.assertTrue((out/'snapshot_additional_192.pt').exists())

    def test_run_directory_protects_stage5_and_existing_runs(self):
        for path in ('stage_5_tinystories', 'stage_5_tinystories/checkpoints', '.'):
            with self.assertRaises(ValueError):
                safe_run_dir(self.root, path)
        out = safe_run_dir(self.root, 'existing')
        (out/'latest.pt').write_text('original')
        with self.assertRaises(ValueError):
            safe_run_dir(self.root, 'existing')

    def cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(STAGE/'train.py'), '--config',
            str(self.config_path), '--project-root', str(self.root), '--device', 'cpu',
            '--cpu-threads', '1']+list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=90)

    def test_fresh_process_restart_matches_uninterrupted_dropout_training(self):
        commands = [('--run-dir', 'full', '--stop-after-updates', '8'),
                    ('--run-dir', 'split', '--stop-after-updates', '3'),
                    ('--run-dir', 'split', '--resume', 'split/latest.pt', '--stop-after-updates', '8')]
        for args in commands:
            completed = self.cli(*args)
            self.assertEqual(completed.returncode, 0, completed.stdout)
        a, b = [load_checkpoint(self.root/name/'latest.pt') for name in ('full', 'split')]
        for key in ('model', 'optimizer', 'rng', 'completed_updates', 'tokens'):
            assert_nested_equal(self, a[key], b[key])

    def test_preflight_cli_does_not_create_run_directory(self):
        result = self.cli('--mode', 'preflight', '--verify-shards', '--run-dir', 'must_not_exist')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('PREFLIGHT PASS', result.stdout)
        self.assertFalse((self.root/'must_not_exist').exists())


class RealStage5Compatibility(unittest.TestCase):
    def test_production_stage5_weights_logits_and_parameter_count(self):
        # Works after installation into the repository; staged tests supply this path.
        root = Path(os.environ.get('NANOGPT_TEST_PROJECT_ROOT', str(STAGE.parent)))
        source = root/'stage_5_tinystories/model.py'
        checkpoint = root/'stage_5_tinystories/checkpoints/best.pt'
        if not source.is_file() or not checkpoint.is_file():
            self.skipTest('Real Stage 5 source/checkpoint not available')
        spec = importlib.util.spec_from_file_location('stage5_compatibility_reference', str(source))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = read_json(STAGE/'configs/continue.json')
        old = module.GPTLanguageModel().eval()
        new = GPTLanguageModel(cfg['model']).eval()
        state = load_checkpoint(checkpoint)['model']
        old.load_state_dict(state, strict=True)
        new.load_state_dict(state, strict=True)
        self.assertEqual(sum(p.numel() for p in new.parameters()), 6897152)
        self.assertIs(new.lm_head.weight, new.token_embedding_table.weight)
        x = torch.tensor([[1, 430, 442, 261, 397, 14]])
        with torch.no_grad():
            self.assertTrue(torch.equal(old(x)[0], new(x)[0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
