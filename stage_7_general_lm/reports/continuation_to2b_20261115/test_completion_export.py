"""The original horizon completes normally; earlier and partial stops stay gated."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

helper_path = Path(os.environ.get('STAGE7_REVIEW_HELPER',
                   str(Path(__file__).resolve().parents[1] / 'hpc/review_segment.py')))
spec = importlib.util.spec_from_file_location('stage7_completion_export_helper', str(helper_path))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class CompletionExportTests(unittest.TestCase):
    def review_fixture(self, root, completed):
        from stage_7_general_lm.common import configure
        from stage_7_general_lm.data import TokenStream
        from stage_7_general_lm.train import Engine
        from stage_7_general_lm.tests.test_stage7 import setup_fixture
        configure('cpu', 2)
        cfg, manifest_path = setup_fixture(root, dropout=0)
        cfg['training'].update(token_budget=16, precision='fp32')
        manifest = helper.read(manifest_path)
        manifest['tokenizer_sha256'] = 'fixture-tokenizer'
        manifest['splits']['heldout'] = json.loads(json.dumps(manifest['splits']['dev']))
        for split in manifest['splits'].values():
            for shard in split['shards']:shard['source'] = 'fixture'
        manifest_path.write_text(json.dumps(manifest))
        config_path = root / 'config.json'
        config_path.write_text(json.dumps(cfg))
        engine = Engine(cfg, TokenStream(manifest_path), TokenStream(manifest_path, 'dev'))
        if completed:engine.update()
        checkpoint = root / 'checkpoint.pt'
        engine.save(checkpoint)
        record = root / 'record.json'
        record.write_text(json.dumps(dict(checkpoint_sha256=helper.digest(checkpoint),
            summary=dict(counters=engine.state()['counters']))))
        metrics = root / 'metrics.jsonl'
        metrics.write_text(json.dumps(engine.evaluate())+'\n')
        probes = root / 'probes.json'
        probes.write_text(json.dumps(dict(tokenizer_sha256='fixture-tokenizer',probes=[])))
        return SimpleNamespace(device='cpu', checkpoint=checkpoint, checkpoint_record=record,
            config=config_path, manifest=manifest_path, metrics=metrics, probes=probes,
            output=root/'review.json', windows=3, heldout_windows=3, reference_review=None)

    def test_heldout_is_blocked_before_completed_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.review_fixture(Path(directory), completed=False)
            with patch.object(helper, 'score_stream') as scoring:
                with self.assertRaisesRegex(ValueError, 'reserved for the original completed horizon'):
                    helper.review(args)
                scoring.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_completed_review_scores_heldout_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.review_fixture(Path(directory), completed=True)
            with patch.object(helper, 'TOKENIZER_SHA', 'fixture-tokenizer'):
                helper.review(args)
            result = helper.read(args.output)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['heldout_score']['positions'], 24)
            self.assertEqual(result['heldout_score']['seed'], 20261115)
            self.assertNotIn('heldout', result['expanded_scores'])
            self.assertTrue(result['original_evaluation_reproduced'])

    def fixture(self, root, updates=122071, reason='completed', target=122071,
                allow_completed=True):
        run, out = root / 'run', root / 'evidence'
        run.mkdir(); out.mkdir()
        (run / 'latest.pt').write_bytes(b'full-restart-checkpoint')
        (run / 'metrics.jsonl').write_text('{}\n')
        (run / 'summary.json').write_text(json.dumps(dict(reason=reason,
            counters=dict(updates=updates, successful_positions=updates * 16384))))
        config = root / 'original.json'
        config.write_text(json.dumps(dict(model=dict(block_size=1024),
            training=dict(micro_batch_size=4, accumulation_steps=4, token_budget=2000000000))))
        return SimpleNamespace(run=run, output=out, start_updates=114441,
            target_updates=target, completion_config=config if allow_completed else None)

    def test_full_original_horizon_exports_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            helper.finalize(args)
            record = helper.read(args.output / 'checkpoint.json')
            self.assertEqual(record['summary']['reason'], 'completed')
            self.assertEqual(record['summary']['counters']['successful_positions'], 2000011264)
            frozen = Path(record['checkpoint'])
            (args.run / 'next.pt').write_bytes(b'different-rolling-checkpoint')
            os.replace(str(args.run / 'next.pt'), str(args.run / 'latest.pt'))
            self.assertEqual(frozen.read_bytes(), b'full-restart-checkpoint')

    def test_completed_without_original_horizon_authority_stays_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), allow_completed=False)
            with self.assertRaisesRegex(RuntimeError, 'dependent training must not start'):
                helper.finalize(args)
            self.assertTrue((args.output / 'checkpoint.json').exists())

    def test_early_target_cannot_claim_original_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), updates=122070, target=122070)
            with self.assertRaisesRegex(ValueError, 'original configured horizon'):
                helper.finalize(args)
            self.assertFalse((args.output / 'checkpoint.json').exists())

    def test_partial_final_allocation_preserves_state_but_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), updates=122060, reason='walltime')
            with self.assertRaisesRegex(RuntimeError, 'dependent training must not start'):
                helper.finalize(args)
            record = helper.read(args.output / 'checkpoint.json')
            self.assertEqual(record['summary']['reason'], 'walltime')
            self.assertTrue(Path(record['checkpoint']).exists())


if __name__ == '__main__':
    unittest.main()
