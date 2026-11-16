"""Review tools must preserve frozen training identity and score every window."""
import math
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from stage_7_general_lm.common import configure, source_identity
from stage_7_general_lm.data import TokenStream
from stage_7_general_lm.hpc.review_segment import finalize, score_stream, sampled_completion, compare_reference
from stage_7_general_lm.model import LanguageModel
from stage_7_general_lm.tests.test_stage7 import setup_fixture


class ReviewTests(unittest.TestCase):
    def test_reference_comparison_rejects_changed_probes_and_regression(self):
        import copy
        reference = dict(status='PASS', config_sha256='config', manifest_sha256='data',
            code_identity={'model.py': 'code'}, tokenizer_sha256='tokenizer', precision='fp32',
            checkpoint_sha256='old-checkpoint', expanded_scores={
                'dev': dict(ce=3.5, seed=1, windows=512, positions=524288),
                'dev_domain:a': dict(ce=3.4, seed=2, windows=512, positions=524288)})
        current = copy.deepcopy(reference)
        current['expanded_scores']['dev']['ce'] = 3.4
        self.assertAlmostEqual(compare_reference(current, reference)['ce_changes']['dev'], -.1)
        for field in ('seed', 'windows', 'positions'):
            bad = copy.deepcopy(current)
            bad['expanded_scores']['dev'][field] += 1
            with self.assertRaisesRegex(ValueError, 'sampling differs'):
                compare_reference(bad, reference)
        bad = copy.deepcopy(current)
        bad['manifest_sha256'] = 'other-data'
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            compare_reference(bad, reference)
        bad = copy.deepcopy(current)
        bad['expanded_scores']['dev_domain:a']['ce'] += .1
        with self.assertRaisesRegex(ValueError, 'regression exceeds'):
            compare_reference(bad, reference)

    def test_remainder_windows_and_independent_sampler(self):
        configure('cpu', 1)
        with tempfile.TemporaryDirectory() as temporary:
            cfg, manifest = setup_fixture(Path(temporary), dropout=0)
            model = LanguageModel(cfg['model']).eval()
            data = TokenStream(manifest, 'dev')
            first = score_stream(model, data, 3, 2, 8, 'cpu', 123)
            second = score_stream(model, data, 3, 1, 8, 'cpu', 123)
            self.assertEqual(first['positions'], 24)
            self.assertAlmostEqual(first['ce'], second['ce'], places=5)
            self.assertAlmostEqual(first['perplexity'], math.exp(first['ce']))

    def test_sampled_cache_matches_full_prefix_sampling(self):
        configure('cpu', 1)
        with tempfile.TemporaryDirectory() as temporary:
            cfg, _ = setup_fixture(Path(temporary), dropout=0)
            model = LanguageModel(cfg['model']).eval()
            ids = torch.tensor([[1, 3, 7]])
            actual = sampled_completion(model, ids, 4, eos_id=-1, seed=77)
            torch.manual_seed(77)
            expected = ids
            with torch.no_grad():
                for _ in range(4):
                    logits, _ = model(expected)
                    values, indices = torch.topk(logits[:, -1].float() / .8, 50, dim=-1)
                    choice = torch.multinomial(torch.softmax(values, dim=-1), 1)
                    expected = torch.cat((expected, indices.gather(-1, choice)), dim=1)
            self.assertTrue(torch.equal(actual, expected))
            with self.assertRaises(ValueError):
                sampled_completion(model, ids, 20, eos_id=2, seed=77)

    def test_helpers_are_outside_frozen_source_identity(self):
        self.assertNotIn('review_segment.py', source_identity())

    def test_endpoint_export_survives_next_rolling_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, evidence = Path(temporary) / 'run', Path(temporary) / 'evidence'
            run.mkdir(); evidence.mkdir()
            (run / 'latest.pt').write_bytes(b'completed-200m-checkpoint')
            (run / 'metrics.jsonl').write_text('{}\n')
            (run / 'summary.json').write_text(json.dumps(dict(reason='review_gate',
                counters=dict(updates=12208, successful_positions=200015872))))
            finalize(SimpleNamespace(run=run, output=evidence))
            frozen = run / 'review_u12208_positions_200015872.pt'
            (run / 'next.pt').write_bytes(b'later-checkpoint')
            os.replace(str(run / 'next.pt'), str(run / 'latest.pt'))
            self.assertEqual(frozen.read_bytes(), b'completed-200m-checkpoint')
            self.assertTrue((evidence / 'checkpoint.json').is_file())

    def test_partial_segment_is_preserved_but_blocks_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, evidence = Path(temporary) / 'run', Path(temporary) / 'evidence'
            run.mkdir(); evidence.mkdir()
            (run / 'latest.pt').write_bytes(b'partial-checkpoint')
            (run / 'metrics.jsonl').write_text('{}\n')
            (run / 'summary.json').write_text(json.dumps(dict(reason='walltime',
                counters=dict(updates=13000, successful_positions=13000 * 16384))))
            with self.assertRaisesRegex(RuntimeError, 'target not reached'):
                finalize(SimpleNamespace(run=run, output=evidence,
                    start_updates=12208, target_updates=21363))
            self.assertEqual((run / 'review_u13000_positions_212992000.pt').read_bytes(),
                             b'partial-checkpoint')
            self.assertTrue((evidence / 'checkpoint.json').is_file())


if __name__ == '__main__':
    unittest.main()
