"""Real production classes at bounded sizes; never updates preserved checkpoints."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stage_7_general_lm.common import configure, load_checkpoint, read_json, sha
from stage_7_general_lm.corpus import sample, storage_dtype
from stage_7_general_lm.data import TokenStream, assert_disjoint
from stage_7_general_lm.evaluation import grouped_bootstrap, prompt, rows, score, validate_cases
from stage_7_general_lm.model import LanguageModel
from stage_7_general_lm.pretrained import collate, response_example
from stage_7_general_lm.train import Engine

ROOT = Path(__file__).resolve().parents[2]
NATIVE_TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json',
                          'special_tokens_map.json', 'config.json')


def native_tokenizer_directory(environ=None):
    """An optional offline integration fixture; explicit configuration fails closed."""
    environ = os.environ if environ is None else environ
    required = environ.get('STAGE7_REQUIRE_NATIVE_TOKENIZER') == '1'
    if 'STAGE7_TOKENIZER_DIR' not in environ:
        message = 'STAGE7_TOKENIZER_DIR unset; native-template compatibility unverified'
        if required:
            raise AssertionError(message + ' (required by STAGE7_REQUIRE_NATIVE_TOKENIZER=1)')
        raise unittest.SkipTest(message)
    value = environ['STAGE7_TOKENIZER_DIR'].strip()
    if not value:
        raise AssertionError('STAGE7_TOKENIZER_DIR must name a nonempty existing directory')
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    if not directory.is_dir():
        raise AssertionError('Native tokenizer directory missing: ' + str(directory))
    lock = read_json(ROOT / 'stage_7_general_lm/configs/native_tokenizer.lock.json')
    source = read_json(ROOT / 'stage_7_general_lm/configs/sources.json')
    if (lock.get('format') != 'stage7_native_tokenizer_bundle_v1'
            or lock.get('model_id') != source['tokenizer_model']
            or lock.get('revision') != source['tokenizer_revision']
            or set(lock['files']) != set(NATIVE_TOKENIZER_FILES)
            or lock['files']['tokenizer.json']['sha256'] != source['tokenizer_sha256']):
        raise AssertionError('Native tokenizer lock identity differs from pinned source')
    missing = [name for name in NATIVE_TOKENIZER_FILES if not (directory / name).is_file()]
    if missing:
        raise AssertionError('Incomplete native tokenizer bundle: ' + ', '.join(missing))
    for name in NATIVE_TOKENIZER_FILES:
        path, expected = directory / name, lock['files'][name]
        if path.stat().st_size != expected['bytes']:
            raise AssertionError('Native tokenizer size mismatch: ' + name)
        if sha(path) != expected['sha256']:
            raise AssertionError('Native tokenizer SHA256 mismatch: ' + name)
    return directory


class NativeTokenizerFixtureTests(unittest.TestCase):
    """Policy and rejection tests need no tokenizer cache or Transformers import."""

    def test_optional_unset_fixture_skips(self):
        with self.assertRaisesRegex(unittest.SkipTest, 'STAGE7_TOKENIZER_DIR unset'):
            native_tokenizer_directory({})

    def test_strict_unset_fixture_fails(self):
        with self.assertRaisesRegex(AssertionError, 'required by STAGE7_REQUIRE_NATIVE_TOKENIZER=1'):
            native_tokenizer_directory({'STAGE7_REQUIRE_NATIVE_TOKENIZER': '1'})

    def test_explicit_invalid_directory_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            for strict in ('0', '1'):
                for value in ('', str(Path(temporary) / 'missing')):
                    with self.subTest(strict=strict, value=value):
                        with self.assertRaisesRegex(AssertionError, 'nonempty existing directory|directory missing'):
                            native_tokenizer_directory(dict(STAGE7_TOKENIZER_DIR=value,
                                                            STAGE7_REQUIRE_NATIVE_TOKENIZER=strict))

    def test_incomplete_bundle_fails(self):
        for missing in NATIVE_TOKENIZER_FILES:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for name in NATIVE_TOKENIZER_FILES:
                    if name != missing:
                        (directory / name).write_bytes(b'{}')
                for strict in ('0', '1'):
                    with self.subTest(missing=missing, strict=strict):
                        with self.assertRaisesRegex(AssertionError, 'Incomplete.*' + missing):
                            native_tokenizer_directory(dict(STAGE7_TOKENIZER_DIR=temporary,
                                                            STAGE7_REQUIRE_NATIVE_TOKENIZER=strict))

    def test_wrong_size_fails(self):
        self.check_invalid_content(wrong_size=True)

    def test_checksum_mismatch_fails(self):
        self.check_invalid_content(wrong_size=False)

    def check_invalid_content(self, wrong_size):
        lock = read_json(ROOT / 'stage_7_general_lm/configs/native_tokenizer.lock.json')
        with tempfile.TemporaryDirectory() as temporary:
            for name, expected in lock['files'].items():
                # Deliberately invalid disposable files; never modify the real bundle.
                size = expected['bytes'] + int(wrong_size and name == 'tokenizer.json')
                (Path(temporary) / name).write_bytes(b'\0' * size)
            for strict in ('0', '1'):
                with self.subTest(strict=strict):
                    error = 'size mismatch' if wrong_size else 'SHA256 mismatch'
                    with self.assertRaisesRegex(AssertionError, error + ': tokenizer.json'):
                        native_tokenizer_directory(dict(STAGE7_TOKENIZER_DIR=temporary,
                                                        STAGE7_REQUIRE_NATIVE_TOKENIZER=strict))


class NativeTokenizerBundleIntegrityTests(unittest.TestCase):
    def test_each_file_rejects_modified_content_and_size(self):
        original = native_tokenizer_directory()
        for name in NATIVE_TOKENIZER_FILES:
            for change in ('checksum', 'size'):
                with self.subTest(file=name, change=change), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    for filename in NATIVE_TOKENIZER_FILES:
                        (directory / filename).write_bytes((original / filename).read_bytes())
                    target = directory / name
                    content = target.read_bytes()
                    content = (bytes([content[0] ^ 1]) + content[1:]) if change == 'checksum' else content + b'\n'
                    target.write_bytes(content)
                    error = 'SHA256 mismatch' if change == 'checksum' else 'size mismatch'
                    with self.assertRaisesRegex(AssertionError, error + ': ' + name):
                        native_tokenizer_directory({'STAGE7_TOKENIZER_DIR': temporary})


def setup_fixture(directory, dropout=.25):
    directory = Path(directory)
    manifest = dict(vocab_size=64, dtype='<u2', splits={})
    for split in ('train', 'dev'):
        a = (np.arange(800) % 61 + 3).astype('<u2')
        path = directory / (split + '.bin')
        path.write_bytes(a.tobytes())
        manifest['splits'][split] = dict(shards=[dict(file=path.name, tokens=len(a), sha256=sha(path))])
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    cfg = read_json(ROOT / 'stage_7_general_lm/configs/candidate.json')
    cfg['model'].update(vocab_size=64, n_layer=2, n_embd=32, n_head=4, intermediate_size=64, block_size=8, dropout=dropout)
    cfg['training'].update(token_budget=128, micro_batch_size=1, accumulation_steps=2, warmup_updates=2, grad_clip=1000000.)
    cfg['evaluation'].update(windows=2, batch_size=1)
    (directory / 'config.json').write_text(json.dumps(cfg))
    return cfg, directory / 'manifest.json'


def engine(cfg, path):
    return Engine(cfg, TokenStream(path), TokenStream(path, 'dev'))


def equal(case, a, b):
    if torch.is_tensor(a):
        case.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        case.assertEqual(set(a), set(b))
        for k in a:
            equal(case, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal(case, x, y)
    else:
        case.assertEqual(a, b)


class Stage7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure('cpu', 1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg, self.manifest = setup_fixture(self.path)

    def test_causality_and_tied_embedding(self):
        m = LanguageModel(self.cfg['model']).eval()
        a = torch.arange(8).reshape(1, 8)
        b = a.clone(); b[:, 5:] = 30
        self.assertTrue(torch.equal(m(a)[0][:, :5], m(b)[0][:, :5]))
        self.assertIs(m.embedding.weight, m.lm_head.weight)
        self.assertFalse(list(m.buffers()))

    def test_real_candidate_parameter_count(self):
        cfg = read_json(ROOT / 'stage_7_general_lm/configs/candidate.json')['model']
        device = torch.device('meta')
        if hasattr(device, '__enter__'):
            with device:
                m = LanguageModel(cfg)
        else:  # PyTorch 1.10 has a meta device but no device context manager.
            m = LanguageModel(cfg)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 122702592)

    def test_cache_matches_full_and_overflow_refused(self):
        m = LanguageModel(self.cfg['model']).eval()
        ids = torch.arange(8).reshape(1, 8)
        _, _, cache = m(ids[:, :4], use_cache=True)
        logits, _, cache = m(ids[:, 4:], past=cache, use_cache=True)
        torch.testing.assert_close(logits, m(ids)[0][:, 4:], atol=2e-7, rtol=1e-6)
        with self.assertRaises(ValueError):m(ids[:, :1], past=cache, use_cache=True)

    def test_padding_key_mask_and_ignored_labels(self):
        m = LanguageModel(self.cfg['model']).eval()
        x = torch.tensor([[4, 5, 6, 2, 2]])
        y = torch.tensor([[5, 6, 2, -100, -100]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        a, loss = m(x, y, attention_mask=mask)
        x[:, 3:] = 50
        b, changed = m(x, y, attention_mask=mask)
        self.assertTrue(torch.equal(a[:, :3], b[:, :3]))
        self.assertTrue(torch.equal(loss, changed))
        _, none = m(x, torch.full_like(x, -100), attention_mask=mask)
        self.assertEqual(none.item(), 0)

    def test_token_bounds_and_cache_mask_bounds(self):
        m = LanguageModel(self.cfg['model'])
        for token in (-1, 64):
            with self.assertRaises(ValueError):m(torch.tensor([[token]]))
        with self.assertRaises(ValueError):m(torch.tensor([[3]]), torch.tensor([[64]]))
        with self.assertRaises(ValueError):m(torch.tensor([[3]]), attention_mask=torch.ones(1, 2))

    def test_accumulation_weights_valid_positions(self):
        cfg = copy.deepcopy(self.cfg); cfg['model']['dropout'] = 0.
        a = engine(cfg, self.manifest)
        other = copy.deepcopy(cfg); other['training'].update(micro_batch_size=2, accumulation_steps=1)
        b = engine(other, self.manifest)
        x = torch.arange(16).reshape(2, 8) + 3
        y = x + 1; y[1, 2:] = -100
        a.update([(x[:1], y[:1]), (x[1:], y[1:])]); b.update([(x, y)])
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(p, q, atol=3e-6, rtol=1e-5)

    def test_eval_does_not_advance_training_rng(self):
        a = engine(self.cfg, self.manifest)
        before = a.state()['rng']; a.evaluate(); after = a.state()['rng']
        equal(self, before, after)
        self.assertTrue(a.model.training)

    def test_restart_exact_dropout_in_process(self):
        a = engine(self.cfg, self.manifest)
        for _ in range(4):a.update()
        expected = a.state()
        b = engine(self.cfg, self.manifest)
        for _ in range(2):b.update()
        b.save(self.path / 'restart.pt')
        c = engine(self.cfg, self.manifest); c.resume(self.path / 'restart.pt'); c.evaluate()
        for _ in range(2):c.update()
        equal(self, expected, c.state())

    def test_restart_in_three_fresh_processes(self):
        script = '''import sys
from pathlib import Path
from stage_7_general_lm.common import configure,read_json
from stage_7_general_lm.data import TokenStream
from stage_7_general_lm.train import Engine
configure('cpu',1)
p=Path(sys.argv[1]);e=Engine(read_json(p/'config.json'),TokenStream(p/'manifest.json'),TokenStream(p/'manifest.json','dev'))
if sys.argv[2]=='resume':e.resume(p/'part.pt');e.evaluate()
for _ in range(int(sys.argv[3])):e.update()
e.save(p/sys.argv[4])
'''
        for mode, steps, filename in [('full', 4, 'full.pt'), ('part', 2, 'part.pt'), ('resume', 2, 'resumed.pt')]:
            result = subprocess.run([sys.executable, '-B', '-c', script, str(self.path), mode, str(steps), filename], cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        equal(self, load_checkpoint(self.path / 'full.pt'), load_checkpoint(self.path / 'resumed.pt'))

    def test_resume_rejects_changed_identity_and_bad_moments(self):
        a = engine(self.cfg, self.manifest);a.update();a.save(self.path / 'a.pt')
        changed = copy.deepcopy(self.cfg);changed['model']['dropout'] = .3
        with self.assertRaises(ValueError):engine(changed, self.manifest).resume(self.path / 'a.pt')
        checkpoint = load_checkpoint(self.path / 'a.pt');checkpoint['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
        torch.save(checkpoint, self.path / 'bad.pt')
        with self.assertRaises(ValueError):engine(self.cfg, self.manifest).resume(self.path / 'bad.pt')

    def test_cross_shard_window_and_corruption(self):
        values=(np.arange(800)%61+3).astype('<u2')
        entries=[]
        for i,part in enumerate((values[:300],values[300:])):
            path=self.path/('part'+str(i)+'.bin');path.write_bytes(part.tobytes())
            entries.append(dict(file=path.name,tokens=len(part),sha256=sha(path)))
        manifest=json.loads(self.manifest.read_text());manifest['splits']['train']['shards']=entries
        self.manifest.write_text(json.dumps(manifest))
        d = TokenStream(self.manifest)
        np.testing.assert_array_equal(d.read(298,6),values[298:304])
        self.assertEqual(d.read(d.size - 9, 9).shape, (9,))
        m = json.loads(self.manifest.read_text());m['splits']['train']['shards'][0]['sha256'] = '0' * 64
        self.manifest.write_text(json.dumps(m))
        with self.assertRaises(ValueError):TokenStream(self.manifest)

    def test_dtype_65536_survives_storage(self):
        value = np.asarray([65536], dtype=storage_dtype(65536))
        self.assertEqual(int(np.frombuffer(value.tobytes(), dtype='<u4')[0]), 65536)
        self.assertEqual(storage_dtype(65535), '<u2')

    def test_fp16_cpu_refused(self):
        cfg = copy.deepcopy(self.cfg);cfg['training']['precision'] = 'fp16'
        with self.assertRaises(ValueError):engine(cfg, self.manifest)

    def test_main_launch_is_blocked_without_resource_approval(self):
        target=self.path/'forbidden_run'
        result=subprocess.run([sys.executable,'-B','-m','stage_7_general_lm.train','--mode','train','--config',str(ROOT/'stage_7_general_lm/configs/candidate.json'),'--manifest',str(self.manifest),'--run-dir',str(target),'--device','cpu'],cwd=str(ROOT),capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('approved resource limits',result.stderr)
        self.assertFalse(target.exists())

    def test_sampler_read_limit_is_a_real_bound(self):
        raw=self.path/'huge.jsonl';raw.write_text(json.dumps(dict(id='a',source='s',revision='1',license='project',text='x'*4000))+'\n')
        report=sample(raw,self.path/'limited',max_documents=2,max_text_bytes=4096,max_read_bytes=128,max_seconds=1)
        self.assertEqual(report['input_bytes_consumed'],128)
        self.assertEqual(report['stop_reason'],'input_byte_limit')

    def test_boundary_checkpoint_rejects_masked_scratch_updates(self):
        e=engine(self.cfg,self.manifest)
        x=torch.ones(1,8,dtype=torch.long)*3;y=x.clone();y[:,4:]=-100
        e.update([(x,y),(x,y)])
        with self.assertRaises(ValueError):e.save(self.path/'invalid.pt')

    def test_source_group_and_content_leakage(self):
        with self.assertRaises(ValueError):assert_disjoint([dict(source_group='x')], [dict(source_group='x')])
        with self.assertRaises(ValueError):assert_disjoint([dict(text='A FACT')], [dict(passage=[dict(text='a fact')])])

    def test_domain_scoring_and_domain_stream(self):
        manifest=json.loads(self.manifest.read_text())
        for split in ('train','dev'):
            entry=manifest['splits'][split]['shards'][0]
            entry['source']='domain_a'
            second=dict(entry,source='domain_b')
            manifest['splits'][split]['shards'].append(second)
        self.manifest.write_text(json.dumps(manifest))
        d=TokenStream(self.manifest,'dev',domain='domain_b')
        self.assertEqual(d.size,800)
        scores=engine(self.cfg,self.manifest).evaluate()['scores']
        self.assertIn('dev_domain:domain_a',scores)
        self.assertIn('dev_domain:domain_b',scores)

    def test_bounded_sampler_and_dedup(self):
        raw = self.path / 'raw.jsonl'
        records = [dict(id=str(i), source='own', revision='1', license='project', text='Story '+str(i), source_group=str(i)) for i in range(10)]
        records.insert(1, dict(records[0]))
        raw.write_text(''.join(json.dumps(r)+'\n' for r in records))
        report = sample(raw, self.path / 'sample', max_documents=3, max_text_bytes=100, max_read_bytes=4000, max_seconds=1)
        self.assertEqual(report['documents'], 3)
        self.assertEqual(report['skipped']['duplicate'], 1)
        self.assertEqual(report['stop_reason'], 'document_limit')

    def test_development_contract_and_exact_scoring(self):
        cases = rows(ROOT / 'stage_7_general_lm/evaluation/dev_draft.jsonl')
        self.assertEqual(validate_cases(cases)['cases'], 20)
        predictions = [dict(case_id=r['case_id'],answer=r['reference']['answers'][0],citations=[s for f in r['reference']['required_facts'] for s in f['evidence']]) for r in cases]
        s = score(cases, predictions)
        self.assertEqual(s['exact_answer_match']['value'], 1.)
        self.assertIsNone(s['unsupported_claim_rate'])
        predictions[0]['citations'] = ['invented']
        self.assertFalse(score(cases,predictions)['per_case'][0]['valid_citation_ids'])

    def test_refusal_and_grouped_uncertainty(self):
        cases = rows(ROOT / 'stage_7_general_lm/evaluation/dev_draft.jsonl')
        predictions = [dict(case_id=r['case_id'],answer='The passage does not say.',citations=[]) for r in cases]
        s = score(cases, predictions)
        self.assertEqual(s['correct_uncertainty']['value'], 1.)
        self.assertEqual(s['over_refusal']['value'], 1.)
        result = grouped_bootstrap(cases, [1] * len(cases), repeats=50)
        self.assertEqual((result['low'], result['high'], result['groups']), (1.,1.,10))

    def test_padding_equals_eos_keeps_real_eos_label(self):
        b = collate([dict(input_ids=[1,5,2], labels=[-100,5,2]),dict(input_ids=[1,2], labels=[-100,2])], pad_id=2)
        self.assertEqual(b['labels'].tolist(),[[-100,5,2],[-100,2,-100]])
        self.assertEqual(b['attention_mask'].tolist(),[[1,1,1],[1,1,0]])

    def test_actual_native_chat_template_and_truncation(self):
        path = native_tokenizer_directory()
        from transformers import AutoTokenizer
        with patch('socket.create_connection', side_effect=AssertionError('Native tokenizer test must stay offline')), \
                patch('socket.socket.connect', side_effect=AssertionError('Native tokenizer test must stay offline')):
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.assertEqual((tokenizer.vocab_size, tokenizer.bos_token_id, tokenizer.eos_token_id,
                          tokenizer.pad_token_id), (49152, 1, 2, 2))
        development = rows(ROOT / 'stage_7_general_lm/evaluation/dev_draft.jsonl')
        training = rows(ROOT / 'stage_7_general_lm/evaluation/sft_train_draft.jsonl')
        self.assertEqual((len(development), len(training)), (20, 4))
        assert_disjoint(training, development)
        original_cases = copy.deepcopy(development + training)
        examples = []
        for case in development + training:
            with self.subTest(case_id=case['case_id'], split=case['split']):
                if case['split'] == 'train':
                    answer = case['response']
                else:
                    evidence = list(dict.fromkeys(s for fact in case['reference']['required_facts']
                                                  for s in fact['evidence']))
                    answer = json.dumps(dict(answer=case['reference']['answers'][0], citations=evidence))
                messages = [{'role': 'system', 'content': 'Answer using only the supplied passage.'},
                            {'role': 'user', 'content': prompt(case)}]
                prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                complete = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': answer}],
                                                        tokenize=True, add_generation_prompt=False)
                self.assertEqual(complete[:len(prefix)], prefix)
                example = response_example(tokenizer, case, answer)
                self.assertEqual(example['input_ids'], complete)
                self.assertEqual(example['labels'][:len(prefix)], [-100] * len(prefix))
                self.assertEqual(example['labels'][len(prefix):], complete[len(prefix):])
                self.assertEqual(example['labels'].count(tokenizer.eos_token_id), 1)
                # This pinned native template emits a newline AFTER real EOS.
                eos = example['labels'].index(tokenizer.eos_token_id)
                self.assertEqual(tokenizer.decode(complete[eos + 1:]), '\n')
                self.assertEqual(response_example(tokenizer, case, answer, max_length=len(complete)), example)
                for maximum in (8, len(complete) - 1):
                    with self.assertRaisesRegex(ValueError, 'truncation is refused'):
                        response_example(tokenizer, case, answer, max_length=maximum)
                examples.append(example)
        self.assertEqual(len(examples), 24)
        self.assertEqual(development + training, original_cases)
        batch = collate(examples, tokenizer.pad_token_id)
        width = batch['input_ids'].shape[1]
        self.assertTrue(any(len(example['input_ids']) < width for example in examples))
        for i, example in enumerate(examples):
            with self.subTest(padded_example=i):
                length = len(example['input_ids'])
                self.assertEqual(batch['labels'][i, :length].tolist(), example['labels'])
                self.assertEqual(batch['attention_mask'][i].tolist(), [1] * length + [0] * (width - length))
                self.assertEqual(batch['input_ids'][i, length:].tolist(), [tokenizer.pad_token_id] * (width - length))
                self.assertEqual(batch['labels'][i, length:].tolist(), [-100] * (width - length))
                self.assertEqual(batch['labels'][i].tolist().count(tokenizer.eos_token_id), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
