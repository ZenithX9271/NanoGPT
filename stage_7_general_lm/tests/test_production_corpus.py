"""Bounded production-corpus utility tests; optional dependencies skip locally."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HAS_PYARROW = importlib.util.find_spec('pyarrow') is not None
HAS_TOKENIZERS = importlib.util.find_spec('tokenizers') is not None


@unittest.skipUnless(HAS_PYARROW and HAS_TOKENIZERS,
                     'production preprocessing dependencies unavailable')
class ProductionCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def make_config(self, parquet_sha, tokenizer_sha):
        config = {
            'format': 'stage7_production_sources_v1',
            'network_cap_bytes': 1000000,
            'extraction_limits': {'max_documents': 20,
                                  'max_output_bytes': 1000000,
                                  'max_seconds': 30},
            'packing': {'batch_documents': 2, 'max_seconds': 30,
                        'split_seed': 1337},
            'known_limitations': ['test fixture'],
            'tokenizer': {'vocab_size': 6, 'bos_id': 1, 'eos_id': 2,
                          'sha256': tokenizer_sha, 'revision': 'test'},
            'sources': [{
                'name': 'fixture', 'dataset_id': 'fixture/data',
                'revision': 'abc', 'relative_path': 'fixture.parquet',
                'sha256': parquet_sha,
                'expected_bytes': (self.path / 'fixture.parquet').stat().st_size,
                'license': 'test-only',
                'text_column': 'text', 'id_columns': ['id'],
                'group_columns': ['group'], 'max_text_bytes': 10000,
                'max_tokens': 100,
            }],
        }
        path = self.path / 'config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        return path

    def test_extract_and_pack_with_provenance_and_dedup(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        from tokenizers import Tokenizer, models, pre_tokenizers
        from stage_7_general_lm.extract_parquet import extract, file_sha256
        from stage_7_general_lm.audit_corpus import audit
        from stage_7_general_lm.pack_production import pack

        parquet_path = self.path / 'fixture.parquet'
        parquet.write_table(arrow.table({
            'id': ['1', '2', '3'], 'group': ['a', 'b', 'c'],
            'text': ['stage seven', 'general model', '  STAGE   SEVEN  '],
        }), parquet_path)
        tokenizer = Tokenizer(models.WordLevel(
            vocab={'[UNK]': 0, '<bos>': 1, '<eos>': 2,
                   'stage': 3, 'seven': 4, 'general': 5},
            unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer_path = self.path / 'tokenizer.json'
        tokenizer.save(str(tokenizer_path))
        config = self.make_config(file_sha256(parquet_path),
                                  file_sha256(tokenizer_path))

        extraction = extract(config, self.path, self.path / 'extracted')
        self.assertEqual(extraction['documents'], 3)
        manifest = pack(config, [self.path / 'extracted/documents.jsonl'],
                        tokenizer_path, self.path / 'packed')
        self.assertEqual(manifest['unique_normalized_documents'], 2)
        self.assertEqual(sum(manifest['source_token_counts'].values()), 8)
        self.assertTrue((self.path / 'packed/COMPLETE.json').is_file())
        with gzip.open(str(self.path / 'packed/documents.jsonl.gz'), 'rt') as stream:
            ledger = [json.loads(line) for line in stream]
        self.assertEqual(len(ledger), 2)
        self.assertTrue(all('text_sha256' in record for record in ledger))
        report = audit(self.path / 'packed/manifest.json',
                       self.path / 'packed/COMPLETE.json', config,
                       tokenizer_path, minimum_train_tokens=1,
                       minimum_domain_eval_tokens=0)
        self.assertEqual(report['status'], 'PASS')

    def test_extraction_rejects_wrong_source_hash(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        from tokenizers import Tokenizer, models
        from stage_7_general_lm.extract_parquet import extract, file_sha256

        parquet_path = self.path / 'fixture.parquet'
        parquet.write_table(arrow.table({'text': ['x']}), parquet_path)
        tokenizer = Tokenizer(models.WordLevel(vocab={'[UNK]': 0},
                                               unk_token='[UNK]'))
        tokenizer_path = self.path / 'tokenizer.json'
        tokenizer.save(str(tokenizer_path))
        config = self.make_config('0' * 64, file_sha256(tokenizer_path))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            extract(config, self.path, self.path / 'rejected')


if __name__ == '__main__':
    unittest.main()
