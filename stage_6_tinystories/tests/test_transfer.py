"""Transfer tests use temporary text files only, never substitute TinyStories data."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE/'tools'))
import verify_transfer as vt
import prepare_transfer as prep


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def ledger(self):
        (self.root/'alpha.txt').write_text('hello')
        (self.root/'nested').mkdir()
        (self.root/'nested/beta.txt').write_text('world')
        (self.root/vt.INVENTORY).parent.mkdir(parents=True, exist_ok=True)
        names = ['alpha.txt', 'nested/beta.txt']
        records = {n: {'bytes': (self.root/n).stat().st_size, 'sha256': vt.sha256(self.root/n)} for n in names}
        (self.root/vt.INVENTORY).write_text(json.dumps({'files': records}))
        names = sorted(names+[vt.FILES, vt.SUMS, vt.INVENTORY])
        (self.root/vt.FILES).write_text('\n'.join(names)+'\n')
        (self.root/vt.SUMS).write_text('\n'.join(vt.sha256(self.root/n)+'  '+n for n in names if n != vt.SUMS)+'\n')

    def run_verify(self, **kwargs):
        # Model/data validation is tested separately on the genuine assets.
        with mock.patch.object(vt, 'validate_dependencies', return_value={'tokenizer_sha256': 'unit-test'}):
            return vt.verify(self.root, **kwargs)

    def test_relative_paths_only(self):
        for name in ('/outside/file', '../outside', 'a/../b', 'C:\\temp', '~/.file', './file', 'a\nfile'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                vt.inside(self.root, name)
        self.assertEqual(vt.inside(self.root, 'a/b'), self.root/'a/b')

    def test_symlinked_file_or_parent_rejected(self):
        (self.root/'physical').mkdir()
        (self.root/'physical/source').write_text('source')
        (self.root/'linked').symlink_to(self.root/'physical', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            vt.inside(self.root, 'linked/source')

    def test_atomic_copy_verifies_and_preserves_source(self):
        src, dest = self.root/'source.txt', self.root/'copy.txt'
        src.write_text('verified source')
        digest = vt.sha256(src)
        prep.atomic_copy(src, dest, digest)
        self.assertEqual(vt.sha256(src), digest)
        self.assertEqual(vt.sha256(dest), digest)
        self.assertFalse(list(self.root.glob('.*')))

    def test_failed_copy_does_not_replace_destination(self):
        src, dest = self.root/'source.txt', self.root/'copy.txt'
        src.write_text('source')
        dest.write_text('preserve')
        with self.assertRaises(ValueError):
            prep.atomic_copy(src, dest, '0'*64)
        self.assertEqual(dest.read_text(), 'preserve')

    def test_baseline_copy_refuses_wrong_existing_file(self):
        (self.root/'source.txt').write_text('verified')
        (self.root/'destination.txt').write_text('other artifact')
        with mock.patch.object(vt, 'BASELINES', ('source.txt', 'destination.txt')), \
                mock.patch.object(vt, 'BASELINE_SHA', vt.sha256(self.root/'source.txt')):
            with self.assertRaisesRegex(ValueError, 'refusing'):
                prep.ensure_baseline(self.root)
        self.assertEqual((self.root/'destination.txt').read_text(), 'other artifact')

    def test_complete_ledger(self):
        self.ledger()
        report = self.run_verify()
        self.assertTrue(report['passed'])
        self.assertEqual(report['files_present'], 5)
        self.assertEqual(report['files_hashed'], 4)

    def test_same_size_corruption_rejected(self):
        self.ledger()
        (self.root/'alpha.txt').write_text('xxxxx')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.run_verify()

    def test_missing_file_rejected(self):
        self.ledger()
        (self.root/'nested/beta.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_verify()

    def test_unlisted_inventory_file_rejected(self):
        self.ledger()
        p = self.root/vt.FILES
        p.write_text(p.read_text()+'unlisted.txt\n')
        with self.assertRaises(ValueError):
            self.run_verify()

    def test_relocated_bundle_without_original_paths(self):
        self.ledger()
        with tempfile.TemporaryDirectory() as other:
            moved = Path(other)/'relocated'
            shutil.copytree(str(self.root), str(moved))
            with mock.patch.object(vt, 'validate_dependencies', return_value={'tokenizer_sha256': 'unit-test'}):
                self.assertTrue(vt.verify(moved)['passed'])

    def test_pbs_resource_and_default_training_contract(self):
        script = (STAGE/'hpc/train.pbs').read_text()
        self.assertIn('select=1:ncpus=4:ngpus=1:mem=20gb:centos=skylake', script)
        self.assertIn('walltime=04:00:00', script)
        self.assertIn('STAGE6_MODE=${STAGE6_MODE:-train}', script)
        self.assertIn('STAGE6_RESUME=${STAGE6_RESUME:-${STAGE6_RUN_DIR}/latest.pt}', script)
        self.assertIn('STAGE6_STOP_AFTER=${STAGE6_STOP_AFTER:-3052}', script)
        self.assertIn('Training will not fall back to the Stage 5 baseline.', script)
        self.assertIn('--walltime-seconds 14400', script)
        self.assertIn('stage_6_tinystories/evaluate.py', script)
        self.assertIn("s.get('reason') == 'review_gate'", script)
        self.assertIn('tools/verify_transfer.py --quick', script)
        self.assertNotIn('NanoGPT_stage6', script)
        subprocess.run(['bash', '-n', str(STAGE/'hpc/train.pbs')], check=True)

    def test_pbs_rejects_wrong_working_directory_without_creating_folders(self):
        import os
        env = dict(os.environ, PBS_O_WORKDIR=str(self.root))
        env.pop('STAGE6_PROJECT_DIR', None)
        result = subprocess.run(['bash', str(STAGE/'hpc/train.pbs')], cwd=str(self.root),
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('repository root', result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
