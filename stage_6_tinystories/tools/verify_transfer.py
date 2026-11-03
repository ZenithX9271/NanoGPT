#!/usr/bin/env python
"""Verify the complete transferred payload without CUDA or third-party packages.

Run from any directory. --quick checks all sizes and hashes critical small files
and both baselines; default checks every file's SHA256. --deep-data also scans
real shards/JSONL with NumPy. No data/checkpoint is written or trained.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

FILES = 'stage_6_tinystories/deployment/files.txt'
SUMS = 'stage_6_tinystories/deployment/sha256sums.txt'
INVENTORY = 'stage_6_tinystories/deployment/inventory.json'
BASELINE_SHA = '2c6d16739cdf7128921a9cc0876793067651debc29484602ac2ce51cf86211ce'
TOKENIZER_SHA = '99055f35c77392e4df4486685b60fee1de1d95e2c373c26f487cf9944746d038'
MANIFEST_SHA = '89c9df839572ef6a1890caf4b1144229090b8ced8b4ea2c66ed5dd090e17c9a7'
BASELINES = ('stage_5_tinystories/checkpoints/best.pt',
             'stage_6_tinystories/baseline/stage5_best.pt')
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(str(path), encoding='utf-8') as stream:
        return json.load(stream)


def inside(root, name):
    """A transfer asset must be a physical repository-relative path, not a link."""
    if not isinstance(name, str) or not name or any(c in name for c in ('\\', '\n', '\r', ':')):
        raise ValueError('Invalid portable path: ' + repr(name))
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or rel.as_posix() != name or name.startswith('~'):
        raise ValueError('Path must be repository-relative: ' + name)
    path = Path(root)
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Required transfer path is a symlink: ' + name)
    return path


def payload_files(root):
    """Explicit transfer scope: code/history sources, prepared data, two baselines."""
    names = {'.gitignore', 'LICENSE', 'input.txt', 'requirements.txt'}
    names.update(p.name for p in root.glob('*.md'))
    required_docs = {
        'stage_6_tinystories/docs/RESULTS.md',
        'stage_6_tinystories/docs/TRANSFER.md',
        'stage_6_tinystories/docs/HPC_VALIDATION.md',
        'stage_6_tinystories/docs/DATA_RECOVERY.md',
    }
    names.update(required_docs)
    excluded = {'runs', 'logs', 'evaluations', '__pycache__', 'output',
                'checkpoints', 'checkpoints_pilot', 'transfer_backups'}
    for folder in sorted(root.glob('stage_[1-6]_*')):
        for p in folder.rglob('*'):
            rel = p.relative_to(root)
            if p.is_file() and rel.as_posix() not in {FILES, SUMS, INVENTORY} and not (excluded & set(rel.parts)) and (
                    p.suffix in ('.py', '.json', '.md', '.pbs', '.txt') or p.name == '.gitignore'):
                names.add(rel.as_posix())
    directory = root/'data/shards/v2'
    manifest = read_json(directory/'manifest.json')
    names.add('data/shards/v2/manifest.json')
    for split, info in manifest['splits'].items():
        names.add('data/tinystories/v2/'+split+'.jsonl')
        names.add('data/shards/v2/'+split+'_offsets.json')
        for entry in info['shards']:
            names.add('data/shards/v2/'+entry['file'])
    names.update(['data/tinystories/v2/provenance.json', 'data/tokenizer/v2/tokenizer.json',
                  'data/tokenizer/v2/vocab_info.json']+list(BASELINES))
    for name in names:
        if not inside(root, name).is_file():
            raise FileNotFoundError('Required transfer file missing: '+name)
    return sorted(names)


def validate_dependencies(root):
    if sha256(inside(root, 'data/shards/v2/manifest.json')) != MANIFEST_SHA:
        raise ValueError('Dataset manifest differs from the verified Stage 5 v2 identity')
    manifest = read_json(inside(root, 'data/shards/v2/manifest.json'))
    tokenizer_path = inside(root, 'data/tokenizer/v2/tokenizer.json')
    tokenizer = read_json(tokenizer_path)
    tokenizer_sha = sha256(tokenizer_path)
    info = read_json(inside(root, 'data/tokenizer/v2/vocab_info.json'))
    if tokenizer_sha != TOKENIZER_SHA or tokenizer_sha != manifest['tokenizer_sha256'] or tokenizer_sha != info['tokenizer_sha256']:
        raise ValueError('Tokenizer/manifest/vocab metadata identity mismatch')
    for name in BASELINES:
        if sha256(inside(root, name)) != BASELINE_SHA:
            raise ValueError('Wrong baseline checkpoint: '+name)
    for filename in ('continue.json', 'control_lr.json'):
        cfg = read_json(inside(root, 'stage_6_tinystories/configs/'+filename))
        if inside(root, cfg['data']['shard_dir']) != root/'data/shards/v2':
            raise ValueError('Unexpected shard directory in '+filename)
        for value in (cfg['data']['tokenizer'], cfg['initial_checkpoint'], cfg['evaluation']['prompts']):
            if not inside(root, value).is_file():
                raise FileNotFoundError('Configured asset missing: '+value)
        if sha256(inside(root, cfg['initial_checkpoint'])) != cfg['initial_checkpoint_sha256']:
            raise ValueError('Configured baseline SHA256 mismatch')
        if cfg['model']['vocab_size'] != manifest['vocab_size'] or len(tokenizer['model']['vocab']) != manifest['vocab_size']:
            raise ValueError('Vocabulary size mismatch')
        for token in ('pad', 'bos', 'eos'):
            expected = cfg['data'][token+'_id']
            if manifest['special_ids'][token] != expected or tokenizer['model']['vocab']['<'+token+'>'] != expected:
                raise ValueError('Special-token identity mismatch')
        prompts = read_json(inside(root, cfg['evaluation']['prompts']))
        if prompts['tokenizer_sha256'] != tokenizer_sha:
            raise ValueError('Prompt tokenizer identity mismatch')
        if sha256(inside(root, 'data/shards/v2/dev_val_offsets.json')) != prompts['dev_offsets_sha256']:
            raise ValueError('Development offset identity mismatch')
    for split, entry in manifest['splits'].items():
        total = 0
        for shard in entry['shards']:
            if inside(root, 'data/shards/v2/'+shard['file']).stat().st_size != shard['tokens']*2:
                raise ValueError('Shard size mismatch: '+shard['file'])
            total += shard['tokens']
        if total != entry['token_count']:
            raise ValueError('Manifest total mismatch: '+split)
    return manifest


def deep_data(root, manifest):
    import numpy as np  # Optional here; available in the tested IITD module.
    import re
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(inside(root, 'data/tokenizer/v2/tokenizer.json')))
    except ImportError:
        tokenizer = None
    result = {}
    heldout = set()
    for split in ('dev_val', 'test', 'train'):
        info = manifest['splits'][split]
        bos = eos = 0
        for entry in info['shards']:
            data = np.memmap(str(inside(root, 'data/shards/v2/'+entry['file'])), dtype='<u2', mode='r')
            for begin in range(0, len(data), 1000000):
                chunk = data[begin:begin+1000000]
                if int(chunk.max()) >= manifest['vocab_size']:
                    raise ValueError('Out-of-range IDs in '+entry['file'])
                bos += int(np.count_nonzero(chunk == manifest['special_ids']['bos']))
                eos += int(np.count_nonzero(chunk == manifest['special_ids']['eos']))
        offsets = np.asarray(read_json(inside(root, 'data/shards/v2/'+split+'_offsets.json')), dtype=np.int64)
        if offsets.shape != (info['story_count'], 2) or offsets[0, 0] != 0 or offsets[-1, 1] != info['token_count']:
            raise ValueError('Document offset endpoints/count mismatch: '+split)
        if not np.all(offsets[:, 1] > offsets[:, 0]) or not np.all(offsets[1:, 0] == offsets[:-1, 1]):
            raise ValueError('Noncontiguous document offsets: '+split)
        with open(str(inside(root, 'data/tinystories/v2/'+split+'.jsonl')), encoding='utf-8') as stream:
            count = 0
            own_heldout = set()
            samples = []
            for line in stream:
                record = json.loads(line)
                if not isinstance(record.get('text'), str) or not isinstance(record.get('id'), int):
                    raise ValueError('Invalid real JSONL record: '+split)
                fingerprint = hashlib.md5(re.sub(r'\s+', ' ', record['text'].lower()).strip().encode('utf-8')).digest()
                if fingerprint in heldout:
                    raise ValueError('Normalized overlap across dataset splits: '+split)
                if split != 'train':
                    if fingerprint in own_heldout:
                        raise ValueError('Duplicate held-out story: '+split)
                    own_heldout.add(fingerprint)
                if count < 8:
                    samples.append(record['text'])
                count += 1
        if split != 'train':
            heldout.update(own_heldout)
        if bos != info['story_count'] or eos != count or count != info['story_count']:
            raise ValueError('BOS/EOS/JSONL/document-count mismatch: '+split)
        reencoded = 0
        if tokenizer is not None:
            first = np.memmap(str(inside(root, 'data/shards/v2/'+info['shards'][0]['file'])), dtype='<u2', mode='r')
            for index, text in enumerate(samples):
                begin, end = offsets[index]
                if end > len(first):
                    raise ValueError('Reencoding sample exceeds first shard')
                expected = [manifest['special_ids']['bos']]+tokenizer.encode(text).ids+[manifest['special_ids']['eos']]
                if not np.array_equal(first[begin:end], np.asarray(expected)):
                    raise ValueError('JSONL/tokenizer/shard sample mismatch: '+split)
                reencoded += 1
        result[split] = {'stories': count, 'tokens': info['token_count'], 'shards': len(info['shards']),
                         'bos': bos, 'eos': eos, 'offsets_valid': True, 'normalized_cross_split_overlap': 0,
                         'sampled_stories_reencoded': reencoded,
                         'optional_tokenizers_available': tokenizer is not None}
    return result


def verify(root, quick=False, deep=False):
    root = Path(root).resolve()
    inventory = read_json(inside(root, INVENTORY))
    paths = inside(root, FILES).read_text(encoding='utf-8').splitlines()
    if paths != sorted(set(paths)):
        raise ValueError('Transfer file list must be sorted and unique')
    records = inventory['files']
    expected = set(records) | {FILES, SUMS, INVENTORY}
    if set(paths) != expected:
        raise ValueError('Transfer list and inventory disagree')
    sums = {}
    for line in inside(root, SUMS).read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if len(digest) != 64 or name in sums:
            raise ValueError('Invalid checksum ledger')
        sums[name] = digest
    if set(sums) != expected-{SUMS}:
        raise ValueError('Checksum ledger is incomplete')
    hashed = checked = 0
    for name in paths:
        path = inside(root, name)
        if not path.is_file():
            raise FileNotFoundError('Transfer incomplete: '+name)
        if name in records and path.stat().st_size != records[name]['bytes']:
            raise ValueError('File size changed: '+name)
        checked += 1
        do_hash = name != SUMS and (not quick or path.suffix not in ('.bin', '.jsonl'))
        if do_hash:
            digest = sha256(path)
            if digest != sums[name] or (name in records and digest != records[name]['sha256']):
                raise ValueError('File SHA256 changed: '+name)
            hashed += 1
    manifest = validate_dependencies(root)
    report = {'passed': True, 'files_present': checked, 'files_hashed': hashed,
              'mode': 'quick_sizes_and_critical_hashes' if quick else 'all_payload_sha256',
              'payload_bytes': sum(r['bytes'] for r in records.values()),
              'baseline_sha256': BASELINE_SHA,
              'tokenizer_sha256': manifest['tokenizer_sha256'], 'missing_files': []}
    if deep:
        report['deep_data'] = deep_data(root, manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=str(ROOT))
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--deep-data', action='store_true')
    args = parser.parse_args()
    try:
        report = verify(args.project_root, args.quick, args.deep_data)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, 'TRANSFER VERIFICATION FAILED: '+str(error)+'\n')
    print(json.dumps(report, indent=2))
    print('TRANSFER VERIFICATION PASS')


if __name__ == '__main__':
    main()
