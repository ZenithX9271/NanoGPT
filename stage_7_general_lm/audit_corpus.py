"""Independent integrity audit for a packed Stage 7 token-stream manifest."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def child(directory, name):
    path = (directory / name).resolve()
    try:
        path.relative_to(directory)
    except ValueError:
        raise ValueError('Manifest path escapes packed directory: ' + name)
    return path


def audit(manifest_path, complete_path, config_path, tokenizer_path,
          minimum_train_tokens=100000000, minimum_domain_eval_tokens=1025):
    started = time.monotonic()
    manifest_path = Path(manifest_path).resolve()
    directory = manifest_path.parent
    complete_path = Path(complete_path).resolve()
    config_path = Path(config_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    manifest, complete, config = (load_json(manifest_path),
                                  load_json(complete_path),
                                  load_json(config_path))
    if manifest.get('format') != 'stage7_token_stream_v2':
        raise ValueError('Unknown packed-corpus format')
    if complete.get('format') != 'stage7_corpus_complete_v1':
        raise ValueError('Unknown completion-marker format')
    manifest_sha = file_sha256(manifest_path)
    if complete.get('manifest_sha256') != manifest_sha:
        raise ValueError('Completion marker does not match manifest')
    if manifest.get('config_sha256') != file_sha256(config_path):
        raise ValueError('Manifest configuration identity mismatch')
    if manifest.get('tokenizer_sha256') != file_sha256(tokenizer_path):
        raise ValueError('Manifest tokenizer identity mismatch')

    dtype = np.dtype(manifest['dtype'])
    if dtype not in (np.dtype('<u2'), np.dtype('<u4')):
        raise ValueError('Unsupported packed token dtype')
    vocab_size = manifest['vocab_size']
    split_source_tokens = {}
    split_tokens = Counter()
    source_tokens = Counter()
    shard_reports = []
    group_sets = {}

    for split in ('train', 'dev', 'heldout'):
        info = manifest['splits'][split]
        for shard in info['shards']:
            path = child(directory, shard['file'])
            expected_bytes = shard['tokens'] * dtype.itemsize
            if path.stat().st_size != expected_bytes:
                raise ValueError('Token-shard byte size mismatch: ' + path.name)
            digest = file_sha256(path)
            if digest != shard['sha256']:
                raise ValueError('Token-shard SHA256 mismatch: ' + path.name)
            values = np.memmap(str(path), dtype=dtype, mode='r')
            maximum = int(values.max()) if len(values) else None
            if maximum is not None and maximum >= vocab_size:
                raise ValueError('Token ID outside vocabulary: ' + path.name)
            key = split + ':' + shard['source']
            split_source_tokens[key] = split_source_tokens.get(key, 0) + shard['tokens']
            split_tokens[split] += shard['tokens']
            source_tokens[shard['source']] += shard['tokens']
            shard_reports.append({'file': path.name, 'tokens': shard['tokens'],
                                  'bytes': expected_bytes, 'sha256': digest,
                                  'maximum_token_id': maximum})

        group_info = info['source_group_index']
        group_path = child(directory, group_info['file'])
        if file_sha256(group_path) != group_info['sha256']:
            raise ValueError('Group-index SHA256 mismatch: ' + split)
        lines = group_path.read_text(encoding='ascii').splitlines()
        if len(lines) != group_info['groups']:
            raise ValueError('Group index count mismatch: ' + split)
        if len(lines) != len(set(lines)):
            raise ValueError('Group index uniqueness mismatch: ' + split)
        if any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
               for value in lines):
            raise ValueError('Invalid group digest: ' + split)
        group_sets[split] = set(lines)

    for left in group_sets:
        for right in group_sets:
            if left < right and group_sets[left] & group_sets[right]:
                raise ValueError('Source-group digest crossed splits')
    if split_tokens['train'] < minimum_train_tokens:
        raise ValueError('Training tokens below review-gate minimum')
    for split in ('dev', 'heldout'):
        for source in manifest['source_token_counts']:
            if split_source_tokens.get(split + ':' + source, 0) < minimum_domain_eval_tokens:
                raise ValueError('Source/domain evaluation split is too short')

    recorded_by_split = manifest['source_token_counts_by_split']
    for split, values in recorded_by_split.items():
        for source, tokens in values.items():
            if split_source_tokens.get(split + ':' + source, 0) != tokens:
                raise ValueError('Split/source token total mismatch')
    if dict(source_tokens) != manifest['source_token_counts']:
        raise ValueError('Source token totals do not match shards')
    if any(tokens > manifest['source_token_caps'][source]
           for source, tokens in source_tokens.items()):
        raise ValueError('Source token cap exceeded')

    ledger_info = manifest['document_ledger']
    ledger_path = child(directory, ledger_info['file'])
    if file_sha256(ledger_path) != ledger_info['sha256']:
        raise ValueError('Document-ledger SHA256 mismatch')
    ledger_documents = Counter()
    ledger_tokens = Counter()
    ledger_rows = 0
    with gzip.open(str(ledger_path), 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            ledger_rows += 1
            split, source, tokens = record['split'], record['source'], record['tokens']
            if split not in ('train', 'dev', 'heldout') or source not in source_tokens or tokens <= 0:
                raise ValueError('Invalid document-ledger record')
            for field in ('text_sha256', 'normalized_sha256'):
                value = record.get(field, '')
                if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Invalid document-ledger digest')
            ledger_documents[split] += 1
            ledger_documents[source] += 1
            ledger_documents[split + ':' + source] += 1
            ledger_tokens[split + ':' + source] += tokens
    if ledger_rows != sum(manifest['document_counts'][source] for source in source_tokens):
        raise ValueError('Document-ledger row count mismatch')
    for key, value in manifest['document_counts'].items():
        if ledger_documents[key] != value:
            raise ValueError('Document count mismatch: ' + key)
    for key, value in split_source_tokens.items():
        if ledger_tokens[key] != value:
            raise ValueError('Document-ledger token total mismatch: ' + key)

    input_reports = []
    for entry in manifest['input_documents']:
        path = Path(entry['file']).resolve()
        digest = file_sha256(path)
        if digest != entry['sha256']:
            raise ValueError('Extracted input SHA256 mismatch')
        input_reports.append({'file': str(path), 'bytes': path.stat().st_size,
                              'sha256': digest})

    return {
        'format': 'stage7_corpus_audit_v1',
        'status': 'PASS',
        'manifest_sha256': manifest_sha,
        'vocab_size': vocab_size,
        'dtype': dtype.str,
        'split_tokens': dict(split_tokens),
        'source_tokens': dict(source_tokens),
        'ledger_documents': ledger_rows,
        'group_counts': {split: len(values) for split, values in group_sets.items()},
        'shards': shard_reports,
        'inputs': input_reports,
        'minimum_train_tokens': minimum_train_tokens,
        'minimum_domain_eval_tokens': minimum_domain_eval_tokens,
        'seconds': time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--complete', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--minimum-train-tokens', type=int, default=100000000)
    parser.add_argument('--minimum-domain-eval-tokens', type=int, default=1025)
    args = parser.parse_args()
    result = audit(args.manifest, args.complete, args.config, args.tokenizer,
                   args.minimum_train_tokens, args.minimum_domain_eval_tokens)
    write_json(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
