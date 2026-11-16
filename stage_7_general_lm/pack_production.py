"""Deduplicate, split, tokenize, and pack a bounded Stage 7 production corpus.

This module is intentionally torch-free and is run in the isolated Python 3.9
preprocessing environment after Parquet extraction.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import unicodedata

import numpy as np
from tokenizers import Tokenizer


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def storage_dtype(maximum_id):
    if maximum_id < 0 or maximum_id > 4294967295:
        raise ValueError('Token IDs cannot be represented')
    return '<u2' if maximum_id <= 65535 else '<u4'


def split_for(group, seed):
    value = int(hashlib.sha256((str(seed) + ':' + group).encode('utf-8')).hexdigest()[:12], 16) % 10
    return 'dev' if value == 8 else 'heldout' if value == 9 else 'train'


def normalized_text(text):
    text = unicodedata.normalize('NFC', text).strip()
    return text, ' '.join(text.casefold().split())


def pack(config_path, documents, tokenizer_path, output):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('format') != 'stage7_production_sources_v1':
        raise ValueError('Unknown production source configuration')
    packing = config['packing']
    if min(packing['batch_documents'], packing['max_seconds']) <= 0:
        raise ValueError('Packing limits must be positive')
    source_config = {item['name']: item for item in config['sources']}
    if len(source_config) != len(config['sources']):
        raise ValueError('Source names must be unique')
    if any(item.get('max_tokens', 0) <= 0 for item in source_config.values()):
        raise ValueError('Every source needs a positive max_tokens')

    tokenizer_path = Path(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    maximum_id = max(vocab.values())
    vocab_size = maximum_id + 1
    if tokenizer.get_vocab_size() != config['tokenizer']['vocab_size'] or vocab_size != config['tokenizer']['vocab_size']:
        raise ValueError('Tokenizer vocabulary identity differs from configuration')
    if file_sha256(tokenizer_path) != config['tokenizer']['sha256']:
        raise ValueError('Tokenizer SHA256 differs from configuration')
    bos_id, eos_id = config['tokenizer']['bos_id'], config['tokenizer']['eos_id']
    if not 0 <= bos_id <= maximum_id or not 0 <= eos_id <= maximum_id:
        raise ValueError('BOS/EOS IDs outside tokenizer vocabulary')
    dtype = np.dtype(storage_dtype(maximum_id))

    document_paths = [Path(path).resolve() for path in documents]
    if not document_paths or any(not path.is_file() for path in document_paths):
        raise ValueError('All extracted document files must exist')
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seen_text = set()
    group_splits = {}
    groups = {name: set() for name in ('train', 'dev', 'heldout')}
    token_counts = Counter()
    document_counts = Counter()
    skipped = Counter()
    scanned = 0
    stop_reason = 'input_exhausted'
    source_order = sorted(source_config)
    paths = {}

    with ExitStack() as stack:
        handles = {}
        for split in ('train', 'dev', 'heldout'):
            for index, source in enumerate(source_order):
                path = directory / ('%s_%02d.bin' % (split, index))
                paths[split, source] = path
                handles[split, source] = stack.enter_context(path.open('xb'))
        ledger = stack.enter_context(gzip.open(str(directory / 'documents.jsonl.gz'),
                                               'xt', encoding='utf-8'))

        batch = []

        def consume(records):
            if not records:
                return
            encodings = tokenizer.encode_batch([record['_text'] for record in records],
                                               add_special_tokens=False)
            for record, encoding in zip(records, encodings):
                source = record['source']
                ids = [bos_id] + encoding.ids + [eos_id]
                remaining = source_config[source]['max_tokens'] - token_counts[source]
                if len(ids) > remaining:
                    skipped[source + ':token_cap'] += 1
                    continue
                if ids and (min(ids) < 0 or max(ids) >= vocab_size):
                    raise ValueError('Tokenizer emitted ID outside vocabulary')
                split = record['_split']
                handles[split, source].write(np.asarray(ids, dtype=dtype).tobytes())
                token_counts[source] += len(ids)
                token_counts[split] += len(ids)
                token_counts[split + ':' + source] += len(ids)
                document_counts[source] += 1
                document_counts[split] += 1
                document_counts[split + ':' + source] += 1
                groups[split].add(record['source_group'])
                ledger_record = {key: record.get(key) for key in
                                 ('id', 'source', 'dataset_id', 'revision', 'license',
                                  'source_file', 'source_file_sha256', 'source_row',
                                  'source_group')}
                ledger_record.update(split=split, tokens=len(ids),
                                     text_sha256=record['_text_sha256'],
                                     normalized_sha256=record['_normalized_sha256'])
                ledger.write(json.dumps(ledger_record, sort_keys=True,
                                        ensure_ascii=False) + '\n')

        for document_path in document_paths:
            with document_path.open(encoding='utf-8') as stream:
                for line in stream:
                    scanned += 1
                    if time.monotonic() - started >= packing['max_seconds']:
                        consume(batch)
                        batch = []
                        stop_reason = 'time_limit'
                        break
                    record = json.loads(line)
                    source = record.get('source')
                    if source not in source_config:
                        raise ValueError('Unknown source in extracted documents')
                    if token_counts[source] >= source_config[source]['max_tokens']:
                        skipped[source + ':already_capped'] += 1
                        if all(token_counts[name] >= source_config[name]['max_tokens']
                               for name in source_order):
                            stop_reason = 'all_source_token_caps'
                            break
                        continue
                    text, normalized = normalized_text(record.get('text', ''))
                    if not text:
                        skipped[source + ':empty'] += 1
                        continue
                    normalized_digest = hashlib.sha256(normalized.encode('utf-8')).digest()
                    if normalized_digest in seen_text:
                        skipped[source + ':duplicate'] += 1
                        continue
                    seen_text.add(normalized_digest)
                    group = record.get('source_group')
                    if not group:
                        raise ValueError('Every extracted document requires source_group')
                    split = split_for(group, packing['split_seed'])
                    previous = group_splits.setdefault(group, split)
                    if previous != split:
                        raise ValueError('Source group crossed corpus splits')
                    record['_text'] = text
                    record['_split'] = split
                    record['_text_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    record['_normalized_sha256'] = normalized_digest.hex()
                    batch.append(record)
                    if len(batch) >= packing['batch_documents']:
                        consume(batch)
                        batch = []
                if stop_reason != 'input_exhausted':
                    break
        consume(batch)

    group_entries = {}
    for split in ('train', 'dev', 'heldout'):
        path = directory / ('group_digests_%s.txt' % split)
        with path.open('x', encoding='ascii') as stream:
            for group in sorted(groups[split]):
                stream.write(hashlib.sha256(group.encode('utf-8')).hexdigest() + '\n')
        group_entries[split] = {'file': path.name, 'groups': len(groups[split]),
                                'sha256': file_sha256(path)}

    splits = {}
    for split in ('train', 'dev', 'heldout'):
        shards = []
        for source in source_order:
            path = paths[split, source]
            tokens = token_counts[split + ':' + source]
            if path.stat().st_size != tokens * dtype.itemsize:
                raise ValueError('Packed shard size mismatch')
            if tokens:
                shards.append({'file': path.name, 'tokens': tokens,
                               'sha256': file_sha256(path), 'source': source})
        splits[split] = {'shards': shards,
                         'source_group_index': group_entries[split]}

    ledger_path = directory / 'documents.jsonl.gz'
    manifest = {
        'format': 'stage7_token_stream_v2',
        'vocab_size': vocab_size,
        'dtype': dtype.str,
        'tokenizer_sha256': file_sha256(tokenizer_path),
        'config_sha256': file_sha256(config_path),
        'input_documents': [{'file': str(path), 'sha256': file_sha256(path)}
                            for path in document_paths],
        'document_ledger': {'file': ledger_path.name,
                            'sha256': file_sha256(ledger_path)},
        'special_ids': {'bos': bos_id, 'eos': eos_id},
        'splits': splits,
        'source_token_counts': {source: token_counts[source]
                                for source in source_order},
        'source_token_caps': {source: source_config[source]['max_tokens']
                              for source in source_order},
        'source_token_counts_by_split': {
            split: {source: token_counts[split + ':' + source]
                    for source in source_order}
            for split in ('train', 'dev', 'heldout')},
        'document_counts': dict(document_counts),
        'documents_scanned': scanned,
        'unique_normalized_documents': len(seen_text),
        'skipped': dict(skipped),
        'split_seed': packing['split_seed'],
        'seconds': time.monotonic() - started,
        'stop_reason': stop_reason,
        'mixture_policy': ('Fixed per-source selected-token caps before random-window '
                           'sampling; actual split/source counts are recorded.'),
        'cross_document_attention': True,
        'known_limitations': config['known_limitations'],
    }
    write_json(directory / 'manifest.json', manifest)
    write_json(directory / 'COMPLETE.json', {
        'format': 'stage7_corpus_complete_v1',
        'manifest_sha256': file_sha256(directory / 'manifest.json'),
    })
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--documents', nargs='+', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = pack(args.config, args.documents, args.tokenizer, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
