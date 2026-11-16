"""Hash-pinned, bounded Parquet-to-JSONL extraction for the Stage 7 corpus.

This module deliberately supports the HPC's Python 3.6 / PyArrow 0.12 reader.
It does not import torch, tokenizers, datasets, or Stage 7 common helpers.
"""
from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import unicodedata


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


def compact_identity(prefix, value):
    raw = str(value).encode('utf-8')
    return prefix + ':' + hashlib.sha256(raw).hexdigest()


def first_value(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def extract(config_path, input_root, output):
    """Verify source files and extract bounded provenance-bearing JSONL."""
    import pyarrow.parquet as parquet

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('format') != 'stage7_production_sources_v1':
        raise ValueError('Unknown production source configuration')
    if not config.get('sources'):
        raise ValueError('At least one source is required')
    expected_network_bytes = sum(item.get('expected_bytes', 0)
                                 for item in config['sources'])
    if (expected_network_bytes <= 0 or
            expected_network_bytes > config.get('network_cap_bytes', 0)):
        raise ValueError('Pinned source sizes exceed or omit the network cap')
    limits = config['extraction_limits']
    if min(limits['max_documents'], limits['max_output_bytes'],
           limits['max_seconds']) <= 0:
        raise ValueError('Extraction limits must be positive')

    input_root = Path(input_root).resolve()
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    documents_path = directory / 'documents.jsonl'
    started = time.monotonic()
    total_documents = total_text_bytes = output_bytes = 0
    source_reports = []
    stop_reason = 'input_exhausted'

    with documents_path.open('x', encoding='utf-8') as destination:
        for source in config['sources']:
            required = ('name', 'dataset_id', 'revision', 'relative_path',
                        'sha256', 'expected_bytes', 'license', 'text_column',
                        'max_text_bytes')
            if any(not source.get(key) for key in required):
                raise ValueError('Incomplete source entry: ' + repr(source))
            path = (input_root / source['relative_path']).resolve()
            try:
                path.relative_to(input_root)
            except ValueError:
                raise ValueError('Source path escapes input root: ' + str(path))
            if not path.is_file():
                raise ValueError('Pinned Parquet file missing: ' + str(path))
            if path.stat().st_size != source['expected_bytes']:
                raise ValueError('Pinned Parquet byte size mismatch: ' + source['name'])
            actual_sha = file_sha256(path)
            if actual_sha != source['sha256']:
                raise ValueError('Pinned Parquet SHA256 mismatch: ' + source['name'])

            reader = parquet.ParquetFile(str(path))
            schema_names = list(reader.schema.names)
            if source['text_column'] not in schema_names:
                raise ValueError('Text column absent for ' + source['name'])
            candidates = ([source['text_column']] + source.get('id_columns', [])
                          + source.get('group_columns', []))
            columns = []
            for name in candidates:
                if name in schema_names and name not in columns:
                    columns.append(name)
            source_documents = source_text_bytes = source_rows = 0
            skipped = Counter()
            exhausted = False

            for group_index in range(reader.num_row_groups):
                if time.monotonic() - started >= limits['max_seconds']:
                    stop_reason = 'time_limit'
                    exhausted = True
                    break
                table = reader.read_row_group(group_index, columns=columns)
                values = table.to_pydict()
                for local_index in range(table.num_rows):
                    source_rows += 1
                    row = {name: values[name][local_index] for name in columns}
                    raw_text = row.get(source['text_column'])
                    if raw_text is None:
                        skipped['null_text'] += 1
                        continue
                    text = unicodedata.normalize('NFC', str(raw_text)).strip()
                    if not text:
                        skipped['empty_text'] += 1
                        continue
                    text_bytes = len(text.encode('utf-8'))
                    if source_text_bytes + text_bytes > source['max_text_bytes']:
                        stop_reason = 'source_text_limit'
                        exhausted = True
                        break
                    if total_documents >= limits['max_documents']:
                        stop_reason = 'document_limit'
                        exhausted = True
                        break

                    global_row = source_rows - 1
                    raw_id = first_value(row, source.get('id_columns', []))
                    if raw_id is None:
                        raw_id = actual_sha + ':' + str(global_row)
                    raw_group = first_value(row, source.get('group_columns', []))
                    if raw_group is None:
                        raw_group = actual_sha + ':' + str(global_row)
                    record = {
                        'id': compact_identity(source['name'] + '-id', raw_id),
                        'source': source['name'],
                        'dataset_id': source['dataset_id'],
                        'revision': source['revision'],
                        'license': source['license'],
                        'source_file': source['relative_path'],
                        'source_file_sha256': actual_sha,
                        'source_row': global_row,
                        'source_group': compact_identity(source['name'] + '-group', raw_group),
                        'text': text,
                    }
                    encoded = (json.dumps(record, ensure_ascii=False,
                                          sort_keys=True) + '\n')
                    encoded_bytes = len(encoded.encode('utf-8'))
                    if output_bytes + encoded_bytes > limits['max_output_bytes']:
                        stop_reason = 'output_byte_limit'
                        exhausted = True
                        break
                    destination.write(encoded)
                    output_bytes += encoded_bytes
                    total_documents += 1
                    source_documents += 1
                    total_text_bytes += text_bytes
                    source_text_bytes += text_bytes
                if exhausted:
                    break

            source_reports.append({
                'name': source['name'],
                'documents': source_documents,
                'text_bytes': source_text_bytes,
                'rows_scanned': source_rows,
                'row_groups_total': reader.num_row_groups,
                'sha256': actual_sha,
                'skipped': dict(skipped),
                'stopped_early': exhausted,
            })
            if stop_reason in ('time_limit', 'document_limit', 'output_byte_limit'):
                break
            stop_reason = 'input_exhausted'
        destination.flush()
        os.fsync(destination.fileno())

    report = {
        'format': 'stage7_parquet_extraction_v1',
        'config_sha256': file_sha256(config_path),
        'documents_sha256': file_sha256(documents_path),
        'documents': total_documents,
        'text_bytes': total_text_bytes,
        'output_bytes': output_bytes,
        'seconds': time.monotonic() - started,
        'stop_reason': stop_reason,
        'sources': source_reports,
        'limits': limits,
        'policy': ('Pinned-file verification; NFC and trim only. Deduplication, '
                   'splitting, and token caps occur during packing.'),
    }
    write_json(directory / 'extraction.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.config, args.input_root, args.output),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
