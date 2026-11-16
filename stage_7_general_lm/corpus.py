"""Bounded local sampling, tokenizer comparison, and new token-stream packing.

No automatic corpus/model downloads. Input records must supply source provenance.
The bounded input pool is shuffled reproducibly; it is not a uniform full-web sample.
"""
import argparse
from contextlib import ExitStack
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import unicodedata

import numpy as np

from .common import read_json, sha, write_json


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def storage_dtype(maximum_id):
    if maximum_id < 0 or maximum_id > 4294967295:
        raise ValueError('Token IDs cannot be represented by supported storage')
    return '<u2' if maximum_id <= 65535 else '<u4'


def split_for(group, seed):
    value = int(digest(str(seed) + ':' + group)[:12], 16) % 10
    return 'dev' if value == 8 else 'heldout' if value == 9 else 'train'


def sample(input_path, output, max_documents=25000, max_text_bytes=1073741824,
           max_read_bytes=4294967296, max_seconds=1800, seed=1337, protected_groups=()):
    if min(max_documents, max_text_bytes, max_read_bytes, max_seconds) <= 0:
        raise ValueError('All sampling limits must be positive')
    if max_documents > 25000 or max_text_bytes > 1073741824 or max_read_bytes > 4294967296 or max_seconds > 1800:
        raise ValueError('Requested sample exceeds the first-milestone ceilings')
    start = time.monotonic()
    pool, seen, read_bytes, kept_bytes = [], set(), 0, 0
    skipped = Counter()
    consumed_hash = hashlib.sha256()
    stop = 'input_exhausted'
    with Path(input_path).open('rb') as stream:
        while True:
            if time.monotonic() - start >= max_seconds:
                stop = 'cpu_time_limit'
                break
            if read_bytes == max_read_bytes:
                stop = 'input_byte_limit'
                break
            raw = stream.readline(max_read_bytes - read_bytes)
            if not raw:
                break
            read_bytes += len(raw)
            consumed_hash.update(raw)
            if read_bytes == max_read_bytes and not raw.endswith(b'\n'):
                stop = 'input_byte_limit'
                break
            r = json.loads(raw)
            if any(not r.get(k) for k in ('id', 'source', 'revision', 'license', 'text')):
                raise ValueError('Each input needs id/source/revision/license/text')
            text = unicodedata.normalize('NFC', r['text']).strip()
            if not text:
                skipped['empty'] += 1
                continue
            normalized = digest(' '.join(text.casefold().split()))
            group = r.get('source_group') or r['source'] + ':' + str(r['id'])
            if group in protected_groups:
                skipped['evaluation_group'] += 1
                continue
            if normalized in seen:
                skipped['duplicate'] += 1
                continue
            size = len(text.encode('utf-8'))
            if kept_bytes + size > max_text_bytes:
                stop = 'retained_text_limit'
                break
            seen.add(normalized)
            record = dict(r, text=text, text_sha256=digest(text), normalized_sha256=normalized,
                          source_group=group, split=split_for(group, seed), filtering='NFC, trim, exact normalized dedup')
            pool.append(record)
            kept_bytes += size
            if len(pool) == max_documents:
                stop = 'document_limit'
                break
    random.Random(seed).shuffle(pool)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'documents.jsonl').open('x', encoding='utf-8') as stream:
        for r in pool:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    report = dict(consumed_input_sha256=consumed_hash.hexdigest(), documents=len(pool), retained_text_bytes=kept_bytes,
                  input_bytes_consumed=read_bytes, seconds=time.monotonic() - start, stop_reason=stop,
                  seed=seed, skipped=dict(skipped), source_counts=dict(Counter(r['source'] for r in pool)),
                  split_counts=dict(Counter(r['split'] for r in pool)),
                  limits=dict(documents=max_documents, text_bytes=max_text_bytes, input_bytes=max_read_bytes, seconds=max_seconds),
                  network_bytes=0, sampling='seeded shuffle of bounded input pool; no claim of full-corpus uniform sampling',
                  leakage_limit='Source groups and exact normalized hashes only; semantic derivative lineage may be incomplete')
    write_json(directory / 'sampling.json', report)
    return report


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def compare(documents, tokenizer_paths):
    records = [json.loads(line) for line in Path(documents).read_text().splitlines()]
    result = {}
    for path in tokenizer_paths:
        tokenizer = load_tokenizer(path)
        domains = {}
        for r in records:
            ids = tokenizer.encode(r['text'], add_special_tokens=False).ids
            d = domains.setdefault(r['source'], dict(documents=0, tokens=0, utf8_bytes=0, words=0, over_1024=0, roundtrip_mismatches=0))
            d['documents'] += 1
            d['tokens'] += len(ids)
            d['utf8_bytes'] += len(r['text'].encode())
            d['words'] += len(r['text'].split())
            d['over_1024'] += len(ids) > 1024
            d['roundtrip_mismatches'] += tokenizer.decode(ids, skip_special_tokens=False) != r['text']
        for d in domains.values():
            d['tokens_per_byte'] = d['tokens'] / max(1, d['utf8_bytes'])
            d['tokens_per_word'] = d['tokens'] / max(1, d['words'])
        maximum = max(tokenizer.get_vocab().values())
        result[str(path)] = dict(sha256=sha(path), vocab_entries=tokenizer.get_vocab_size(), maximum_id=maximum,
                                 required_storage_dtype=storage_dtype(maximum), domains=domains,
                                 fragments={w: tokenizer.encode(w, add_special_tokens=False).tokens for w in ('Zainab', '1048576', 'photosynthesis', "couldn't")},
                                 status='Small-sample efficiency diagnostics; no tokenizer quality or throughput winner selected')
    return result


def pack(documents, tokenizer_path, output, bos_id, eos_id):
    tokenizer = load_tokenizer(tokenizer_path)
    maximum = max(tokenizer.get_vocab().values())
    if not 0 <= bos_id <= maximum or not 0 <= eos_id <= maximum:
        raise ValueError('Native BOS/EOS IDs must exist in selected vocabulary')
    dtype = storage_dtype(maximum)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    entries, sources, split_sources = {}, Counter(), {}
    with Path(documents).open() as stream:
        domains = sorted({json.loads(line)['source'] for line in stream})
    if not domains or len(domains) > 16:
        raise ValueError('Packing requires one to sixteen explicit source families')
    counters, groups, paths = Counter(), {}, {}
    with ExitStack() as stack:
        handles = {}
        for split in ('train', 'dev', 'heldout'):
            groups[split], split_sources[split] = set(), Counter()
            for i, domain in enumerate(domains):
                path = directory / (split + '_%02d.bin' % i)
                paths[split, domain] = path
                handles[split, domain] = stack.enter_context(path.open('xb'))
        with Path(documents).open() as stream:
            for line in stream:
                r = json.loads(line)
                split, domain = r['split'], r['source']
                if split not in groups:
                    raise ValueError('Unknown corpus split')
                ids = [bos_id] + tokenizer.encode(r['text'], add_special_tokens=False).ids + [eos_id]
                handles[split, domain].write(np.asarray(ids, dtype=np.dtype(dtype)).tobytes())
                counters[split, domain] += len(ids)
                sources[domain] += len(ids)
                split_sources[split][domain] += len(ids)
                groups[split].add(r['source_group'])
    for split in groups:
        shards = [dict(file=paths[split, domain].name, tokens=counters[split, domain], sha256=sha(paths[split, domain]), source=domain)
                  for domain in domains if counters[split, domain]]
        entries[split] = dict(shards=shards, source_groups=sorted(groups[split]))
    for a in entries:
        for b in entries:
            if a != b and set(entries[a]['source_groups']) & set(entries[b]['source_groups']):
                raise ValueError('Source group crossed a corpus split')
    manifest = dict(format='stage7_token_stream_v1', vocab_size=maximum + 1, dtype=dtype,
                    tokenizer_sha256=sha(tokenizer_path), documents_sha256=sha(documents),
                    special_ids=dict(bos=bos_id, eos=eos_id), splits=entries,
                    mixture_policy='Uniform random windows over fixed concatenation; source token exposure is recorded, not actively reweighted',
                    source_token_counts=dict(sources), cross_document_attention=True)
    manifest['source_token_counts_by_split'] = {k: dict(v) for k, v in split_sources.items()}
    write_json(directory / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--input', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--max-documents', type=int, default=25000)
    s.add_argument('--max-text-bytes', type=int, default=1073741824)
    s.add_argument('--max-read-bytes', type=int, default=4294967296)
    s.add_argument('--max-seconds', type=int, default=1800)
    s.add_argument('--seed', type=int, default=1337)
    s.add_argument('--protected-groups', help='JSON list, including evaluation source groups')
    c = sub.add_parser('compare')
    c.add_argument('--documents', required=True)
    c.add_argument('--tokenizers', nargs='+', required=True)
    c.add_argument('--report', required=True)
    b = sub.add_parser('pack')
    b.add_argument('--documents', required=True)
    b.add_argument('--tokenizer', required=True)
    b.add_argument('--output', required=True)
    b.add_argument('--bos-id', type=int, required=True)
    b.add_argument('--eos-id', type=int, required=True)
    args = p.parse_args()
    if args.mode == 'sample':
        report = sample(args.input, args.output, args.max_documents, args.max_text_bytes, args.max_read_bytes,
                        args.max_seconds, args.seed, read_json(args.protected_groups) if args.protected_groups else ())
    elif args.mode == 'compare':
        report = compare(args.documents, args.tokenizers)
        write_json(args.report, report)
    else:
        report = pack(args.documents, args.tokenizer, args.output, args.bos_id, args.eos_id)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
