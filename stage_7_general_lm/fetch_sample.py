"""Explicit, capped public sample/tokenizer fetch; never fetches model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .common import read_json, write_json


def fetch(config, output):
    cfg = read_json(config)
    cap = cfg['max_network_bytes']
    if cap > 8388608 or cap < 1 or not 1 <= cfg['rows_per_source'] <= 12:
        raise ValueError('First fetch limited to 8 MiB and twelve rows/source')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    received, started = 0, time.monotonic()

    def get(url, per_file=1048576):
        nonlocal received
        if time.monotonic() - started > 120:
            raise RuntimeError('Fetch walltime budget exhausted')
        remaining = min(per_file, cap - received)
        if remaining <= 0:
            raise RuntimeError('Fetch byte budget exhausted')
        request = urllib.request.Request(url, headers={'User-Agent': 'NanoGPT-Stage7-bounded-research/1.0'})
        with urllib.request.urlopen(request, timeout=20) as stream:
            value = stream.read(remaining)
            # Refuse a response that fills its allowance; never read past the cap.
            if len(value) == remaining:
                raise RuntimeError('Response reached byte allowance; refusing possible truncation')
        received += len(value)
        return value

    documents, sources = [], []
    for source in cfg['datasets']:
        meta = json.loads(get('https://huggingface.co/api/datasets/' + source['id']))
        if meta['sha'] != source['observed_metadata_revision']:
            raise ValueError('Dataset head changed; select and review a new sample explicitly')
        url = ('https://datasets-server.huggingface.co/rows?dataset=' + source['id'].replace('/', '%2F') +
               '&config=' + source['config'] + '&split=train&offset=' + str(cfg['offset']) + '&length=' + str(cfg['rows_per_source']))
        raw = get(url, 2097152)
        fingerprint = hashlib.sha256(raw).hexdigest()
        if fingerprint != source['captured_response_sha256']:
            raise ValueError('Viewer response changed; never silently substitute another sample')
        (directory / (source['config'] + '-response.json')).write_bytes(raw)
        data = json.loads(raw)
        for item in data['rows']:
            r = item['row']; identifier = str(r.get('id', item['row_idx']))
            documents.append(dict(id=identifier, source=source['id'] + '/' + source['config'],
                revision=meta['sha'], license='odc-by (dataset card); underlying-source rights need source-level records',
                source_group=r.get('url', source['id'] + ':' + identifier), text=r['text'],
                viewer_row_index=item['row_idx'], row_response_sha256=fingerprint,
                revision_binding='Viewer revision is not independently attested'))
        sources.append(dict(source, documents=len(data['rows'])))
    native = directory / 'native_tokenizer';native.mkdir()
    prefix = 'https://huggingface.co/' + cfg['tokenizer_model'] + '/resolve/' + cfg['tokenizer_revision'] + '/'
    tokenizer = get(prefix + 'tokenizer.json', 4194304)
    if hashlib.sha256(tokenizer).hexdigest() != cfg['tokenizer_sha256']:
        raise ValueError('Pinned tokenizer SHA256 mismatch')
    (directory / 'general-tokenizer.json').write_bytes(tokenizer)
    os.link(str(directory / 'general-tokenizer.json'), str(native / 'tokenizer.json'))
    for name in ('tokenizer_config.json', 'special_tokens_map.json', 'config.json'):
        try:
            data = get(prefix + name)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        (native / name).write_bytes(data)
    with (directory / 'raw.jsonl').open('x', encoding='utf-8') as stream:
        for record in documents:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    report = dict(network_bytes=received, network_cap_bytes=cap, sources=sources,
                  seconds=time.monotonic() - started, model_weights_downloaded=False,
                  tokenizer_revision=cfg['tokenizer_revision'], limitation=cfg['limitation'])
    write_json(directory / 'download_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).parent / 'configs/sources.json'))
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(fetch(args.config, args.output), indent=2))


if __name__ == '__main__':main()
