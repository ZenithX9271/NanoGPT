"""Bounded-memory access to existing uint16 shards, including cross-shard windows."""
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(str(path), encoding='utf-8') as stream:
        return json.load(stream)


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


class TokenShards:
    def __init__(self, directory, split, verify_hashes=False):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / 'manifest.json')
        if self.manifest['dtype'] != 'uint16' or self.manifest.get('byte_order') != 'little-endian':
            raise ValueError('Expected little-endian uint16 shards')
        self.split = split
        self.arrays, self.ends = [], []
        self.n = 0
        for entry in self.manifest['splits'][split]['shards']:
            path = self.directory / entry['file']
            if path.parent.resolve() != self.directory.resolve():
                raise ValueError('Shard path must stay inside shard directory')
            if path.stat().st_size != entry['tokens']*2:
                raise ValueError('Shard byte count mismatch: ' + str(path))
            if verify_hashes and sha256(path) != entry['sha256']:
                raise ValueError('Shard SHA256 mismatch: ' + str(path))
            self.arrays.append(np.memmap(str(path), dtype='<u2', mode='r', shape=(entry['tokens'],)))
            self.n += entry['tokens']
            self.ends.append(self.n)
        if self.n != self.manifest['splits'][split]['token_count'] or not self.n:
            raise ValueError('Manifest token count mismatch or empty split')

    def read(self, start, length):
        if start < 0 or length < 1 or start+length > self.n:
            raise ValueError('Window outside split')
        index = bisect.bisect_right(self.ends, start)
        offset = start - (self.ends[index-1] if index else 0)
        pieces = []
        remaining = length
        while remaining:
            take = min(remaining, len(self.arrays[index])-offset)
            pieces.append(self.arrays[index][offset:offset+take])
            remaining -= take
            index += 1
            offset = 0
        # Only this window is materialised in int64, never the entire corpus.
        return np.concatenate(pieces).astype(np.int64)

    def batch(self, batch_size, context, generator, device):
        if self.n <= context:
            raise ValueError('Split must contain at least context+1 tokens')
        starts = torch.randint(self.n-context, (batch_size,), generator=generator).tolist()
        windows = np.stack([self.read(i, context+1) for i in starts])
        if windows.min() < 0 or windows.max() >= self.manifest['vocab_size']:
            raise ValueError('Token ID outside vocabulary')
        x = torch.from_numpy(windows[:, :-1].copy()).to(device)
        y = torch.from_numpy(windows[:, 1:].copy()).to(device)
        return x, y


class BPEDecoder:
    """Decode the saved byte-level tokenizer without a Tokenizers dependency."""
    def __init__(self, path):
        data = read_json(path)
        self.vocab = {int(v): k for k, v in data['model']['vocab'].items()}
        for token in data.get('added_tokens', []):
            self.vocab[token['id']] = token['content']
        # Use the standard GPT-2 byte-to-unicode construction explicitly.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        codes = list(visible)
        n = 0
        for b in range(256):
            if b not in visible:
                visible.append(b)
                codes.append(256+n)
                n += 1
        self.byte_decoder = {chr(c): b for b, c in zip(visible, codes)}

    def decode(self, ids, skip=(0, 1, 2)):
        text = ''.join(self.vocab[int(i)] for i in ids if int(i) not in skip)
        return bytes(self.byte_decoder[c] for c in text).decode('utf-8', errors='replace')


def validate_assets(root, cfg, verify_hashes=False):
    directory = resolve(root, cfg['data']['shard_dir'])
    manifest_path = directory / 'manifest.json'
    manifest = read_json(manifest_path)
    tokenizer = resolve(root, cfg['data']['tokenizer'])
    if sha256(tokenizer) != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer SHA256 does not match shard manifest')
    if manifest['vocab_size'] != cfg['model']['vocab_size']:
        raise ValueError('Vocabulary mismatch')
    for name in ('pad', 'bos', 'eos'):
        if manifest['special_ids'][name] != cfg['data'][name+'_id']:
            raise ValueError('Special token mismatch: ' + name)
    splits = {s: TokenShards(directory, s, verify_hashes) for s in ('train', 'dev_val')}
    prompts_path = resolve(root, cfg['evaluation']['prompts'])
    prompts = read_json(prompts_path)
    if prompts['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Prompt/tokenizer identity mismatch')
    if prompts.get('dev_offsets_sha256') is not None:
        if sha256(directory / 'dev_val_offsets.json') != prompts['dev_offsets_sha256']:
            raise ValueError('Development offsets changed since prompt preparation')
    if not prompts.get('prompts'):
        raise ValueError('No evaluation prompts')
    for p in prompts['prompts']:
        ids = p['ids']
        if not ids or ids[0] != cfg['data']['bos_id'] or any(
                not isinstance(i, int) or i < 0 or i >= cfg['model']['vocab_size'] for i in ids):
            raise ValueError('Invalid explicit prompt IDs')
        if p.get('kind') == 'development_prefix':
            actual = splits['dev_val'].read(p['offset'], len(ids)).tolist()
            if actual != ids:
                raise ValueError('Development prompt does not match shard content')
    identity = {'manifest_sha256': sha256(manifest_path),
                'tokenizer_sha256': sha256(tokenizer), 'prompts_sha256': sha256(prompts_path)}
    return splits, identity, prompts
