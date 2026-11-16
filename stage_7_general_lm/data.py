"""Manifest-pinned memory maps and token-weighted accumulation inputs."""
from pathlib import Path
import hashlib

import numpy as np
import torch

from .common import read_json, sha


class TokenStream:
    def __init__(self, manifest, split='train', verify=True, domain=None):
        self.path = Path(manifest).resolve()
        self.manifest = read_json(self.path)
        self.identity = sha(self.path)
        self.arrays = []
        info = self.manifest['splits'][split]
        self.dtype = np.dtype(self.manifest['dtype'])
        if self.dtype not in (np.dtype('<u2'), np.dtype('<u4')):
            raise ValueError('Only explicit little-endian uint16/uint32 streams are supported')
        for entry in info['shards']:
            if domain is not None and entry.get('source') != domain:
                continue
            p = self.path.parent / entry['file']
            if p.stat().st_size != entry['tokens'] * self.dtype.itemsize or (verify and sha(p) != entry['sha256']):
                raise ValueError('Shard identity differs: ' + str(p))
            a = np.memmap(str(p), dtype=self.dtype, mode='r')
            if len(a) and int(a.max()) >= self.manifest['vocab_size']:
                raise ValueError('Shard token outside vocabulary')
            self.arrays.append(a)
        self.ends = np.cumsum([len(a) for a in self.arrays])
        self.size = int(self.ends[-1]) if len(self.ends) else 0

    def read(self, start, length):
        if start < 0 or length < 0 or start + length > self.size:
            raise ValueError('Window outside stream')
        result = []
        while length:
            i = int(np.searchsorted(self.ends, start, side='right'))
            begin = start - (int(self.ends[i - 1]) if i else 0)
            take = min(length, len(self.arrays[i]) - begin)
            result.append(self.arrays[i][begin:begin + take])
            start, length = start + take, length - take
        return np.concatenate(result).astype(np.int64) if result else np.empty(0, dtype=np.int64)

    def batch(self, size, context, generator, device):
        if self.size <= context:
            raise ValueError('Split too short for context+1')
        starts = torch.randint(self.size - context, (size,), generator=generator).tolist()
        data = torch.from_numpy(np.stack([self.read(i, context + 1) for i in starts])).to(device)
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()


def assert_disjoint(training, heldout):
    for field in ('source_group', 'text_sha256'):
        left = {r[field] for r in training if r.get(field)}
        right = {r[field] for r in heldout if r.get(field)}
        if left & right:
            raise ValueError('Source leakage via ' + field)
    def text_hash(record):
        text = record.get('text') or ' '.join(s['text'] for s in record.get('passage', []))
        return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest() if text else None
    left = {text_hash(r) for r in training} - {None}
    right = {text_hash(r) for r in heldout} - {None}
    if left & right:
        raise ValueError('Source leakage via normalized passage text')
