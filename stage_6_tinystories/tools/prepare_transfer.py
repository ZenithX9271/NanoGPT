#!/usr/bin/env python
"""Local-only packaging: exact baseline, physical raw files, inventory and SHA ledger.

Never starts training, submits jobs, downloads data or contacts the HPC.
Does not delete historical outputs/checkpoints. Re-run after editing transfer files.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import verify_transfer as vt


def atomic_copy(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.'+destination.name+'.', dir=str(destination.parent))
    os.close(handle)
    temporary = Path(name)
    try:
        shutil.copyfile(str(source), str(temporary))
        if vt.sha256(temporary) != expected:
            raise ValueError('Copied bytes do not match genuine source: '+str(destination))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_baseline(root):
    source = vt.inside(root, vt.BASELINES[0])
    destination = vt.inside(root, vt.BASELINES[1])
    if vt.sha256(source) != vt.BASELINE_SHA:
        raise ValueError('Stage 5 best.pt is not the pinned checkpoint; original left unchanged')
    if destination.exists():
        if vt.sha256(destination) != vt.BASELINE_SHA:
            raise ValueError('Unexpected existing Stage 6 baseline; refusing to overwrite it')
        return 'already_verified'
    atomic_copy(source, destination, vt.BASELINE_SHA)
    return 'copied_exact_stage5_checkpoint'


def materialize_raw(root):
    provenance = vt.read_json(root/'data/tinystories/v2/provenance.json')
    result = []
    for filename, key in [('TinyStories-train.txt', 'train_raw_sha256'),
                          ('TinyStories-valid.txt', 'valid_raw_sha256')]:
        path = root/'data/raw'/filename
        if not path.exists():
            raise FileNotFoundError('Genuine raw source missing: '+str(path))
        expected = provenance['source'][key]
        if vt.sha256(path) != expected:
            raise ValueError('Raw source differs from recorded dataset provenance: '+filename)
        was_link = path.is_symlink()
        if was_link:
            atomic_copy(path.resolve(strict=True), path, expected)
        result.append({'path': path.relative_to(root).as_posix(), 'sha256': expected,
                       'bytes': path.stat().st_size, 'materialized_external_symlink': was_link})
    return result


def category(name):
    if name in vt.BASELINES:
        return 'required_baseline'
    if name.startswith('data/shards/v2/'):
        return 'prepared_shards_manifest_and_offsets'
    if name.startswith('data/tinystories/v2/'):
        return 'genuine_prepared_text_and_provenance'
    if name.startswith('data/tokenizer/v2/'):
        return 'required_tokenizer_identity'
    if name.startswith('stage_6_tinystories/'):
        return 'stage6_code_configuration_tests_and_documentation'
    return 'project_sources_and_documentation'


def prepare(root, materialize=False):
    root = Path(root).resolve()
    baseline = ensure_baseline(root)
    raw = materialize_raw(root) if materialize else []
    vt.validate_dependencies(root)
    names = vt.payload_files(root)
    records = {name: {'bytes': vt.inside(root, name).stat().st_size,
                       'sha256': vt.sha256(vt.inside(root, name)), 'category': category(name)} for name in names}
    # Control files are listed separately to avoid a self-referential hash cycle.
    inventory = {'format': 'nanogpt_hpc_transfer_v1', 'files': records,
                 'control_files': [vt.FILES, vt.SUMS, vt.INVENTORY],
                 'baseline_sha256': vt.BASELINE_SHA,
                 'root_policy': 'All paths repository-relative; physical files only. Raw source is optional for HPC and excluded from payload.',
                 'preparation_scope': 'Baseline copy, optional raw materialization and transfer control files only; this tool never edits training code/configuration.'}
    for name in (vt.INVENTORY, vt.FILES, vt.SUMS):
        (root/name).parent.mkdir(parents=True, exist_ok=True)
    (root/vt.INVENTORY).write_text(json.dumps(inventory, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    paths = sorted(names+[vt.FILES, vt.SUMS, vt.INVENTORY])
    (root/vt.FILES).write_text('\n'.join(paths)+'\n', encoding='utf-8')
    sums = [vt.sha256(root/name)+'  '+name for name in paths if name != vt.SUMS]
    (root/vt.SUMS).write_text('\n'.join(sums)+'\n', encoding='utf-8')
    return {'baseline': baseline, 'raw_sources': raw, 'files_in_transfer': len(paths),
            'payload_bytes': sum(r['bytes'] for r in records.values()), 'missing_required_files': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=str(vt.ROOT))
    parser.add_argument('--materialize-raw', action='store_true', help='Replace external raw-data links with exact verified local bytes')
    args = parser.parse_args()
    try:
        result = prepare(args.project_root, args.materialize_raw)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, 'TRANSFER PREPARATION FAILED: '+str(error)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
