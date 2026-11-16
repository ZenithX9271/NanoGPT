"""Read-only checkpoint diagnostics, kept outside frozen training source identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


TOKENIZER_SHA = '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'
PROMPTS = (
    'Plants use sunlight to make food through photosynthesis. During this process,',
    'The water cycle moves water between the land, oceans, and atmosphere. Evaporation occurs when',
    'A useful way to understand gravity is to imagine',
    'To compare two fractions with different denominators, first',
    'Leena planted a seed in a small pot. She watered it each morning. After two weeks,',
    'The library was about to close when Daniel found a letter inside an old book. He',
    'A metal spoon feels colder than a wooden spoon in the same room because',
    'When writing a clear explanation, it helps to begin with',
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def tokenizer(path):
    from tokenizers import Tokenizer
    if digest(path) != TOKENIZER_SHA:
        raise ValueError('Tokenizer checksum differs')
    result = Tokenizer.from_file(str(path))
    if result.get_vocab_size() != 49152:
        raise ValueError('Tokenizer vocabulary differs')
    return result


def prepare(path, output):
    import tokenizers
    native = tokenizer(path)
    probes = []
    for index, text in enumerate(PROMPTS):
        ids = native.encode(text, add_special_tokens=False).ids
        assert native.decode(ids, skip_special_tokens=False) == text
        assert 0 < len(ids) + 1 <= 1024 - 96
        probes.append(dict(id=index, prompt=text, input_ids=[1] + ids))
    publish(output, dict(format='stage7_completion_probes_v1',
                        tokenizer_sha256=TOKENIZER_SHA, tokenizers=tokenizers.__version__,
                        new_tokens=96, eos_id=2, probes=probes))


def score_stream(model, data, windows, batch_size, context, device, seed, deadline=None):
    import torch
    if min(windows, batch_size, context) <= 0:
        raise ValueError('Positive evaluation dimensions required')
    generator = torch.Generator().manual_seed(seed)
    total, positions = 0., 0
    with torch.no_grad():
        for start in range(0, windows, batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Review evaluation time budget exhausted')
            count = min(batch_size, windows - start)
            x, y = data.batch(count, context, generator, device)
            _, loss = model(x, y, reduction='sum')
            value = float(loss)
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite development loss')
            total += value
            positions += y.numel()
    ce = total / positions
    return dict(ce=ce, perplexity=math.exp(min(700., ce)), positions=positions,
                windows=windows, seed=seed)


def sampled_completion(model, ids, new_tokens, eos_id, seed):
    import torch
    if ids.size(1) + new_tokens > model.cfg['block_size']:
        raise ValueError('Completion exceeds model context')
    torch.manual_seed(seed)
    with torch.no_grad():
        logits, _, cache = model(ids, use_cache=True)
        for index in range(new_tokens):
            values, indices = torch.topk(logits[:, -1].float() / .8,
                                        min(50, logits.size(-1)), dim=-1)
            choice = torch.multinomial(torch.softmax(values, dim=-1), 1)
            token = indices.gather(-1, choice)
            ids = torch.cat((ids, token), dim=1)
            if bool(token.eq(eos_id).all()):
                break
            if index + 1 < new_tokens:
                logits, _, cache = model(token, past=cache, use_cache=True)
    return ids


def compare_reference(current, reference, maximum_regression=.05):
    """Compare like-for-like probes; fail before authorizing more optimization."""
    if not math.isfinite(maximum_regression) or maximum_regression < 0:
        raise ValueError('Invalid maximum evaluation regression')
    for name in ('config_sha256', 'manifest_sha256', 'code_identity', 'tokenizer_sha256', 'precision'):
        if current[name] != reference[name]:
            raise ValueError('Reference identity differs: ' + name)
    if reference['status'] != 'PASS' or set(current['expanded_scores']) != set(reference['expanded_scores']):
        raise ValueError('Reference review is incomplete')
    changes = {}
    for name, score in current['expanded_scores'].items():
        old = reference['expanded_scores'][name]
        for field in ('seed', 'windows', 'positions'):
            if score[field] != old[field]:
                raise ValueError('Reference sampling differs: ' + name + ':' + field)
        if not math.isfinite(score['ce']) or not math.isfinite(old['ce']):
            raise ValueError('Nonfinite reference comparison')
        changes[name] = score['ce'] - old['ce']
        if name.startswith('dev') and changes[name] > maximum_regression:
            raise ValueError('Development regression exceeds limit: ' + name)
    return dict(reference_checkpoint_sha256=reference['checkpoint_sha256'],
                maximum_ce_regression=maximum_regression, ce_changes=changes)


def review(args):
    import torch
    from stage_7_general_lm.common import configure, sha, source_identity
    from stage_7_general_lm.data import TokenStream
    from stage_7_general_lm.train import Engine

    started = time.monotonic()
    device = configure(args.device, 2)
    record = read(args.checkpoint_record)
    assert digest(args.checkpoint) == record['checkpoint_sha256']
    cfg = read(args.config)
    train, dev = TokenStream(args.manifest), TokenStream(args.manifest, 'dev')
    engine = Engine(cfg, train, dev, device)
    engine.resume(args.checkpoint)
    expected = record['summary']['counters']
    for name in ('updates', 'attempts', 'successful_positions', 'processed_positions'):
        assert getattr(engine, name) == expected[name], 'Checkpoint counter differs: ' + name
    heldout_windows = getattr(args, 'heldout_windows', 0)
    if heldout_windows < 0:
        raise ValueError('Heldout windows cannot be negative')
    if heldout_windows and engine.updates != engine.horizon:
        raise ValueError('Heldout scoring is reserved for the original completed horizon')
    assert all(torch.isfinite(p).all().item() for p in engine.model.parameters())
    # Reproduce the original small evaluation before increasing its sample size.
    with Path(args.metrics).open() as metrics:
        old = [r for r in (json.loads(line) for line in metrics)
               if r['event'] == 'evaluation' and r['updates'] == engine.updates][-1]
    reproduced = engine.evaluate()
    for name in old['scores']:
        assert abs(old['scores'][name]['ce'] - reproduced['scores'][name]['ce']) < 1e-5
    engine.model.eval()
    streams = [('train', train), ('dev', dev)]
    domains = sorted({r['source'] for r in dev.manifest['splits']['dev']['shards']})
    streams += [('dev_domain:' + name, TokenStream(args.manifest, 'dev',
                 verify=False, domain=name)) for name in domains]
    scores = {}
    for index, (name, data) in enumerate(streams):
        scores[name] = score_stream(engine.model, data, args.windows, 2,
                                   cfg['model']['block_size'], device,
                                   20261111 + index, started + 600)
        print('EXPANDED_EVALUATION', name, json.dumps(scores[name]), flush=True)
    heldout = None
    if heldout_windows:
        heldout = score_stream(engine.model, TokenStream(args.manifest, 'heldout'),
                               heldout_windows, 2, cfg['model']['block_size'], device,
                               20261115, started + 600)
        print('FINAL_HELDOUT_EVALUATION', json.dumps(heldout), flush=True)
    # Broad numerical sanity checks, not assistant-quality acceptance thresholds.
    assert scores['dev']['ce'] < old['scores']['dev']['ce'] + .5
    assert scores['dev']['ce'] - scores['train']['ce'] < .5
    probes = read(args.probes)
    assert probes['tokenizer_sha256'] == dev.manifest['tokenizer_sha256'] == TOKENIZER_SHA
    completions = []
    for probe in probes['probes']:
        ids = torch.tensor([probe['input_ids']], dtype=torch.long, device=device)
        prefix = ids.size(1)
        greedy = engine.model.greedy(ids, probes['new_tokens'], eos_id=probes['eos_id'])
        sampled = sampled_completion(engine.model, ids, probes['new_tokens'],
                                     probes['eos_id'], 20261111 + probe['id'])
        completions.append(dict(id=probe['id'], prompt=probe['prompt'],
            greedy_ids=greedy[0, prefix:].tolist(), sampled_ids=sampled[0, prefix:].tolist()))
    result = dict(format='stage7_checkpoint_review_v1', status='PASS',
        quality_claim='Numerical/resume sanity only; completions are qualitative diagnostics.',
        checkpoint_sha256=record['checkpoint_sha256'], config_sha256=sha(args.config),
        manifest_sha256=train.identity, code_identity=source_identity(),
        counters=dict(updates=engine.updates, successful_positions=engine.successful_positions),
        precision='fp32', original_evaluation_reproduced=True,
        expanded_scores=scores, completions=completions,
        decoding=dict(greedy=True, sampled_temperature=.8, sampled_top_k=50,
                      sampled_seed_base=20261111, new_tokens=96, eos_id=2),
        tokenizer_sha256=TOKENIZER_SHA, seconds=time.monotonic() - started)
    if getattr(args, 'reference_review', None):
        result['reference_comparison'] = compare_reference(
            result, read(args.reference_review), args.max_dev_regression)
    if heldout is not None:
        result['heldout_score'] = heldout
        result['heldout_policy'] = 'Scored only at the original completed horizon; not used for development admission.'
    publish(args.output, result)
    print('STAGE7_EXPANDED_REVIEW_PASS', flush=True)


def decode(path, input_path, output):
    native = tokenizer(path)
    result = read(input_path)
    assert result['tokenizer_sha256'] == TOKENIZER_SHA
    for row in result['completions']:
        for mode in ('greedy', 'sampled'):
            row[mode + '_text'] = native.decode(row[mode + '_ids'], skip_special_tokens=False)
            print('PROBE', row['id'], mode, row['prompt'] + row[mode + '_text'], flush=True)
    publish(output, result)


def finalize(args):
    import os
    import shutil
    run, evidence = Path(args.run), Path(args.output)
    summary = read(run / 'summary.json')
    counters = summary['counters']
    start = getattr(args, 'start_updates', 6104)
    target = getattr(args, 'target_updates', 12208)
    expected_reason = 'review_gate'
    completion_config = getattr(args, 'completion_config', None)
    if completion_config:
        cfg = read(completion_config)
        positions = (cfg['training']['micro_batch_size'] *
                     cfg['training']['accumulation_steps'] * cfg['model']['block_size'])
        budget = cfg['training']['token_budget']
        if positions <= 0 or budget <= 0 or target != (budget + positions - 1) // positions:
            raise ValueError('Completed export target must equal the original configured horizon')
        expected_reason = 'completed'
    assert start < counters['updates'] <= target
    if summary['reason'] == 'review_gate':
        assert counters['updates'] == target
    assert counters['successful_positions'] == counters['updates'] * 16384
    frozen = run / ('review_u%d_positions_%d.pt' % (counters['updates'], counters['successful_positions']))
    os.link(str(run / 'latest.pt'), str(frozen))
    publish(evidence / 'checkpoint.json', dict(checkpoint=str(frozen),
        checkpoint_sha256=digest(frozen), checkpoint_bytes=frozen.stat().st_size,
        summary=summary))
    for name in ('summary.json', 'metrics.jsonl'):
        shutil.copyfile(str(run / name), str(evidence / name))
    print('STAGE7_SEGMENT_SAVED', str(frozen), flush=True)
    if summary['reason'] != expected_reason or counters['updates'] != target:
        raise RuntimeError('Checkpoint preserved, but target not reached; dependent training must not start')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    prep = sub.add_parser('prepare')
    prep.add_argument('--tokenizer', required=True)
    prep.add_argument('--output', required=True)
    dec = sub.add_parser('decode')
    dec.add_argument('--tokenizer', required=True)
    dec.add_argument('--input', required=True)
    dec.add_argument('--output', required=True)
    rev = sub.add_parser('review')
    for name in ('checkpoint', 'checkpoint-record', 'config', 'manifest', 'metrics', 'probes', 'output'):
        rev.add_argument('--' + name, required=True)
    rev.add_argument('--device', default='cuda')
    rev.add_argument('--windows', type=int, default=512)
    rev.add_argument('--reference-review')
    rev.add_argument('--max-dev-regression', type=float, default=.05)
    rev.add_argument('--heldout-windows', type=int, default=0)
    fin = sub.add_parser('finalize')
    fin.add_argument('--run', required=True)
    fin.add_argument('--output', required=True)
    fin.add_argument('--start-updates', type=int, default=6104)
    fin.add_argument('--target-updates', type=int, default=12208)
    fin.add_argument('--completion-config', help='Accept completed only at this original config horizon')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.tokenizer, args.output)
    elif args.command == 'decode': decode(args.tokenizer, args.input, args.output)
    elif args.command == 'review': review(args)
    elif args.command == 'finalize': finalize(args)
    else: parser.error('Subcommand required')


if __name__ == '__main__':
    main()
