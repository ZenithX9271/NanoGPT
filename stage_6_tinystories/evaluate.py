#!/usr/bin/env python
"""Same fixed development windows and explicit prompts for Stage 5/6 comparison."""
import os
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import csv
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from stage_6_tinystories.core import (Engine, FORMAT, EvaluationInterrupted, assert_tied_state,
    configure_device, evaluate, load_checkpoint, runtime_info, synchronize, write_json)
from stage_6_tinystories.data import BPEDecoder, read_json, resolve, sha256, validate_assets
from stage_6_tinystories.train import Deadline, safe_run_dir


def repetition_fraction(ids, n=4):
    grams = [tuple(ids[i:i+n]) for i in range(max(0, len(ids)-n+1))]
    return 0. if not grams else 1.-len(set(grams))/len(grams)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=str(Path(__file__).parent/'configs/continue.json'))
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--output-dir', required=True, help='New empty output directory, outside Stage 5')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--windows', type=int, help='Override development windows; report records evaluation spec')
    parser.add_argument('--max-prompts', type=int, help='Smoke check only; omit for complete comparison')
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument('--walltime-seconds', type=float)
    parser.add_argument('--job-start-epoch', type=float)
    args = parser.parse_args()
    for value in (args.cpu_threads, args.windows, args.max_prompts, args.max_new_tokens):
        if value is not None and value < 1:
            parser.error('Counts must be positive')
    root = Path(args.project_root).resolve()
    cfg = read_json(resolve(root, args.config))
    if args.windows:
        cfg['evaluation']['windows'] = args.windows
    if args.max_new_tokens:
        cfg['evaluation']['max_new_tokens'] = args.max_new_tokens
    device = configure_device(args.device, args.cpu_threads)
    deadline = Deadline(args.walltime_seconds, cfg['safety']['margin_seconds'], args.job_start_epoch)
    signal.signal(signal.SIGTERM, deadline.receive)
    signal.signal(signal.SIGINT, deadline.receive)
    datasets, identity, prompts = validate_assets(root, cfg)
    engine = Engine(cfg, device, datasets, identity)
    path = resolve(root, args.checkpoint)
    checkpoint = load_checkpoint(path)
    if checkpoint.get('format') == FORMAT:
        if checkpoint['config']['model'] != cfg['model'] or checkpoint['data_identity'] != identity:
            raise ValueError('Checkpoint architecture or data identity differs from evaluator')
        engine.base_tokens, engine.base_updates = checkpoint['base_tokens'], checkpoint['base_updates']
        engine.run_updates = checkpoint['stage6_updates']
    else:
        if any(checkpoint['config'].get(k) != v for k, v in cfg['model'].items()):
            raise ValueError('Stage 5 architecture mismatch')
        engine.base_tokens, engine.base_updates = checkpoint['tokens'], checkpoint['step']
    assert_tied_state(checkpoint['model'])
    engine.model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    out = safe_run_dir(root, args.output_dir)
    summary = {'checkpoint': str(path), 'checkpoint_sha256': sha256(path), 'runtime': runtime_info(device),
               'evaluation_spec': cfg['evaluation'], 'data_identity': identity,
               'generation_complete': False, 'scored_split': 'dev_val',
               'note': 'Repetition/EOS are diagnostics; grammar, consistency and prompt retention need human ratings.'}
    begin = time.perf_counter()
    try:
        summary['evaluation'] = evaluate(engine, stop=lambda: deadline.stop())
    except EvaluationInterrupted:
        summary['status'] = 'walltime_during_ce_no_complete_score'
        write_json(summary, out/'evaluation.json')
        print(json.dumps(summary, indent=2), flush=True)
        return
    synchronize(device)
    summary['evaluation_seconds'] = time.perf_counter()-begin
    summary['status'] = 'ce_complete'
    write_json(summary, out/'evaluation.json')
    print('Development CE: {:.6f}; perplexity: {:.3f}'.format(
        summary['evaluation']['dev_val']['ce'], summary['evaluation']['dev_val']['perplexity']), flush=True)
    if args.skip_generation:
        return
    decoder = BPEDecoder(resolve(root, cfg['data']['tokenizer']))
    selected = prompts['prompts'][:args.max_prompts] if args.max_prompts else prompts['prompts']
    columns = ['prompt_id', 'kind', 'seed', 'prompt', 'continuation', 'stop', 'new_tokens',
               'repeated_4gram_fraction', 'grammar_0_to_4', 'character_consistency_0_to_4',
               'prompt_facts_0_to_4', 'ending_0_to_4']
    count, eos_count, repetition_sum, total_length = 0, 0, 0., 0
    ec = cfg['evaluation']
    with open(str(out/'stories.jsonl'), 'w', encoding='utf-8') as story_file, \
            open(str(out/'ratings.csv'), 'w', encoding='utf-8', newline='') as rating_file:
        writer = csv.DictWriter(rating_file, fieldnames=columns)
        writer.writeheader()
        for prompt in selected:
            for seed in ec['generation_seeds']:
                if deadline.stop():
                    break
                generator = torch.Generator(device=device).manual_seed(seed)
                context = torch.tensor([prompt['ids']], dtype=torch.long, device=device)
                generated, reason = engine.model.generate(context, ec['max_new_tokens'], generator,
                    pad_id=cfg['data']['pad_id'], bos_id=cfg['data']['bos_id'], eos_id=cfg['data']['eos_id'],
                    temperature=ec['temperature'], top_k=ec['top_k'], stop=lambda: deadline.stop())
                ids = generated[0, len(prompt['ids']):].tolist()
                if cfg['data']['eos_id'] in ids:
                    ids = ids[:ids.index(cfg['data']['eos_id'])]
                row = dict(prompt_id=prompt['id'], kind=prompt['kind'], seed=seed,
                    prompt=decoder.decode(prompt['ids']), continuation=decoder.decode(ids),
                    stop=reason, new_tokens=len(ids), repeated_4gram_fraction=repetition_fraction(ids))
                story_file.write(json.dumps(dict(row, generated_ids=ids), ensure_ascii=False)+'\n')
                story_file.flush()
                writer.writerow(row)
                rating_file.flush()
                count += 1
                eos_count += reason == 'eos'
                repetition_sum += row['repeated_4gram_fraction']
                total_length += len(ids)
                print('Story {} prompt={} seed={} stop={} tokens={}'.format(
                    count, prompt['id'], seed, reason, len(ids)), flush=True)
    expected = len(selected)*len(ec['generation_seeds'])
    summary.update(generation_complete=count == expected and not deadline.stop(),
        generated_samples=count, expected_samples=expected,
        eos_fraction=eos_count/max(1, count), mean_repeated_4gram_fraction=repetition_sum/max(1, count),
        mean_new_tokens=total_length/max(1, count), total_seconds=time.perf_counter()-begin)
    summary['status'] = 'complete' if summary['generation_complete'] else 'partial_generation_walltime'
    write_json(summary, out/'evaluation.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
