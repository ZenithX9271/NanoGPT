"""Local-cache-only native-template baseline and bounded LoRA/SFT entry points.

No model-weight downloads or environment installations. An installed Transformers
package or a random small configuration is not proof of 1.7B compatibility.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import time

import torch

from .common import configure, runtime, sha, write_json, seed_all, rng_state, source_identity
from .data import assert_disjoint
from .evaluation import prompt, rows, score, validate_cases


def response_example(tokenizer, case, answer, max_length=1024):
    messages = [{'role': 'system', 'content': 'Answer using only the supplied passage.'},
                {'role': 'user', 'content': prompt(case)}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': answer}],
                                              tokenize=True, add_generation_prompt=False)
    if complete[:len(prefix)] != prefix:
        raise ValueError('Native template is not prefix-stable; response masking requires explicit alignment')
    if len(complete) > max_length or len(complete) == len(prefix):
        raise ValueError('Complete source/response must fit; truncation is refused')
    return dict(input_ids=complete, labels=[-100] * len(prefix) + complete[len(prefix):])


def collate(examples, pad_id):
    if not examples or pad_id is None:
        raise ValueError('Examples and native pad/EOS fallback required')
    length = max(len(r['input_ids']) for r in examples)
    inputs, labels, masks = [], [], []
    for r in examples:
        n = len(r['input_ids'])
        inputs.append(r['input_ids'] + [pad_id] * (length - n))
        labels.append(r['labels'] + [-100] * (length - n))
        masks.append([1] * n + [0] * (length - n))
    return {k: torch.tensor(v, dtype=torch.long) for k, v in
            dict(input_ids=inputs, labels=labels, attention_mask=masks).items()}


def snapshot(model_id, revision, max_weight_bytes):
    if Path(model_id).is_dir():
        path = Path(model_id).resolve()
    else:
        if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('An immutable 40-character model revision is required')
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    weights = sorted(path.glob('*.safetensors')) or sorted(path.glob('pytorch_model*.bin'))
    size = sum(p.stat().st_size for p in weights)
    if not weights or size > max_weight_bytes:
        raise ValueError('Missing weights or checkpoint exceeds explicit weight-byte allowance')
    return path, size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['baseline', 'sft'], required=True)
    p.add_argument('--model', default='HuggingFaceTB/SmolLM2-1.7B-Instruct')
    p.add_argument('--revision')
    p.add_argument('--cases', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--max-weight-bytes', type=int, required=True)
    p.add_argument('--max-ram-bytes', type=int, required=True)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--max-examples', type=int, default=16)
    p.add_argument('--max-updates', type=int, default=2)
    p.add_argument('--walltime-seconds', type=int, default=120)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--protected-cases', default=str(Path(__file__).parent / 'evaluation/dev_draft.jsonl'))
    p.add_argument('--allow-draft-training', action='store_true')
    args = p.parse_args()
    if not 0 < args.max_examples <= 50 or not 0 < args.max_updates <= 100 or not 0 < args.walltime_seconds <= 1800 or args.max_length < 2 or args.max_new_tokens < 1:
        p.error('This entry point is bounded to 50 examples / 100 updates / 30 minutes')
    if args.mode == 'sft' and importlib.util.find_spec('peft') is None:
        p.error('PEFT is unavailable; prepare a separately pinned environment before SFT')
    device = configure(args.device)
    seed_all(args.seed)
    if args.precision == 'fp16' and device.type != 'cuda':
        p.error('Use FP32 CPU reference; FP16 path needs validated CUDA')
    destination = Path(args.output)
    if destination.exists():
        p.error('Output must be a new directory')
    path, weight_bytes = snapshot(args.model, args.revision, args.max_weight_bytes)
    import psutil
    if min(args.max_ram_bytes, psutil.virtual_memory().available) < 3 * weight_bytes:
        p.error('Loading/conversion reserve exceeds explicit/available host memory')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        p.error('Native chat template required')
    cases = rows(args.cases)[:args.max_examples]
    if not cases:
        p.error('At least one example is required')
    if args.mode == 'baseline':
        validate_cases(cases)
    else:
        if any(r.get('split') != 'train' for r in cases):
            p.error('SFT needs separately authored training groups')
        if not args.allow_draft_training and any(r.get('review_status') != 'human_approved' for r in cases):
            p.error('Training references need review, or explicit draft-pilot labeling')
        assert_disjoint(cases, rows(args.protected_cases))
    dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                trust_remote_code=False, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = dict(mode=args.mode, external_pretrained_model=args.model, revision=args.revision,
                      config_sha256=sha(path / 'config.json'), cases_sha256=sha(args.cases),
                      runtime=runtime(device), precision=args.precision, native_chat_template=tokenizer.chat_template,
                      weights_bytes=weight_bytes, claim='Development compatibility/pilot only; product selection thresholds and complete development set are not locked')
    provenance.update(seed=args.seed, code_identity=source_identity())
    started = time.monotonic()
    if args.mode == 'baseline':
        model.eval()
        predictions = []
        for case in cases:
            if time.monotonic() - started >= args.walltime_seconds:
                break
            ids = tokenizer.apply_chat_template([{'role': 'system', 'content': 'Answer using only the supplied passage.'},
                {'role': 'user', 'content': prompt(case)}], tokenize=True, add_generation_prompt=True)
            if len(ids) + args.max_new_tokens > args.max_length:
                raise ValueError('Source + response budget exceeds fixed context; no silent truncation')
            inputs = torch.tensor([ids], device=device)
            with torch.no_grad():
                output = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    num_beams=1, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=model.generation_config.eos_token_id, use_cache=True)
            text = tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)
            try:
                decoded = json.loads(text)
                if not isinstance(decoded, dict):
                    raise ValueError('Expected JSON object')
            except (ValueError, TypeError):
                decoded = dict(answer=text, citations=[])
            predictions.append(dict(case_id=case['case_id'], raw=text, answer=decoded.get('answer', ''), citations=decoded.get('citations', [])))
        with (destination / 'predictions.jsonl').open('x') as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction) + '\n')
        provenance['complete'] = len(predictions) == len(cases)
        if provenance['complete']:
            provenance['diagnostics'] = score(cases, predictions)
    else:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
        model.config.use_cache = False
        examples = [response_example(tokenizer, r, r['response'], args.max_length) for r in cases]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0002)
        scaler = torch.cuda.amp.GradScaler(enabled=args.precision == 'fp16')
        losses = []
        skipped = 0
        attempted = 0
        model.train()
        for update in range(args.max_updates):
            if time.monotonic() - started >= args.walltime_seconds:
                break
            attempted += 1
            batch = collate([examples[update % len(examples)]], tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('SFT loss nonfinite')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError('SFT gradients nonfinite')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                losses.append(float(loss.detach()))
            else:
                skipped += 1
        model.save_pretrained(str(destination / 'adapter'))
        tokenizer.save_pretrained(str(destination / 'tokenizer'))
        diagnostic_sampler = torch.Generator().manual_seed(args.seed + 1)
        torch.save(dict(kind='diagnostic_training_state_not_validated_resume_checkpoint',
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), completed_updates=len(losses),
            rng=rng_state(diagnostic_sampler), round_robin_next_index=attempted % len(examples),
            attempted_microsteps=attempted, code_identity=source_identity()), destination / 'pilot_diagnostic_state.pt')
        provenance.update(completed_updates=len(losses), skipped_overflow_attempts=skipped, losses=losses, resumable=False,
                          limitation='Bounded adaptation pilot; full adapter/scaler/sampler restart must be validated before a larger SFT run')
    write_json(destination / 'report.json', provenance)


if __name__ == '__main__':
    main()
