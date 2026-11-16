"""Cost worksheet: never substitute a tiny CPU measurement for candidate GPU rate."""
import argparse
import json
import math

from .common import read_json, write_json


def estimate(report, tokens, evaluation_interval, save_interval):
    if report['runtime']['device'] == 'cpu':
        raise ValueError('CPU fixture throughput is not a V100 or candidate GPU estimate')
    if report['training_tokens_per_second'] <= 0:
        raise ValueError('No successful-update throughput; resolve numerical stability first')
    updates = math.ceil(tokens / report['tokens_per_update'])
    train = tokens / report['training_tokens_per_second']
    evaluations = math.ceil(updates / evaluation_interval)
    saves = math.ceil(updates / save_interval) + 1
    overhead = evaluations * report['evaluation_seconds'] + saves * report['save_seconds']
    return dict(tokens=tokens, optimizer_updates=updates, training_hours=train / 3600,
                measured_periodic_overhead_hours=overhead / 3600, subtotal_hours=(train + overhead) / 3600,
                queue_time_unknown=True, restart_and_startup_allowance='Add from the actual planned number of jobs; restore timing is in the benchmark report',
                approximate_dense_flops=6 * report['parameters'] * tokens)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True)
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--eval-interval', type=int, default=100)
    p.add_argument('--save-interval', type=int, default=100)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    if min(args.tokens, args.eval_interval, args.save_interval) <= 0:
        p.error('Positive counts required')
    result = estimate(read_json(args.benchmark), args.tokens, args.eval_interval, args.save_interval)
    write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
