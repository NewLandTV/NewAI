"""Score UA prompts with the same Qwen2.5 prompt, with and without its learned head."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import subprocess
from time import perf_counter

from ua_runtime import QUANTIZATION, ROOT, UAEngine


def score(case, answer):
    return bool(answer.strip() and len(answer) <= case['max_chars'] and
                all(re.search(expr, answer, flags=re.I) for expr in case['all']) and
                not any(re.search(expr, answer, flags=re.I) for expr in case.get('none', [])))


def measure(case, ask):
    start = perf_counter()
    answer = ask(case)
    return {'id': case['id'], 'group': case['group'], 'text': answer, 'pass': score(case, answer),
            'self_reference': bool(re.search(r'\b유아(?:는|가|도|라|와|랑|의)', answer)),
            'ms': round((perf_counter() - start) * 1000)}


def summarize(rows):
    counts = Counter(row['group'] for row in rows)
    return {'passed': sum(row['pass'] for row in rows), 'total': len(rows),
            'groups': {k: [sum(row['pass'] for row in rows if row['group'] == k), n] for k, n in counts.items()},
            'self_reference': sum(row['self_reference'] for row in rows),
            'median_ms': statistics.median(row['ms'] for row in rows)}


def paired(cases, base, adapter, quantization):
    model = UAEngine(base, adapter, quantization=quantization)
    groups = {'prompt_only': [], 'lora': []}
    for case in cases:
        for name in groups:
            groups[name].append(measure(case, lambda row: model.answer(
                row['user'], row.get('history', ()), tuned=name == 'lora', use_guards=False)))
    return groups


def gguf(cases, persona=False):
    script = Path(__file__).with_name('gguf-context.mjs')
    command = 'serve-ua' if persona else 'serve'
    with subprocess.Popen(['node', str(script), command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding='utf-8') as proc:
        try:
            if json.loads(proc.stdout.readline()).get('type') != 'ready':
                raise RuntimeError('이전 Qwen3 모드를 열지 못했어요.')

            def ask(case):
                proc.stdin.write(json.dumps({'prompt': case['user'], 'history': case.get('history', []),
                                             'seed': 19}) + '\n')
                proc.stdin.flush()
                return json.loads(proc.stdout.readline()).get('text', '')

            return {'ua_qwen3' if persona else 'prior_context': [measure(case, ask) for case in cases]}
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    default_cases = Path(__file__).with_name('ua_eval.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=default_cases)
    parser.add_argument('--base', type=Path, default=ROOT / 'qwen')
    parser.add_argument('--quantization', choices=QUANTIZATION, default=QUANTIZATION[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--adapter', type=Path)
    mode.add_argument('--prior', action='store_true', help='Use the prior PR #2 Qwen3 context mode.')
    mode.add_argument('--ua-qwen3', action='store_true', help='Use the same Qwen3 weights with the UA profile.')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    data = paired(cases, args.base, args.adapter, args.quantization) if args.adapter else gguf(cases, persona=args.ua_qwen3)
    settings = {'quantization': args.quantization} if args.adapter else {'seed': 19}
    print(json.dumps({'summary': {k: summarize(v) for k, v in data.items()}, 'settings': settings, 'answers': data},
                     ensure_ascii=False, indent=2))
