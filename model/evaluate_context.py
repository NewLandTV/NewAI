"""Compare daily retrieval, baseline generation and optional adapter on the same prompts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from context_model import ContextModel, show
from daily_model import answer, load_model
from fetch_runtime import ROOT


def cases():
    return json.loads(Path(__file__).with_name('context_eval.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'qwen')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--daily', type=Path, default=Path('checkpoints/daily-final/model.json'))
    parser.add_argument('--daily-only', action='store_true', help='Evaluate the retrieval baseline without PyTorch.')
    args = parser.parse_args()
    daily = load_model(args.daily)
    generation = None if args.daily_only else ContextModel(args.base, args.adapter)
    for case in cases():
        start = perf_counter()
        retrieved = answer(case['prompt'], daily)
        result = dict(**case, daily=retrieved, daily_ms=round((perf_counter() - start) * 1000, 2))
        if generation is not None:
            start = perf_counter()
            result.update(generated=generation.generate(case['prompt'], case.get('history'), sample=False),
                          guarded=case['id'] == 'live', generation_ms=round((perf_counter() - start) * 1000))
        show(result)
