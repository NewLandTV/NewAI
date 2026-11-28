"""Train only two rank-4 matrices on assistant tokens; keep the Qwen2.5 head and decoder frozen."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
import unicodedata

from ua_runtime import QUANTIZATION, ROOT, UA, UAEngine, clean


def load_file(name):
    return [json.loads(line) for line in name.read_text(encoding='utf-8').splitlines() if line.strip()]


def key(value):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', value).casefold())


def check_splits(*splits):
    seen = set()
    for split in splits:
        for item in split:
            value = key(item['user'])
            if not value or value in seen:
                raise ValueError('분할 내부나 분할 사이에 중복 질문이 있어요.')
            seen.add(value)


def encode(model, data):
    torch = model.torch
    result = []
    with torch.no_grad():
        for row in data:
            prefix = model.prepare(row['user'])
            target = model.tokenizer.encode(clean(row['assistant']), add_special_tokens=False) + [model.eos]
            ids = torch.tensor([prefix + target[:-1]])
            hidden = model.hidden(ids).last_hidden_state[0, len(prefix) - 1:, :].contiguous()
            full = model.head(hidden)
            top, indices = full.topk(128, dim=-1)
            labels = torch.tensor(target).unsqueeze(-1)
            gold = full.gather(-1, labels)
            # Contrast the supervised answer token with the frozen head's top candidates.
            # Mask the duplicate if the answer token was already in the top 128.
            top = top.masked_fill(indices == labels, -1e9)
            result.append((hidden, torch.cat([gold, top], -1), torch.cat([labels, indices], -1)))
    return [torch.cat(parts, dim=0) for parts in zip(*result)]


def train(base, out, epochs, threads, quantization):
    import torch
    from safetensors.torch import save_file

    if out.exists():
        raise ValueError('결과 폴더가 이미 있어요. 새 폴더를 지정하세요.')
    here = Path(__file__).parent
    files = [here / name for name in ('ua_training.jsonl', 'ua_validation.jsonl', 'ua_eval.json')]
    training, validation = map(load_file, files[:2])
    evaluation = json.loads(files[2].read_text(encoding='utf-8'))
    check_splits(training, validation, evaluation)
    torch.manual_seed(19)
    start = perf_counter()
    model = UAEngine(base, threads=threads, quantization=quantization)
    train_data, valid_data = (encode(model, split) for split in (training, validation))
    print(json.dumps({'train_turns': len(training), 'validation_turns': len(validation),
                      'train_tokens': len(train_data[0]), 'feature_seconds': round(perf_counter() - start, 2)}), flush=True)

    width, vocab = model.shape
    del model
    a = torch.nn.Parameter(torch.randn(4, width) * .014)
    b = torch.nn.Parameter(torch.zeros(vocab, 4))
    optimizer = torch.optim.AdamW([a, b], lr=.005)

    def loss(rows, subset):
        hidden, frozen, indices = (value[subset] for value in rows)
        delta = 2 * torch.einsum('br,bcr->bc', torch.nn.functional.linear(hidden, a), b[indices])
        scores = frozen + delta
        nll = -torch.log_softmax(scores, dim=-1)[:, 0].mean()
        return nll, delta.square().mean()

    def validate():
        with torch.no_grad():
            return float(loss(valid_data, slice(None))[0])

    baseline = validate()
    best = (baseline, a.detach().clone(), b.detach().clone(), 0)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_data[0]))
        for batch in order.split(48):
            optimizer.zero_grad(set_to_none=True)
            nll, penalty = loss(train_data, batch)
            (nll + .018 * penalty).backward()
            torch.nn.utils.clip_grad_norm_([a, b], max_norm=3.)
            optimizer.step()
        nll = validate()
        print(json.dumps({'epoch': epoch, 'validation_candidate_nll': round(nll, 6)}), flush=True)
        if nll < best[0]:
            best = (nll, a.detach().clone(), b.detach().clone(), epoch)
    score, a_best, b_best, epoch = best
    if epoch == 0:
        raise RuntimeError('검증 후보 손실이 개선되지 않아 어댑터를 저장하지 않았어요.')
    out.mkdir(parents=True, exist_ok=False)
    save_file({'A': a_best.contiguous(), 'B': b_best.contiguous()}, out / 'adapter.safetensors')
    result = {'rank': 4, 'alpha': 8, 'trainable': a.numel() + b.numel(), 'seed': 19, 'epochs': epochs, 'best_epoch': epoch,
              'quantization': quantization,
              'train_turns': len(training), 'validation_turns': len(validation), 'test_turns': len(evaluation),
              'train_tokens': len(train_data[0]), 'validation_tokens': len(valid_data[0]),
              'baseline_candidate_nll': baseline, 'best_candidate_nll': score, 'seconds': round(perf_counter() - start, 2),
              'sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in files}, 'profile_sha256': sha256(UA.encode()).hexdigest()}
    (out / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'qwen')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--quantization', choices=QUANTIZATION, default=QUANTIZATION[0])
    args = parser.parse_args()
    if not 1 <= args.threads <= 16 or not 1 <= args.epochs <= 100:
        parser.error('threads: 1~16, epochs: 1~100')
    train(args.base, args.out, args.epochs, args.threads, args.quantization)
