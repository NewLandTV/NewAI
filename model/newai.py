"""Train, evaluate and run the NewAI prototype entirely on the local machine."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F


PAD, BOS, SEP, EOS, UNK = range(5)
IGNORE = -100
VERSION = 1


def clean(text):
    return unicodedata.normalize('NFC', text).strip() if isinstance(text, str) else ''


def dialogue(row):
    if not isinstance(row, dict):
        return
    for question, answer in [('user', 'assistant'), ('question', 'answer'),
                             ('prompt', 'response'), ('instruction', 'output')]:
        if clean(row.get(question)) and clean(row.get(answer)):
            prompt = clean(row[question])
            if question == 'instruction' and clean(row.get('input')):
                prompt += '\n' + clean(row['input'])
            yield prompt, clean(row[answer])
            return
    turns = row.get('messages')
    if isinstance(turns, list):
        prompt = ''
        for turn in turns:
            role = turn.get('role') if isinstance(turn, dict) else None
            if role == 'user':
                prompt = clean(turn.get('content'))
            else:
                answer = clean(turn.get('content')) if role == 'assistant' else ''
                if prompt and answer:
                    yield prompt, answer
                prompt = ''


def load_data(directory, limit, seed):
    paths = [Path(directory) / f'train_part_{i}.jsonl' for i in range(8)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError('학습 파일이 없어요: ' + ', '.join(missing))
    groups = {'train': [], 'validation': [], 'test': []}
    fingerprint = hashlib.sha256()
    seen = set()
    skipped = 0
    for path in paths:
        with path.open(encoding='utf-8-sig') as source:
            for number, line in enumerate(source, 1):
                if limit and number > limit:
                    break
                if not line.strip():
                    continue
                try:
                    pairs = list(dialogue(json.loads(line)))
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path.name}의 {number}번째 줄이 올바른 JSON이 아니에요.') from error
                if not pairs:
                    skipped += 1
                for prompt, answer in pairs:
                    pair = (prompt, answer)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    key = ' '.join(prompt.casefold().split())
                    bucket = int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8], 'big') % 100
                    group = 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'
                    groups[group].append(pair)
                    fingerprint.update((json.dumps((group, prompt, answer), ensure_ascii=False) + '\n').encode())
    if skipped:
        print(f'지원하는 대화 형식과 맞지 않는 행 {skipped}개를 건너뛰었어요.', file=sys.stderr)
    if any(not pairs for pairs in groups.values()):
        raise ValueError(f'학습·검증·테스트에 각각 대화가 필요해요. 현재: { {k: len(v) for k, v in groups.items()} }')
    return groups, fingerprint.hexdigest()


class Characters:
    def __init__(self, characters):
        self.characters = characters
        self.ids = {char: index for index, char in enumerate(characters, 5)}

    @classmethod
    def learn(cls, pairs, maximum):
        counts = Counter()
        for prompt, answer in pairs:
            counts.update(prompt)
            counts.update(answer)
        return cls(''.join(char for char, _ in counts.most_common(maximum - 5)))

    def encode(self, text):
        return [self.ids.get(char, UNK) for char in text]

    def decode(self, ids):
        return ''.join(self.characters[index - 5] for index in ids if index >= 5).strip()


class NewAI(nn.Module):
    def __init__(self, vocab, context, width, heads, layers):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, width * 3, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab, bias=False))
        self.register_buffer('causal', torch.ones((context, context), dtype=torch.bool).triu(1), persistent=False)

    def forward(self, ids):
        length = ids.size(1)
        states = self.tokens(ids) + self.positions(torch.arange(length))
        for block in self.blocks:
            states = block(states, src_mask=self.causal[:length, :length], is_causal=True)
        return self.output(states)


def prefix(prompt, tokenizer, context):
    return [BOS, *tokenizer.encode(clean(prompt)[-(context // 3):]), SEP]


def batches(pairs, tokenizer, context, size, order):
    for start in range(0, len(order), size):
        inputs, labels = [], []
        for index in order[start:start + size]:
            prompt, answer = pairs[index]
            start_ids = prefix(prompt, tokenizer, context)
            available = context - len(start_ids) + 1
            encoded_answer = tokenizer.encode(answer[:available])
            # Don't teach the model that a truncated answer ended naturally.
            if len(encoded_answer) < available:
                encoded_answer.append(EOS)
            sequence = start_ids + encoded_answer
            inputs.append(sequence[:-1])
            labels.append([IGNORE] * (len(start_ids) - 1) + encoded_answer)
        width = max(map(len, inputs))
        yield (torch.tensor([row + [PAD] * (width - len(row)) for row in inputs]),
               torch.tensor([row + [IGNORE] * (width - len(row)) for row in labels]))


def run_epoch(model, pairs, tokenizer, size, optimizer=None, shuffle_seed=0):
    model.train(optimizer is not None)
    order = (torch.randperm(len(pairs), generator=torch.Generator().manual_seed(shuffle_seed)).tolist()
             if optimizer else range(len(pairs)))
    total_loss = token_count = known_tokens = correct = 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in batches(pairs, tokenizer, model.context, size, order):
            scores = model(inputs)
            summed = F.cross_entropy(scores.flatten(0, 1), labels.flatten(), ignore_index=IGNORE, reduction='sum')
            count = int(labels.ne(IGNORE).sum())
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                (summed / count).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            comparable = labels.ne(IGNORE) & labels.ne(UNK) & labels.ne(EOS)
            known_tokens += int(comparable.sum())
            correct += int((scores.detach().argmax(-1).eq(labels) & comparable).sum())
            total_loss += summed.item()
            token_count += count
    mean = total_loss / token_count
    return {'loss': mean, 'perplexity': math.exp(min(mean, 700)), 'target_tokens': token_count,
            'known_char_accuracy': correct / known_tokens if known_tokens else None}


def restore(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('version') != VERSION:
        raise ValueError('지원하는 형식의 NewAI 체크포인트가 아니에요.')
    tokenizer = Characters(checkpoint['characters'])
    model = NewAI(len(tokenizer.ids) + 5, **checkpoint['architecture'])
    model.load_state_dict(checkpoint['weights'])
    return model.eval(), tokenizer, checkpoint


def save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def show(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def train(args):
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output / 'last.pt', output / 'best.pt'
    if args.resume and (Path(args.resume).resolve() != last_path.resolve() or not best_path.is_file()):
        raise ValueError('같은 출력 폴더에 last.pt와 best.pt를 두고 last.pt에서 이어서 학습해 주세요.')
    if not args.resume and (last_path.exists() or best_path.exists()):
        raise ValueError('기존 체크포인트를 보호하기 위해 --resume 또는 새 --out 폴더를 지정해 주세요.')
    checkpoint = None
    if args.resume:
        model, tokenizer, checkpoint = restore(args.resume)
    data_config = checkpoint['data_config'] if checkpoint else {'seed': args.seed, 'limit': args.limit}
    groups, fingerprint = load_data(args.data, **data_config)
    show({name: len(pairs) for name, pairs in groups.items()})
    if checkpoint:
        if checkpoint['data_fingerprint'] != fingerprint:
            raise ValueError('데이터가 이전 체크포인트와 달라서 학습을 이어갈 수 없어요.')
        architecture = checkpoint['architecture']
        training = checkpoint['training']
    else:
        torch.manual_seed(data_config['seed'])
        architecture = dict(context=args.context, width=args.width, heads=args.heads, layers=args.layers)
        training = dict(batch_size=args.batch_size, learning_rate=args.learning_rate)
        tokenizer = Characters.learn(groups['train'], args.vocab_size)
        model = NewAI(len(tokenizer.ids) + 5, **architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'])
    previous, best = 0, math.inf
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        previous, best = checkpoint['epoch'], checkpoint['best_loss']
    for epoch in range(previous + 1, previous + args.epochs + 1):
        train_stats = run_epoch(model, groups['train'], tokenizer, training['batch_size'],
                                optimizer, data_config['seed'] + epoch)
        validation = run_epoch(model, groups['validation'], tokenizer, training['batch_size'])
        improved = validation['loss'] < best
        best = min(best, validation['loss'])
        state = dict(version=VERSION, weights=model.state_dict(), characters=tokenizer.characters,
                     epoch=epoch, best_loss=best, architecture=architecture, data_config=data_config,
                     data_fingerprint=fingerprint, training=training, optimizer=optimizer.state_dict(),
                     rng=torch.get_rng_state())
        if improved:
            save(state, best_path)
        save(state, last_path)
        show(dict(epoch=epoch, train=train_stats, validation=validation, best_saved=improved))


def evaluate(args):
    model, tokenizer, checkpoint = restore(args.checkpoint)
    groups, fingerprint = load_data(args.data, **checkpoint['data_config'])
    if checkpoint['data_fingerprint'] != fingerprint:
        raise ValueError('학습 당시와 데이터가 달라 동일한 테스트 분할을 확인할 수 없어요.')
    results = run_epoch(model, groups['test'], tokenizer, checkpoint['training']['batch_size'])
    show(dict(split='test', examples=len(groups['test']), epoch=checkpoint['epoch'], **results))


@torch.inference_mode()
def answer(model, tokenizer, prompt):
    ids = prefix(prompt, tokenizer, model.context)
    output = []
    for _ in range(min(160, model.context + 1 - len(ids))):
        logits = model(torch.tensor([ids]))[0, -1]
        logits[[PAD, BOS, SEP, UNK]] = -torch.inf
        # Greedy decoding can repeat itself; forbid any exact, already generated 4-character span.
        if len(output) >= 4:
            repeat = {output[i + 3] for i in range(len(output) - 3) if output[i:i + 3] == output[-3:]}
            logits[list(repeat)] = -torch.inf
        token = int(logits.argmax())
        if token == EOS:
            break
        output.append(token)
        ids.append(token)
    return tokenizer.decode(output)


def serve(model, tokenizer):
    show({'type': 'ready'})
    for line in sys.stdin:
        try:
            prompt = clean(json.loads(line)['prompt'])
            if not prompt or len(prompt) > 4000:
                raise ValueError('지원하지 않는 질문 길이예요.')
            show({'type': 'answer', 'text': answer(model, tokenizer, prompt)})
        except (ValueError, KeyError, TypeError):
            show({'type': 'answer', 'error': 'invalid_prompt'})


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('1 이상의 정수를 입력해 주세요.')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', default='dataset')
    training.add_argument('--out', default='checkpoints')
    training.add_argument('--resume')
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--limit', type=int, default=2000, help='Maximum rows per file; 0 reads all rows.')
    training.add_argument('--learning-rate', type=float, default=0.001)
    for name, default in [('epochs', 5), ('context', 128), ('width', 96), ('heads', 4),
                          ('layers', 2), ('batch-size', 16), ('vocab-size', 4096)]:
        training.add_argument('--' + name, type=positive, default=default)
    for name in ('evaluate', 'ask', 'serve'):
        command = commands.add_parser(name)
        command.add_argument('--checkpoint', default='checkpoints/best.pt')
        if name == 'evaluate':
            command.add_argument('--data', default='dataset')
        if name == 'ask':
            command.add_argument('prompt')
    for command in commands.choices.values():
        command.add_argument('--threads', type=positive, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        if (args.context < 12 or args.vocab_size < 6 or args.width % args.heads or args.limit < 0
                or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
            parser.error('context ≥ 12, vocab-size ≥ 6, width는 heads의 배수, limit ≥ 0, learning-rate > 0이어야 해요.')
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        model, tokenizer, _ = restore(args.checkpoint)
        if args.command == 'ask':
            print(answer(model, tokenizer, args.prompt) or '답변을 만들지 못했어요.')
        else:
            serve(model, tokenizer)


if __name__ == '__main__':
    # Keep the stdin/stdout protocol UTF-8 even on Windows local installations.
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
