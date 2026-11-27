"""Fit offline Korean intent retrieval, select on validation, and serve reviewed replies."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from prepare_daily import key, normalize


VERSION = 1
DIALOGUES = Path(__file__).with_name('daily_dialogues.json')
MODES = ('char', 'char_word', 'char_cues', 'char_word_cues')
FALLBACK = 'fallback'
# Short phrases derived from the reviewed training examples. They help keep
# Korean endings or generic wording from outweighing topic-bearing words.
CUES = {
    'greeting': ('안녕하세요', '반가워', '처음 뵙', '처음 왔', '좋은 아침'),
    'thanks': ('고마워', '감사', '덕분'),
    'farewell': ('자러 갈', '잘 자', '내일 봐', '잘 가', '다음에 봐'),
    'celebrate': ('기분 좋', '기분이 좋', '홀가분', '붙었', '해냈', '기뻐'),
    'tired': ('지쳤', '피곤', '힘들었', '기운이', '지쳐'),
    'nervous': ('긴장', '면접', '떨려'),
    'indoor': ('집에서', '실내에서'),
    'food': ('점심', '저녁 메뉴', '저녁으로', '뭐 먹', '뭘 먹', '밥 뭐'),
    'outing': ('주말', '갈 만한 곳', '가 볼 곳', '밖에 나가', '어디 갈'),
    'identity': ('자기소개', '챗봇', '이름이 뭐', '뭐 할 수 있'),
}


def features(prompt, mode):
    text = key(prompt)
    result = Counter('c:' + text[i:i + size] for size in (2, 3, 4) for i in range(len(text) - size + 1))
    if 'word' in mode:
        result.update('w:' + word for word in re.findall(r'[^\W_]+', normalize(prompt).casefold()) if len(word) >= 2)
    return result


def vector(prompt, mode, idf):
    weights = {term: (1 + math.log(count)) * idf[term]
               for term, count in features(prompt, mode).items() if term in idf}
    length = math.sqrt(sum(value * value for value in weights.values()))
    return {term: value / length for term, value in weights.items()} if length else {}


def split(text):
    digest = hashlib.sha256(f'42:{key(text)}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % 100


def score_intents(prompt, model):
    query = vector(prompt, model['mode'], model['idf'])
    intents = {}
    for label, prototype in model['prototypes']:
        score = sum(weight * prototype.get(term, 0) for term, weight in query.items())
        intents[label] = max(intents.get(label, 0), score)
    if 'cues' in model:
        compact = key(prompt)
        for label, cues in model['cues'].items():
            matches = sum(cue in compact for cue in cues)
            if matches:
                intents[label] += .25 + .055 * min(matches - 1, 2)
    best, second = sorted(intents.items(), key=lambda pair: (-pair[1], pair[0]))[:2]
    return best[0], best[1], best[1] - second[1]


def select(scored, threshold, margin):
    label, confidence, gap = scored
    return label if confidence >= threshold and gap >= margin else FALLBACK


def measure(rows, threshold, margin):
    total = matched = in_scope = answered = out_scope = rejected = 0
    for expected, scores in rows:
        actual = select(scores, threshold, margin)
        total += 1
        matched += actual == expected
        if expected == FALLBACK:
            out_scope += 1
            rejected += actual == FALLBACK
        else:
            in_scope += 1
            answered += actual == expected
    return dict(total=total, correct=matched, accuracy=matched / total, in_scope=in_scope,
                in_scope_correct=answered, out_scope=out_scope, out_scope_rejected=rejected)


def train(data, output):
    if output.exists() or output.with_name('report.json').exists():
        raise ValueError('기존 모델을 덮어쓰지 않아요. 다른 출력 폴더를 지정해 주세요.')
    dialogues = json.loads(DIALOGUES.read_text(encoding='utf-8'))
    curated = {name: [(label, prompt) for label, entry in dialogues.items()
                      for prompt in (entry['validation'] + entry['test'] if name == 'validation' else entry[name])]
               for name in ('train', 'validation')}
    release = json.loads(DIALOGUES.with_name('daily_release_test.json').read_text(encoding='utf-8'))
    curated['test'] = [(label, prompt) for label, prompts in release.items() for prompt in prompts]
    canon = {name: {key(prompt) for _, prompt in items} for name, items in curated.items()}
    if any(canon[a] & canon[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('검수한 질문이 서로 다른 분할에서 겹쳐요.')
    holdout = canon['validation'] | canon['test']
    corpus = []
    for index in range(8):
        with (data / f'train_part_{index}.jsonl').open(encoding='utf-8') as source:
            for row in map(json.loads, source):
                if split(row['user']) < 80 and key(row['user']) not in holdout:
                    corpus.append(row['user'])
    if not corpus:
        raise ValueError('학습용 공개 대화가 없어요.')
    candidates = []
    for mode in MODES:
        df = Counter(term for prompt in corpus for term in features(prompt, mode))
        # Only features used by the reviewed prototypes are needed for a compact, non-verbatim artifact.
        vocabulary = {term for _, prompt in curated['train'] for term in features(prompt, mode)}
        idf = {term: min(4.0, 1 + math.log((len(corpus) + 1) / (df[term] + 1)))
               for term in sorted(vocabulary)}
        model = dict(version=VERSION, mode=mode, idf=idf,
                     prototypes=[(label, vector(prompt, mode, idf)) for label, prompt in curated['train']],
                     responses={label: entry['response'] for label, entry in dialogues.items()})
        if 'cues' in mode:
            model['cues'] = {label: [key(cue) for cue in cues] for label, cues in CUES.items()}
        validation = [(label, score_intents(prompt, model)) for label, prompt in curated['validation']]
        results = []
        for threshold in (.08, .12, .16, .20, .24, .28, .32, .36):
            for margin in (.02, .05, .08, .12):
                metrics = measure(validation, threshold, margin)
                # Prefer total validation correctness, then safe rejection, then conservative settings.
                results.append(((metrics['correct'], metrics['out_scope_rejected'], threshold, margin), metrics))
        ranking, metrics = max(results)
        model.update(threshold=ranking[2], margin=ranking[3])
        candidates.append((ranking, metrics, model))
    _, _, best = max(candidates, key=lambda entry: entry[0])
    tested = [(label, score_intents(prompt, best)) for label, prompt in curated['test']]
    report = dict(corpus_train_questions=len(corpus), curated_train=len(curated['train']),
                  validation_candidates=[dict(mode=model['mode'], threshold=model['threshold'], margin=model['margin'], **metrics)
                                         for _, metrics, model in candidates],
                  selected=best['mode'], test=measure(tested, best['threshold'], best['margin']),
                  test_cases=[dict(prompt=prompt, expected=label, actual=select(scores, best['threshold'], best['margin']))
                              for (label, prompt), (_, scores) in zip(curated['test'], tested)])
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in ((output, best), (output.with_name('report.json'), report)):
        with path.open('x', encoding='utf-8') as target:
            target.write(json.dumps(contents, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


def load_model(path):
    model = json.loads(path.read_text(encoding='utf-8'))
    if model.get('version') != VERSION or model.get('mode') not in MODES:
        raise ValueError('지원하지 않는 일상 대화 모델이에요.')
    return model


def answer(prompt, model):
    label = select(score_intents(prompt, model), model['threshold'], model['margin'])
    return model['responses'][label]


def show(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def serve(model):
    show({'type': 'ready'})
    for line in sys.stdin:
        try:
            prompt = json.loads(line)['prompt']
            if not isinstance(prompt, str) or not normalize(prompt) or len(prompt) > 4000:
                raise ValueError('지원하지 않는 질문 길이예요.')
            show({'type': 'answer', 'text': answer(prompt, model)})
        except (ValueError, TypeError, KeyError):
            show({'type': 'answer', 'error': 'invalid_prompt'})


if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--data', type=Path, default=Path('dataset/daily'))
    fit.add_argument('--out', type=Path, default=Path('checkpoints/daily-final/model.json'))
    for name in ('ask', 'serve'):
        command = commands.add_parser(name)
        command.add_argument('--checkpoint', type=Path, default=Path('checkpoints/daily-final/model.json'))
        if name == 'ask':
            command.add_argument('prompt')
    args = parser.parse_args()
    try:
        if args.command == 'train':
            train(args.data, args.out)
        elif args.command == 'ask':
            print(answer(args.prompt, load_model(args.checkpoint)))
        else:
            serve(load_model(args.checkpoint))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
