"""Create eight UA-format JSONL shards using pinned Korean daily chat data."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen


REVISION = '4cf20d13fc46f5037fd1c531cd566e2dd9f72974'
URL = f'https://raw.githubusercontent.com/songys/Chatbot_data/{REVISION}/ChatbotData.csv'
SHA256 = '287eb129695b577321c80ad397bb3c2279164d4ca577874d129fd3db5b30afe2'
PRIVATE_OR_LINK = re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?<!\d)0\d{1,2}[ -]?\d{3,4}[ -]?\d{4}(?!\d)', re.I)
HANGUL = re.compile('[가-힣]')
ANSWER_CAP = 12


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text or '').split())


def key(text):
    return ''.join(char for char in normalize(text).casefold() if char.isalnum())


def reject(prompt, answer):
    if not (2 <= len(prompt) <= 32 and 3 <= len(answer) <= 60):
        return 'empty_or_length'
    if not HANGUL.search(prompt) or not HANGUL.search(answer):
        return 'no_hangul'
    if PRIVATE_OR_LINK.search(prompt) or PRIVATE_OR_LINK.search(answer):
        return 'contact_or_link'
    if len(key(answer)) < 3 or key(prompt) == key(answer):
        return 'low_information'


def prepare(source, output):
    paths = [source / f'train_part_{i}.jsonl' for i in range(8)]
    if not all(path.is_file() for path in paths):
        raise ValueError('UA-N1-Queen 파일 8개가 먼저 필요해요. model/fetch_dataset.py를 실행해 주세요.')
    if output.exists():
        raise ValueError('기존 폴더를 덮어쓰지 않아요. 새 --out 경로를 지정해 주세요.')
    stats = {'ua': Counter(), 'songys': Counter()}
    kept = []
    seen = set()
    ua_questions = set()
    songys_questions = set()
    frequent = Counter()

    def append(prompt, answer, origin):
        reason = reject(prompt, answer)
        prompt_key, answer_key = key(prompt), key(answer)
        pair = (prompt_key, answer_key)
        if origin == 'ua':
            ua_questions.add(prompt_key)
        elif not reason and (prompt_key in ua_questions or prompt_key in songys_questions):
            reason = 'duplicate_question'
        if not reason and pair in seen:
            reason = 'duplicate_pair'
        if not reason and origin == 'songys' and frequent[answer_key] >= ANSWER_CAP:
            reason = 'repeated_answer'
        if reason:
            stats[origin][reason] += 1
            return
        seen.add(pair)
        if origin == 'songys':
            songys_questions.add(prompt_key)
            frequent[answer_key] += 1
        kept.append((prompt, answer))
        stats[origin]['kept'] += 1

    for path in paths:
        for line in path.read_text(encoding='utf-8-sig').split('\n'):
            if line.strip():
                row = json.loads(line)
                append(normalize(row['user']), normalize(row['assistant']), 'ua')
    with urlopen(URL, timeout=30) as response:
        raw = response.read(2_000_000)
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('일상 대화 원본 검사에 실패했어요.')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if not {'Q', 'A', 'label'}.issubset(reader.fieldnames or []):
        raise ValueError('일상 대화 CSV 형식이 맞지 않아요.')
    for row in reader:
        if row['label'] != '0':
            stats['songys']['not_daily'] += 1
        else:
            append(normalize(row['Q']), normalize(row['A']), 'songys')

    output.mkdir(parents=True, exist_ok=False)
    for index in range(8):
        with (output / f'train_part_{index}.jsonl').open('x', encoding='utf-8') as target:
            for prompt, answer in kept[index::8]:
                target.write(json.dumps({'user': prompt, 'assistant': answer}, ensure_ascii=False) + '\n')
    report = {'songys_revision': REVISION, 'total': len(kept), 'by_source': stats}
    (output / 'preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('dataset'))
    parser.add_argument('--out', type=Path, default=Path('dataset/daily'))
    args = parser.parse_args()
    try:
        prepare(args.source, args.out)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
