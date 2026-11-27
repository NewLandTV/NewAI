"""Generate side-by-side replies from two local checkpoints on unseen prompts."""

import argparse
from pathlib import Path

import torch

from newai import answer, load_data, restore, show
from prepare_daily import key


PROBES = (
    ('인사', '안녕하세요, 오늘 처음 왔어요.'),
    ('인사', '좋은 아침이에요, 잘 잤어요?'),
    ('감사', '도와줘서 고마워요, 덕분이에요.'),
    ('작별', '이제 자러 갈게요. 내일 봐요!'),
    ('맞장구', '오늘 산책했더니 기분이 좋아요.'),
    ('맞장구', '해야 할 일을 다 끝내서 홀가분해요.'),
    ('공감', '오늘 일이 많아서 많이 지쳤어요.'),
    ('공감', '발표가 끝났는데도 긴장이 안 풀려요.'),
    ('일상 질문', '비 오는 날 집에서 뭘 하면 좋을까요?'),
    ('일상 질문', '점심 메뉴로 먹기 편한 거 추천해 줘요.'),
    ('일상 질문', '주말에 시간이 비는데 어디 가 볼까요?'),
    ('이어 묻기 형태', '그럼 산책 말고 집에서 할 건 없나요?'),
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('checkpoints/ua-n1-5a65364/best.pt'))
    parser.add_argument('--baseline-data', type=Path, default=Path('dataset'))
    parser.add_argument('--candidate', type=Path, default=Path('checkpoints/daily-v2/best.pt'))
    parser.add_argument('--candidate-data', type=Path, default=Path('dataset/daily'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    models = []
    seen = set()
    for checkpoint, directory in ((args.baseline, args.baseline_data), (args.candidate, args.candidate_data)):
        model, tokenizer, metadata = restore(checkpoint)
        groups, fingerprint = load_data(directory, **metadata['data_config'])
        if fingerprint != metadata['data_fingerprint']:
            parser.error(f'{directory}의 데이터가 체크포인트를 만든 데이터와 달라요.')
        seen.update(key(prompt) for name in ('train', 'validation') for prompt, _ in groups[name])
        models.append((model, tokenizer))
    for category, prompt in PROBES:
        if key(prompt) in seen:
            show({'category': category, 'prompt': prompt, 'skipped': 'exact_overlap'})
            continue
        baseline, candidate = (answer(model, tokenizer, prompt) for model, tokenizer in models)
        show({'category': category, 'prompt': prompt, 'baseline': baseline, 'candidate': candidate})
