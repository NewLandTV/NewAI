"""Write a tiny synthetic corpus to exercise the pretraining pipeline, not model quality."""

import argparse
import json
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    with args.out.open('x', encoding='utf-8') as file:
        for index in range(160):
            text = (f'연습용 이야기 {index}편입니다. 비가 그친 뒤 도서관에 가서 책을 읽고, '
                    '집으로 돌아와 따뜻한 차를 마셨어요.')
            file.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
