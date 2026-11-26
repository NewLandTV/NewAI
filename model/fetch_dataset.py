"""Download the pinned UA-N1-Queen shards without overwriting local data."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


REVISION = '5a653644a2452583a16a29d766e01e1d20a5d180'
BASE = f'https://raw.githubusercontent.com/NewLandTV/UA-N1-Queen/{REVISION}/dataset'
SHA256 = (
    '0f5e3e1504ee534326d6fc8c04d4f4003e8b038896bb7179702ee576c68abe3c',
    '7921290369f3e62b8b86cceae1346f09eadccf9b5052145907b1dff7a9dfec6b',
    '9c83f37d94337147838fbc3f5605112bebce9f4df81d5b0691a5adb2f445ce63',
    'a0cface11f6f8a74c21beec2c7d9254838e42fb7e5d72347b487b1d7e25c12cc',
    '5f1cb3f2b1b5f7edd9ebcc834dd9ad47a2372e1bf3f2eccf6017e7033d043a15',
    '1e7b0403196984c2442651fe66d04968d4d6f33a57e168aceed7573776f7b904',
    '49a836c2c9f88dc8a460fd3567c8a850b6fb9c41b550612ca6a77701002c41de',
    '6e1b4516a0d46ee9d59d18740cac6732a32a4bcf40683f3a8e2dad548a42f83a',
)


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for index, expected in enumerate(SHA256):
        file = directory / f'train_part_{index}.jsonl'
        if file.exists():
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{file}이 원본과 달라요. 덮어쓰지 않았으니 다른 --out 폴더를 지정해 주세요.')
        else:
            pending.append((file, expected))
    for file, expected in pending:
        with urlopen(f'{BASE}/{file.name}', timeout=20) as response:
            data = response.read(1_000_000)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'{file.name} 다운로드의 무결성 검사에 실패했어요.')
        with file.open('xb') as target:
            target.write(data)
    print(f'{directory}에서 원본 학습 파일 {len(SHA256)}개의 무결성을 확인했어요.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('dataset'))
    try:
        download(parser.parse_args().out)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
