"""Download pinned NewAI runtime assets (no credentials; keep weights outside git)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen


ROOT = Path(os.environ.get('NEWAI_MODEL_ROOT', Path.home() / '.cache' / 'newai')).expanduser()
PACKS = {
    'context': ('unsloth/Qwen3-1.7B-GGUF', 'd7f544eead698dbd1f15126ef60b45a1e1933222', {
        'README.md': '', 'Qwen3-1.7B-Q4_K_M.gguf': 'b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897',
    }),
    'qwen': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775', {
        'LICENSE': '', 'config.json': '', 'generation_config.json': '', 'tokenizer.json': '',
        'tokenizer_config.json': '', 'merges.txt': '', 'vocab.json': '',
        'model.safetensors': 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
    }),
    'whisper': ('Systran/faster-whisper-base', 'ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66', {
        'README.md': '', 'config.json': '', 'tokenizer.json': '', 'vocabulary.txt': '',
        'model.bin': 'd01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9',
    }),
    'canora': ('empiriolabsai/canora-korean', 'ad11531a683672b83385582ac4321a30ebb4d182', {
        'LICENSE': '', 'NOTICE': '', 'README.md': '', 'config.json': '',
        'g2p_ko.py': '7a2136d3365693aa564785885d5373f63cc6e7005950199fce78dc0a2d1b9429',
        'onnx/model_quantized.onnx': '720d1f79f733193e75262e3e97d716876a9ce1afe9714db4d7116cfe6606430e',
        'voices/seoyeon.bin': '205bf19f91671760f651799c30e182fe35bff505465a31de884f823833c0cce1',
    }),
}
UPSTREAM_LICENSES = {
    'context': 'https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/90862c4b9d2787eaed51d12237eafdfe7c5f6077/LICENSE',
    'whisper': 'https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/LICENSE',
}


def checksum(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def fetch_one(destination, address, expected):
    if destination.is_file():
        if expected and checksum(destination) != expected:
            raise ValueError(f'{destination}의 체크섬이 달라요. 기존 파일을 옮긴 뒤 다시 받으세요.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            staged = Path(target.name)
        try:
            suffix = f'?download=true&newai={time.time_ns()}' if expected else '?download=true'
            request = Request(address + suffix, headers={'User-Agent': 'NewAI-runtime-fetch/1'})
            with urlopen(request, timeout=90) as response, staged.open('wb') as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            if expected and checksum(staged) != expected:
                raise ValueError(f'{destination.name}의 체크섬이 달라 설치하지 않았어요.')
            staged.replace(destination)
            return
        except OSError:
            if attempt == 2:
                raise
        finally:
            staged.unlink(missing_ok=True)


def fetch(root, names):
    jobs = []
    for name in names:
        repo, revision, files = PACKS[name]
        jobs.extend((root / name / file, f'https://huggingface.co/{repo}/resolve/{revision}/{file}', digest)
                    for file, digest in files.items())
        if name in UPSTREAM_LICENSES:
            jobs.append((root / name / 'LICENSE', UPSTREAM_LICENSES[name], ''))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for job in [pool.submit(fetch_one, *args) for args in jobs]:
            job.result()
    print(f'{root} 준비 완료: {", ".join(names)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack', choices=(*PACKS, 'all', 'voice'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        fetch(args.root, tuple(PACKS) if args.pack == 'all' else
              ('whisper', 'canora') if args.pack == 'voice' else (args.pack,))
    except (OSError, ValueError) as error:
        raise SystemExit(f'모델 다운로드 실패: {error}') from error
