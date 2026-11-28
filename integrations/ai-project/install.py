"""Copy the optional UA integration to a local Ai-project checkout without modifying existing code."""

import argparse
from pathlib import Path


def install(target):
    src = target.resolve() / 'src'
    if not ((src / 'chatbot.py').is_file() and (src / 'commands/__init__.py').is_file()):
        raise ValueError('Ai-project 리포지토리의 최상위 폴더를 지정하세요.')
    directory = Path(__file__).resolve().parent
    copies = [('newai_ua_bridge.py', src / 'newai_ua_bridge.py'), ('ua_broadcast.py', src / 'ua_broadcast.py'),
              ('command.py', src / 'commands/newai_ua/__init__.py')]
    files = [(destination, (directory / name).read_bytes()) for name, destination in copies]
    for path, data in files:
        if not path.resolve().is_relative_to(target.resolve()) or path.is_symlink():
            raise ValueError(f'연결된 경로에는 설치하지 않았어요: {path}')
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f'내용이 다른 기존 파일을 덮어쓰지 않았어요: {path}')
    for path, data in files:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as file:
            file.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ai_project', type=Path)
    args = parser.parse_args()
    try:
        install(args.ai_project)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    print('Ai-project에 유아 리허설 명령과 방송 릴레이를 추가했어요. 기존 파일은 그대로예요.')
