"""Operator-gated broadcast relay. It does not import or dispatch Ai-project commands."""

import argparse
import os
import re
import subprocess
from time import monotonic, sleep
import unicodedata

from newai_ua_bridge import NewAIUABridge


def viewer_question(message):
    if not isinstance(message, str) or not message.startswith('!ua '):
        return ''
    text = message[4:].strip()
    return text[:300] if text and not text.startswith(('/', '!')) else ''


def present(bridge, obs, question, approve, *, voice=False):
    raw = ''.join(bridge.get_response(question))
    # Model output can echo chat. Never let terminal or bidi controls hide the draft or its prompt.
    answer = ' '.join(''.join(char for char in raw if not unicodedata.category(char).startswith('C')
                              or char in '\n\r\t').split())
    if not answer or not approve(answer):
        return False
    if obs is not None:
        obs.update_obs_text('Answer Text', answer)
    if voice:
        bridge.read_aloud(answer)
    return True


def preview(answer, *, public):
    print(f'유아 답변 초안: {answer}', flush=True)
    return not public or input('이 답만 자막/음성으로 내보내려면 ‘공개’를 입력하세요: ').strip() == '공개'


def handle_turn(bridge, obs, question, *, voice, public):
    try:
        present(bridge, obs, question, lambda answer: preview(answer, public=public), voice=voice)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'이번 답을 처리하지 못했어요: {error}', flush=True)


def follow(chat, bridge, obs, voice):
    next_turn = 0.
    while chat.is_alive():
        question = ''
        for item in chat.get().sync_items():
            candidate = viewer_question(item.message)
            if candidate:
                question = candidate
        if question and monotonic() >= next_turn:
            handle_turn(bridge, obs, question, voice=voice, public=True)
            next_turn = monotonic() + 8
        sleep(1)


def rehearsal(bridge, obs=None, voice=False):
    print('유아 리허설. /exit 종료 · /reset 문맥 지우기', flush=True)
    while True:
        text = input('질문: ').strip()
        if text == '/exit':
            break
        if text == '/reset':
            bridge.reset()
        elif text:
            handle_turn(bridge, obs, text, voice=voice, public=obs is not None or voice)


def obs_connected(obs):
    return bool(getattr(getattr(getattr(obs, 'ws', None), 'ws', None), 'connected', False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-id', help='YouTube video ID, optional (operator approval required).')
    parser.add_argument('--obs', action='store_true', help='Update Ai-project OBS source Answer Text.')
    parser.add_argument('--voice', action='store_true', help='Play local synthetic voice via ffplay.')
    args = parser.parse_args()
    if args.video_id and not re.fullmatch(r'[A-Za-z0-9_-]{11}', args.video_id):
        parser.error('YouTube 영상 ID 11글자를 입력하세요.')
    obs = chat = None
    try:
        if args.obs:
            from dotenv import load_dotenv
            from obs import OBS

            load_dotenv()
            obs = OBS(os.getenv('obs_host') or '127.0.0.1', int(os.getenv('obs_port') or 4455), os.getenv('obs_password'))
            obs.connect_obs_websocket()
            if not obs_connected(obs):
                raise RuntimeError('OBS가 연결되지 않았어요. 방송 툴의 WebSocket 설정을 확인하세요.')
        with NewAIUABridge(rehearsal=not args.video_id, voice=args.voice) as bridge:
            if args.video_id:
                import pytchat

                chat = pytchat.create(video_id=args.video_id)
                follow(chat, bridge, obs, args.voice)
            else:
                rehearsal(bridge, obs, args.voice)
    except (KeyboardInterrupt, EOFError):
        pass
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    finally:
        if chat is not None:
            chat.terminate()
        if obs_connected(obs):
            obs.disconnect_obs_websocket()


if __name__ == '__main__':
    main()
