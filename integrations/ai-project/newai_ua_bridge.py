"""Drop-in Ai-project get_response() provider; all inference stays in a local child process."""

import base64
from collections import deque
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import shutil
import subprocess
import sys
from threading import Lock, Thread
from time import monotonic


MAX_WIRE = 12 * 1024 * 1024
WORKER_ENV = {'PATH', 'SYSTEMROOT', 'WINDIR', 'HOME', 'USERPROFILE', 'TMP', 'TEMP',
              'NEWAI_MODEL_ROOT', 'NEWAI_GGUF', 'NEWAI_THREADS', 'NEWAI_GPU'}


def worker_environment():
    # Do not pass bot, broadcast or hosting credentials to model children.
    return {key: value for key, value in os.environ.items() if key.upper() in WORKER_ENV}


class Worker:
    def __init__(self, command, cwd):
        self.inbox = Queue(maxsize=2)
        self.proc = subprocess.Popen(command, cwd=cwd, env=worker_environment(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, encoding='utf-8', text=True, bufsize=1)
        Thread(target=self._reader, daemon=True).start()
        try:
            self._next('ready', 120)
        except Exception:
            self.close()
            raise

    def _reader(self):
        try:
            while line := self.proc.stdout.readline(MAX_WIRE + 1):
                if len(line) > MAX_WIRE or not line.endswith('\n'):
                    break
                self.inbox.put_nowait(json.loads(line))
        except (Full, ValueError, OSError):
            pass
        finally:
            try:
                self.inbox.put_nowait(None)
            except Full:
                pass

    def _next(self, kind, timeout):
        try:
            result = self.inbox.get(timeout=timeout)
        except Empty as error:
            raise RuntimeError('로컬 NewAI가 제시간에 응답하지 않았어요.') from error
        if not isinstance(result, dict) or result.get('type') != kind or result.get('error'):
            raise RuntimeError('로컬 NewAI의 모델 또는 음성 응답을 읽을 수 없어요.')
        return result

    def exchange(self, request, kind, timeout):
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
        self.proc.stdin.flush()
        reply = self._next(kind, timeout).get('text')
        if not isinstance(reply, str):
            raise RuntimeError('로컬 NewAI가 잘못된 답변 형식을 반환했어요.')
        return reply

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
        self.proc.stdin.close()
        self.proc.stdout.close()


class NewAIUABridge:
    def __init__(self, *, rehearsal=False, voice=False, worker_command=None):
        root = os.getenv('NEWAI_ROOT')
        if not root:
            raise ValueError('NEWAI_ROOT에 NewAI 폴더의 절대 경로를 설정해 주세요.')
        self.root = Path(root).expanduser().resolve()
        self.python = os.getenv('NEWAI_PYTHON', sys.executable)
        self.node = os.getenv('NEWAI_NODE', 'node')
        self.worker_command = tuple(worker_command) if worker_command is not None else None
        self.worker = self.speech = None
        self.rehearsal = rehearsal
        self.voice = voice
        self.history = deque(maxlen=3)
        self.last_turn = 0
        self.lock = Lock()

    def _answer(self, message):
        if self.worker is None:
            script = self.root / 'model/gguf-context.mjs'
            if self.worker_command is None and not script.is_file():
                raise FileNotFoundError('NEWAI_ROOT가 NewAI 폴더를 가리키는지 확인해 주세요.')
            command = self.worker_command if self.worker_command is not None else [self.node, str(script), 'serve-ua']
            self.worker = Worker(command, self.root)
        if monotonic() - self.last_turn >= 900:
            self.history.clear()
        text = self.worker.exchange({'prompt': message, 'history': list(self.history) if self.rehearsal else [],
                                     'voice': self.voice}, 'answer', 55 if self.voice else 130).strip()
        if self.rehearsal and text:
            self.history.append({'user': message, 'assistant': text[:500]})
            self.last_turn = monotonic()
        return text[:350]

    def get_response(self, message):
        """Ai-project Studio/CLI contract: an iterable of strings; never tool commands."""
        if not isinstance(message, str) or not (text := message.strip()[:420]):
            return
        with self.lock:
            try:
                result = self._answer(text)
            except (OSError, RuntimeError):
                if self.worker:
                    self.worker.close()
                    self.worker = None
                raise
        if result:
            yield result

    def read_aloud(self, answer):
        if not self.voice:
            return
        player = shutil.which('ffplay')
        if not player:
            raise RuntimeError('음성을 재생하려면 ffplay를 설치하고 실행 경로에 추가해 주세요.')
        text = re.sub(r'https?://\S+|[`*_#~]', '', answer).strip()[:180]
        if not text:
            return
        with self.lock:
            try:
                if self.speech is None:
                    self.speech = Worker([self.python, '-u', str(self.root / 'model/speech_runtime.py')], self.root)
                encoded = self.speech.exchange({'operation': 'tts', 'text': text}, 'speech', 75)
            except (OSError, RuntimeError):
                if self.speech is not None:
                    self.speech.close()
                    self.speech = None
                raise
        audio = base64.b64decode(encoded, validate=True)
        if audio:
            subprocess.run([player, '-nodisp', '-autoexit', '-loglevel', 'error', '-f', 's16le', '-ar', '48000',
                            '-ac', '2', '-i', 'pipe:0'], input=audio, timeout=45, check=True)

    def reset(self):
        with self.lock:
            self.history.clear()

    def close(self):
        with self.lock:
            self.history.clear()
            for name in ('worker', 'speech'):
                process = getattr(self, name)
                if process is not None:
                    process.close()
                    setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
