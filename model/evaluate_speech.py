"""Synthetic Korean voice loopback benchmark. It never opens a microphone."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import wave

from speech_runtime import Speech, ROOT


CASES = ('안녕하세요. 오늘도 반가워요.', '점심은 따뜻한 국수 어때요?', '제가 말한 화분 이름이 뭐였죠?')


def run(root, sample):
    before = perf_counter()
    model = Speech(root)
    print(json.dumps({'load_ms': round((perf_counter() - before) * 1000)}), flush=True)
    for index, phrase in enumerate(CASES):
        start = perf_counter()
        audio = model.synth(phrase)
        ready = perf_counter()
        recognized = model.transcript(audio)
        done = perf_counter()
        print(json.dumps({'phrase': phrase, 'transcript': recognized, 'duration_ms': round(len(audio) / 192),
                          'tts_ms': round((ready - start) * 1000), 'stt_ms': round((done - ready) * 1000)},
                         ensure_ascii=False), flush=True)
        if sample and index == 0:
            with sample.open('xb') as output, wave.open(output, 'wb') as wav:
                wav.setparams((2, 2, 48_000, 0, 'NONE', 'not compressed'))
                wav.writeframes(audio)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT)
    cli.add_argument('--sample', type=Path, help='Optional generated WAV (does not overwrite).')
    args = cli.parse_args()
    run(args.root, args.sample)
