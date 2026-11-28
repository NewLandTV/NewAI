"""Resident Korean ASR/TTS worker; audio stays in memory and is never used for training."""

import argparse
import base64
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import wave

from fetch_runtime import PACKS, ROOT, checksum


MAX_INPUT = 48_000 * 2 * 2 * 12


class Speech:
    def __init__(self, root):
        import numpy as np
        import onnxruntime as ort
        from faster_whisper import WhisperModel

        self.np = np
        self.asr = WhisperModel(str(root / 'whisper'), device='cpu', compute_type='int8',
                                local_files_only=True, cpu_threads=2, num_workers=1)
        home = root / 'canora'
        frontend = home / 'g2p_ko.py'
        if checksum(frontend) != PACKS['canora'][2]['g2p_ko.py']:
            raise ValueError('Canora 발음 표기 파일의 체크섬이 달라요.')
        spec = importlib.util.spec_from_file_location('newai_canora_g2p', frontend)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.g2p = module.KoreanG2P()
        self.vocab = json.loads((home / 'config.json').read_text(encoding='utf-8'))['vocab']
        self.voice = np.fromfile(home / 'voices/seoyeon.bin', dtype='<f4').reshape(510, 256)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.tts = ort.InferenceSession(str(home / 'onnx/model_quantized.onnx'), options,
                                       providers=['CPUExecutionProvider'])

    def transcript(self, pcm):
        if len(pcm) > MAX_INPUT or len(pcm) % 4 or len(pcm) < 48_000 * 4 * .28:
            return ''
        audio = io.BytesIO()
        with wave.open(audio, 'wb') as wav:
            wav.setparams((2, 2, 48_000, 0, 'NONE', 'not compressed'))
            wav.writeframes(pcm)
        audio.seek(0)
        segments, _ = self.asr.transcribe(audio, language='ko', task='transcribe', beam_size=1,
                                           condition_on_previous_text=False, vad_filter=True,
                                           vad_parameters={'min_silence_duration_ms': 350})
        parts = [item.text.strip() for item in segments if item.no_speech_prob < .7 and item.avg_logprob > -1.35]
        return ' '.join(parts)[:800]

    def synth(self, text):
        np = self.np
        text = re.sub(r'[`*_#~]', '', text).strip()
        phonemes, _ = self.g2p(text[:200])
        ids = [self.vocab[p] for p in phonemes if p in self.vocab][:500]
        if not ids:
            return b''
        waveform = self.tts.run(None, {
            'input_ids': np.array([[0, *ids, 0]], dtype=np.int64),
            'style': self.voice[max(0, min(len(phonemes) - 1, 509))][None, :],
            'speed': np.ones(1, dtype=np.float32),
        })[0].reshape(-1)
        if not len(waveform) or not np.isfinite(waveform).all():
            return b''
        # Double the 24 kHz waveform by linear interpolation; Discord takes 48 kHz stereo.
        mono = np.empty(len(waveform) * 2, dtype=np.float32)
        mono[::2] = waveform
        mono[1:-1:2] = (waveform[:-1] + waveform[1:]) * .5
        mono[-1] = waveform[-1]
        samples = np.clip(mono * 32767, -32768, 32767).astype('<i2')
        return np.repeat(samples, 2).tobytes()


def show(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def serve(model):
    show({'type': 'ready'})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get('operation')
            if operation == 'stt' and len(request['pcm']) <= MAX_INPUT * 4 // 3 + 4:
                text = model.transcript(base64.b64decode(request['pcm'], validate=True))
                show({'type': 'speech', 'text': text})
            elif operation == 'tts' and isinstance(request['text'], str) and 1 <= len(request['text']) <= 200:
                pcm = model.synth(request['text'])
                show({'type': 'speech', 'text': base64.b64encode(pcm).decode('ascii')})
            else:
                raise ValueError('invalid speech request')
        except Exception:
            # Do not log utterances, transcripts or exception text that may contain them.
            show({'type': 'speech', 'error': 'speech_unavailable'})


if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT)
    args = cli.parse_args()
    serve(Speech(args.root))
