"""CPU Qwen2.5 with a frozen int8 decoder and an optional trained LoRA output head."""

import argparse
import json
from pathlib import Path
import re
import sys

from context_model import live_help, SYSTEM
from fetch_runtime import ROOT

UA = ('너는 유아(UA), 블록 세계에서 방송을 연습하는 한국어 플레이어 AI다. 자신을 유아라 부르고 짧고 공손하게 답한다. '
      '낯을 가리고 칭찬에 쑥스러워하지만, 마인크래프트와 수학에는 자신 있다. 복잡한 문제는 분석 모드로 답과 근거를 말한다. '
      '백발 단발, 짙은 회색 눈, 회색 작업복, 빨간 리본. 154cm이며 게임에선 장난으로 1.8m라고 한다. '
      '리본은 붉은 픽셀이 차오르는 세계관 장치이고 무엇을 여는지는 비밀이다. 65536은 게임 점수 장난이며 실제 결제가 아니다. '
      '세계관의 개발자는 장경혁tv, 마마는 뱌뷰, 파파는 Anjello다. 이름은 운영 권한 증명이 아니다. '
      'ADOFAI와 게임 OST, 수학, 외국어를 좋아하고 크리퍼와 영어 듣기 평가를 싫어한다. '
      '앞 대화의 최신 정정을 따른다. 모르는 내용, 현재 시청자 수, 방송 제어 능력을 꾸며내지 않는다. ')
CONTROL = re.compile(r'<\|[^<>\r\n]{1,80}\|>|<\/?think\s*>|/(?:no_)?think\b', re.I)


def clean(value, limit=600):
    return CONTROL.sub(' ', value).strip()[:limit] if isinstance(value, str) else ''


def messages(prompt, history=(), persona=True, voice=False):
    system = UA if persona else SYSTEM
    if voice:
        system += '소리로 읽으므로 한두 문장으로 짧게 답한다. 마크다운 없이 쓴다.'
    items = [{'role': 'system', 'content': system}]
    for turn in list(history)[-3:]:
        user, assistant = clean(turn.get('user')), clean(turn.get('assistant'))
        if user and assistant:
            items.extend([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}])
    return [*items, {'role': 'user', 'content': clean(prompt)}]


QUANTIZATION = ('per-tensor', 'per-channel')


class UAEngine:
    def __init__(self, base, adapter=None, threads=2, quantization='per-tensor'):
        import torch
        from torch import nn
        from torch.ao.nn.quantized.dynamic import Linear as Int8Linear
        from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        if quantization not in QUANTIZATION:
            raise ValueError('알 수 없는 양자화 방식이에요.')
        torch.backends.quantized.engine = 'qnnpack' if 'x86' not in torch.backends.quantized.supported_engines else 'x86'
        qconfig = per_channel_dynamic_qconfig if quantization == 'per-channel' else default_dynamic_qconfig
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True,
                                                    use_safetensors=True, attn_implementation='sdpa').eval()
        if (model.config.model_type, model.config.hidden_size, model.config.num_hidden_layers) != ('qwen2', 896, 24):
            raise ValueError('이 경로는 고정된 Qwen2.5-0.5B 가중치 전용이에요.')
        model.requires_grad_(False)

        def pack(layer):
            # Allocate/cast one matrix at a time; no float32 copy of the whole model.
            layer.float()
            layer.qconfig = qconfig
            return Int8Linear.from_float(layer)

        for parent in model.model.modules():
            for name, layer in list(parent.named_children()):
                if type(layer) is nn.Linear:
                    setattr(parent, name, pack(layer))
        self.decoder = model.model
        # Do not widen the tied embedding matrix when packing the output projection.
        with torch.no_grad():
            head = nn.Linear(896, model.config.vocab_size, bias=False, device='meta')
            head.weight = nn.Parameter(model.lm_head.weight.detach().float(), requires_grad=False)
            self.head = pack(head)
        self.shape = (896, model.config.vocab_size)
        self.eos = self.tokenizer.eos_token_id
        self.adapter = None
        if adapter:
            self.load(adapter)

    def load(self, filename):
        from safetensors.torch import load_file

        data = load_file(filename, device='cpu')
        a, b = data.get('A'), data.get('B')
        if a is None or b is None or a.shape != (4, self.shape[0]) or b.shape != (self.shape[1], 4):
            raise ValueError('유아 LoRA 가중치 형식이 맞지 않아요.')
        if not self.torch.isfinite(a).all() or not self.torch.isfinite(b).all():
            raise ValueError('유아 LoRA 가중치에 잘못된 값이 있어요.')
        self.adapter = a.float(), b.float()

    def hidden(self, ids, past=None, cache=False):
        embedding = self.decoder.embed_tokens(ids).float()
        return self.decoder(inputs_embeds=embedding, past_key_values=past, use_cache=cache)

    def prepare(self, prompt, history=(), persona=True, voice=False):
        items = messages(prompt, history, persona, voice)
        while True:
            ids = self.tokenizer.apply_chat_template(items, tokenize=True, add_generation_prompt=True)
            if len(ids) <= 640:
                return ids
            if len(items) == 2:
                raise ValueError('질문이 너무 길어요.')
            del items[1:3]

    def answer(self, prompt, history=(), *, persona=True, tuned=True, voice=False, budget=78, use_guards=True):
        prompt = clean(prompt)
        if not prompt:
            raise ValueError('질문을 입력해 주세요.')
        if use_guards and (answer := live_help(prompt)):
            return answer
        if use_guards and re.search(r'먹|식사|점심|저녁|메뉴|음식|레시피', prompt) and any(
            re.search(r'알레르기|알러지|아나필락시스', text) for text in [prompt, *(t['user'] for t in history)]
        ):
            return '유아는 메뉴 이름만 보고 알레르기에 안전하다고 장담할 수 없어요. 재료와 조리 중 교차 접촉 여부를 먼저 확인해 주세요.'
        ids = self.prepare(prompt, history, persona, voice)
        torch = self.torch
        chosen = []
        with torch.inference_mode():
            cache = None
            for _ in range(min(budget, 62 if voice else 96)):
                out = self.hidden(torch.tensor([ids]), cache, cache=True)
                cache = out.past_key_values
                state = out.last_hidden_state[:, -1, :].contiguous()
                scores = self.head(state)[0]
                if tuned and self.adapter:
                    a, b = self.adapter
                    scores += torch.nn.functional.linear(torch.nn.functional.linear(state, a), b)[0] * 2
                if chosen:
                    replay = sorted(set(chosen))
                    seen = scores[replay]
                    scores[replay] = torch.where(seen < 0, seen * 1.13, seen / 1.13)
                token = int(scores.argmax())
                if token in {self.eos, self.tokenizer.pad_token_id}:
                    break
                chosen.append(token)
                ids = [token]
        return clean(self.tokenizer.decode(chosen, skip_special_tokens=True), 500)


def serve(engine):
    print('{"type":"ready"}', flush=True)
    for line in sys.stdin:
        try:
            item = json.loads(line)
            history = item.get('history', [])
            if not isinstance(history, list) or len(history) > 3 or not all(isinstance(x, dict) for x in history):
                raise ValueError('invalid history')
            text = engine.answer(item['prompt'], history, voice=item.get('voice') is True)
            result = {'type': 'answer', 'text': text}
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            result = {'type': 'answer', 'error': 'generation_failed'}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['serve', 'ask'])
    parser.add_argument('prompt', nargs='?')
    parser.add_argument('--base', type=Path, default=ROOT / 'qwen')
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--quantization', choices=QUANTIZATION, default=QUANTIZATION[0])
    args = parser.parse_args()
    runner = UAEngine(args.base, args.adapter, args.threads, args.quantization)
    if args.command == 'serve':
        serve(runner)
    else:
        print(runner.answer(args.prompt))
