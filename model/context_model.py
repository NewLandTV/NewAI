"""Local decoder-only generation for NewAI. External networks are not used at inference."""

import argparse
import json
from pathlib import Path
import re
import sys

from fetch_runtime import ROOT


SYSTEM = ('너는 한국어 대화 도우미 NewAI야. 사용자 문장을 따라 하지 말고 질문에 직접 답해. '
          '앞 대화의 사실과 제약을 지켜. 감정을 단정하지 말고 필요할 때 짧게 공감해. '
          '창작 요청에는 실제 예시를 새로 만들어 줘. 설명이 필요하면 결론과 공개할 수 있는 이유를 짧게 말해. '
          '모르는 사실은 지어내지 말고, 모호하면 중요한 한 가지만 되물어 봐. '
          '한국어 해요체로 짧게 답해. 마크다운, 역할 표식, 생각 과정을 출력하지 마.')
VOICE_SYSTEM = ' 음성 대화이므로 목록 대신 말로 자연스럽게 한두 문장으로 답해.'
ROLE_MARKER = re.compile(r'<\|[^<>\n]{1,80}\|>')
LIVE = re.compile(r'실시간|최신|지금|현재|오늘|내일|이번\s*주')
LIVE_TOPIC = re.compile(r'날씨|기온|강수\s*확률|미세\s*먼지|환율|주가|비트코인\s*가격')
LOOKUP = re.compile(r'얼마|알려|예보|어때|몇|조회|확인|보여|검색|비가\s*와')


def clean(text, maximum=800):
    return ROLE_MARKER.sub('', text).strip()[:maximum] if isinstance(text, str) else ''


def messages_for(prompt, history, voice=False):
    messages = [{'role': 'system', 'content': SYSTEM + (VOICE_SYSTEM if voice else '')}]
    if isinstance(history, list):
        for pair in history[-3:]:
            if isinstance(pair, dict) and clean(pair.get('user')) and clean(pair.get('assistant')):
                messages += [{'role': 'user', 'content': clean(pair['user'])},
                             {'role': 'assistant', 'content': clean(pair['assistant'])}]
    return messages + [{'role': 'user', 'content': clean(prompt)}]


def live_help(prompt):
    topic = LIVE_TOPIC.search(prompt)
    if not (topic and LIVE.search(prompt) and LOOKUP.search(prompt)):
        return None
    name = ''.join(topic.group().split())
    source = '기상청 예보' if name in {'날씨', '기온', '강수확률', '미세먼지'} else '은행·거래소의 최신 시세'
    return f'실시간 {name} 정보는 지금 조회할 수 없어요. {source}에서 확인해 보세요. 확인한 수치를 보내주면 해석을 도울게요.'


class ContextModel:
    def __init__(self, base, adapter=None, threads=2, device='cpu'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                                                        use_safetensors=True, dtype=torch.float32).to(device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, local_files_only=True).merge_and_unload()
        self.model.eval()

    def generate(self, prompt, history=None, voice=False, sample=True):
        if not clean(prompt):
            return ''
        support = live_help(prompt)
        if support:
            return support
        tokenizer = self.tokenizer
        items = messages_for(prompt, history, voice)
        maximum = min(1536, getattr(self.model.config, 'max_position_embeddings', 1536) - 160)
        maximum = max(96, maximum)
        while True:
            rendered = tokenizer.apply_chat_template(items, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
            if encoded.input_ids.shape[1] <= maximum or len(items) <= 2:
                break
            del items[1:3]  # Keep system + current question; discard oldest pair first.
        if encoded.input_ids.shape[1] > maximum:
            return '질문을 한두 문장으로 줄여 주면 그 부분부터 답해 볼게요.'
        encoded = encoded.to(self.device)
        eos = tokenizer.eos_token_id
        im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        stop = sorted({token for token in (eos, im_end) if token is not None and token != tokenizer.unk_token_id})
        options = dict(max_new_tokens=58 if voice else 112, do_sample=sample, repetition_penalty=1.12,
                       pad_token_id=tokenizer.pad_token_id, eos_token_id=stop, use_cache=True)
        if sample:
            options.update(temperature=.72, top_p=.88, top_k=35)
        else:
            options.update(temperature=None, top_p=None, top_k=None)
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **options)[0, encoded.input_ids.shape[1]:]
        text = clean(tokenizer.decode(output, skip_special_tokens=True), 700)
        # Avoid speaking a broken final fragment when generation hits the token budget.
        end = list(re.finditer(r'[.!?。！？](?=\s|$)', text))
        if len(output) == options['max_new_tokens'] and end and end[-1].end() >= len(text) // 2:
            text = text[:end[-1].end()]
        return text


def show(item):
    print(json.dumps(item, ensure_ascii=False), flush=True)


def serve(model):
    show({'type': 'ready'})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not clean(request.get('prompt')):
                raise ValueError('prompt is required')
            text = model.generate(request['prompt'], request.get('history'), request.get('voice') is True)
            show({'type': 'answer', 'text': text})
        except (ValueError, KeyError, TypeError, RuntimeError):
            show({'type': 'answer', 'error': 'generation_failed'})


if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('serve', 'ask'))
    parser.add_argument('prompt', nargs='?')
    parser.add_argument('--base', type=Path, default=ROOT / 'qwen')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    args = parser.parse_args()
    if args.command == 'ask' and not clean(args.prompt):
        parser.error('질문을 입력해 주세요.')
    model = ContextModel(args.base, args.adapter, args.threads, args.device)
    if args.command == 'ask':
        print(model.generate(args.prompt, sample=False))
    else:
        serve(model)
