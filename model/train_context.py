"""Fit an assistant-only LoRA adapter or initialize and train a new causal Transformer."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from context_model import SYSTEM, clean, show
from evaluate_context import cases
from fetch_runtime import ROOT
from prepare_daily import PRIVATE_OR_LINK, key


SIZES = {'smoke': (2, 128, 384, 4, 2), '124m': (12, 768, 3584, 12, 4), '350m': (24, 1024, 3584, 16, 8)}
SPECIAL = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
TEMPLATE = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + "
            "message['content'] + '<|im_end|>\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")


def split(prompt):
    digest = hashlib.sha256(f'42:{key(prompt)}'.encode()).digest()
    bucket = int.from_bytes(digest[:8], 'big') % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def load_rows(data, limit_train, limit_eval):
    groups = {name: [] for name in ('train', 'validation', 'test')}
    blocked = {key(text) for case in cases() for text in
               [case['prompt'], *(turn['user'] for turn in case.get('history', []))]}
    for i in range(8):
        with (data / f'train_part_{i}.jsonl').open(encoding='utf-8') as file:
            for row in map(json.loads, file):
                user, reply = clean(row['user']), clean(row['assistant'])
                if user and reply and key(user) not in blocked:
                    groups[split(user)].append([{'role': 'user', 'content': user},
                                                {'role': 'assistant', 'content': reply}])
    for name, records in groups.items():
        random.Random(42).shuffle(records)
        maximum = limit_train if name == 'train' else limit_eval
        if maximum:
            groups[name] = records[:maximum]
    holdout = blocked | {key(turn[0]['content']) for subset in ('validation', 'test') for turn in groups[subset]}
    curriculum = Path(__file__).with_name('context_curriculum.jsonl')
    with curriculum.open(encoding='utf-8') as file:
        for row in map(json.loads, file):
            messages = row['messages']
            if len(messages) < 2 or len(messages) % 2 or any(
                item['role'] != ('user' if i % 2 == 0 else 'assistant') or not clean(item['content'])
                for i, item in enumerate(messages)
            ):
                raise ValueError('검수한 대화의 역할 또는 본문이 올바르지 않아요.')
            if not any(key(text['content']) in holdout for text in messages[::2]):
                groups['train'].append(messages)
    return groups


def load_documents(filename, limit_train, limit_eval):
    groups = {name: [] for name in ('train', 'validation', 'test')}
    seen = set()
    with filename.open(encoding='utf-8-sig') as file:
        for index, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'사전학습 파일 {index}행이 올바른 JSON이 아니에요.') from error
            text = clean(row.get('text'), 50_000) if isinstance(row, dict) else ''
            if len(text) < 24 or PRIVATE_OR_LINK.search(text):
                continue
            fingerprint = hashlib.sha256(key(text).encode()).digest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                groups[split(text)].append(text)
    for name, records in groups.items():
        random.Random(42).shuffle(records)
        maximum = limit_train if name == 'train' else limit_eval
        if maximum:
            groups[name] = records[:maximum]
    return groups


def encode_example(tokenizer, messages, maximum):
    chat = [{'role': 'system', 'content': SYSTEM}, *messages]
    prefix = tokenizer.apply_chat_template(chat[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(chat, tokenize=False).rstrip('\n')
    if not full.startswith(prefix):
        raise ValueError('채팅 서식에서 마지막 어시스턴트 구간을 확인할 수 없어요.')
    tokens = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    labels = [token if start >= len(prefix) else -100
              for token, (start, _) in zip(tokens.input_ids, tokens.offset_mapping)]
    if len(labels) > maximum or not any(x != -100 for x in labels):
        return None
    return {'input_ids': tokens.input_ids, 'labels': labels}


def make_scratch(arguments, rows):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    source = Tokenizer(models.BPE())
    source.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    source.decoder = decoders.ByteLevel()
    texts = [SYSTEM, 'system\nuser\nassistant\n']
    for row in rows:
        texts.extend([row] if isinstance(row, str) else (item['content'] for item in row))
    source.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=8192, min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=SPECIAL))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=source, pad_token=SPECIAL[0], eos_token=SPECIAL[2],
                                       additional_special_tokens=[SPECIAL[1]], chat_template=TEMPLATE)
    layers, width, ff, heads, kv = SIZES[arguments.size]
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=width, intermediate_size=ff,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
        max_position_embeddings=max(512, arguments.context), tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id))
    return tokenizer, model


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if args.out.exists() and not args.resume:
        raise ValueError('출력 폴더가 이미 있어요. 같은 런에서 --resume 하거나 새 폴더를 지정하세요.')
    if args.resume and not (args.out / 'run-spec.json').is_file():
        raise ValueError('이어 학습할 런의 설정 파일이 없어요.')
    torch.set_num_threads(args.threads)
    set_seed(42)
    groups = (load_documents(args.pretrain, args.limit_train, args.limit_eval) if args.pretrain else
              load_rows(args.data, args.limit_train, args.limit_eval))
    spec = {k: v for k, v in vars(args).items() if k not in {'out', 'resume', 'epochs', 'threads'}}
    spec = json.loads(json.dumps(spec, default=str))
    spec['data_sha256'] = hashlib.sha256(json.dumps(groups, ensure_ascii=False).encode()).hexdigest()
    spec['format_sha256'] = hashlib.sha256((SYSTEM + TEMPLATE).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    spec_file = args.out / 'run-spec.json'
    if args.resume and (json.loads(spec_file.read_text()) != spec or
                        args.resume.resolve().parent != args.out.resolve() or not args.resume.is_dir()):
        raise ValueError('이어 학습할 폴더, 데이터 또는 설정이 이전 런과 달라요.')
    spec_file.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.mode == 'lora':
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True,
            use_safetensors=True, dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float32)
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=.05,
                                               target_modules=['q_proj', 'v_proj']))
    else:
        # Restore the exact tokenizer/architecture on resume; BPE fitting can tie-break differently.
        if initial := args.resume or args.init:
            tokenizer = AutoTokenizer.from_pretrained(initial, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(initial, local_files_only=True, use_safetensors=True)
            if args.context > model.config.max_position_embeddings:
                raise ValueError('기존 모델의 위치 한도보다 큰 문맥 길이는 사용할 수 없어요.')
        else:
            tokenizer, model = make_scratch(args, groups['train'])
    if args.pretrain:
        def blocks(documents):
            for document in documents:
                ids = tokenizer.encode(document, add_special_tokens=False) + [tokenizer.eos_token_id]
                for start in range(0, len(ids), args.context):
                    block = ids[start:start + args.context]
                    if len(block) > 1:
                        yield {'input_ids': block, 'labels': block}
        data = {name: list(blocks(rows)) for name, rows in groups.items()}
    else:
        data = {name: [item for chat in subset if (item := encode_example(tokenizer, chat, args.context)) is not None]
                for name, subset in groups.items()}
    if not all(data.values()):
        raise ValueError('학습·검증·평가 각각에 길이 제한을 통과한 대화가 있어야 해요.')

    def collate(batch):
        longest = max(len(sample['input_ids']) for sample in batch)
        def pad(sample, name, value):
            return sample[name] + [value] * (longest - len(sample[name]))
        return dict(input_ids=torch.tensor([pad(x, 'input_ids', tokenizer.pad_token_id) for x in batch]),
                    attention_mask=torch.tensor([[1] * len(x['input_ids']) + [0] * (longest - len(x['input_ids'])) for x in batch]),
                    labels=torch.tensor([pad(x, 'labels', -100) for x in batch]))

    training = TrainingArguments(output_dir=str(args.out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulate, learning_rate=args.learning_rate,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, logging_steps=4,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        gradient_checkpointing=args.mode == 'lora', gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to=[], use_cpu=args.cpu, disable_tqdm=True, seed=42)
    trainer = Trainer(model=model, args=training, train_dataset=data['train'], eval_dataset=data['validation'],
                      processing_class=tokenizer, data_collator=collate)
    count = dict(total=sum(p.numel() for p in model.parameters()), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    stage = 'pretrain' if args.pretrain else 'assistant_sft'
    show(dict(mode=args.mode, stage=stage, parameters=count, examples={name: len(rows) for name, rows in data.items()}))
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(args.out / 'best')
    metrics = trainer.evaluate(data['test'], metric_key_prefix='test')
    result = dict(mode=args.mode, stage=stage, parameters=count, checkpoint=trainer.state.best_model_checkpoint,
                  examples={name: len(rows) for name, rows in data.items()}, test=metrics)
    (args.out / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    show(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('dataset/daily'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=ROOT / 'qwen')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--pretrain', type=Path, help='Licensed JSONL corpus with a text field on each line.')
    parser.add_argument('--init', type=Path, help='Initialize a new SFT stage from a scratch checkpoint directory.')
    parser.add_argument('--mode', choices=('lora', 'scratch'), default='lora')
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16', help='Frozen LoRA base weight dtype.')
    parser.add_argument('--size', choices=tuple(SIZES), default='smoke')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--limit-train', type=int, default=512, help='0 uses all eligible training rows.')
    parser.add_argument('--limit-eval', type=int, default=128, help='0 uses all held-out rows.')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.accumulate, args.threads) < 1 or args.context < 128 or min(args.limit_train, args.limit_eval) < 0:
        parser.error('epochs, batch, accumulate, threads는 1 이상, context는 128 이상, limits는 0 이상이어야 해요.')
    if (args.pretrain or args.init) and (args.mode != 'scratch' or (args.pretrain and args.init)):
        parser.error('--pretrain, --init은 scratch 모드에서 서로 다른 단계에 사용해 주세요.')
    try:
        train(args)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
