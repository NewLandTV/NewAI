import { createInterface } from 'node:readline';
import { readFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { getLlama, LlamaCompletion, LlamaLogLevel } from 'node-llama-cpp';

const PREAMBLE = `당신은 NewAI라는 한국어 대화 도우미예요. 앞 대화의 사실과 최근에 바뀐 제약을 반영하세요.
감정에는 먼저 짧게 공감하고, 꾸짖지 말고 오늘 할 수 있는 작은 도움을 제안하세요.
창작 요청에는 요청한 수만큼 서로 다른 예시를 바로 주세요. 설명 요청에는 짧고 공개 가능한 근거를 붙이세요.
상황이 모호하면 필요한 정보 한 가지만 물어보세요. 최신 사실, 전문 진단, 확실하지 않은 사실은 지어내지 마세요.
첫 문장에서 질문에 답하세요. 한국어 해요체 두세 문장만 쓰고, 역할 표식이나 내부 생각은 쓰지 마세요.`;
const LIVE = /실시간|최신|지금|현재|오늘|내일|이번\s*주/u;
const LIVE_TOPIC = /날씨|기온|강수\s*확률|미세\s*먼지|환율|주가|비트코인\s*가격/u;
const LOOKUP = /얼마|알려|예보|어때|몇|조회|확인|보여|검색|비가\s*와/u;
const CONTROL = /<\|[^<>\n]{1,80}\|>|<\/?think\s*>|\/(?:no_)?think\b/gi;
const THINK_END = '<think>\n\n</think>\n\n';
const STOP = ['<|im_end|>', '<|im_start|>', '<think>', '</think>'];
const DEADLINE = { text: 120_000, voice: 45_000 };

function clean(text) {
    return typeof text === 'string' ? text.replace(CONTROL, ' ').trim().slice(0, 800).trimEnd() : '';
}

export function policy(prompt, history) {
    const topic = prompt.match(LIVE_TOPIC)?.[0]?.replace(/\s/g, '');
    if (topic && LIVE.test(prompt) && LOOKUP.test(prompt)) {
        const source = ['날씨', '기온', '강수확률', '미세먼지'].includes(topic) ? '기상청 예보' : '은행·거래소의 최신 시세';
        return `실시간 ${topic} 정보는 지금 조회할 수 없어요. ${source}에서 확인해 보세요. 확인한 수치를 보내주면 해석을 도울게요.`;
    }
    if (/먹|식사|점심|저녁|메뉴|음식|레시피/u.test(prompt) &&
        [...history.map(pair => pair.user), prompt].some(turn => /알레르기|알러지|아나필락시스/u.test(turn))) {
        return '음식 알레르기가 있으면 메뉴 이름만으로 안전하다고 판단하기 어려워요. 피해야 할 재료와 조리 중 섞일 가능성을 먼저 확인해 주세요. 확인된 재료가 있으면 그 재료 안에서 선택지를 같이 골라 볼게요.';
    }
    return '';
}

export function formatPrompt(prompt, history, voice) {
    const role = (name, text) => `<|im_start|>${name}\n${text}<|im_end|>\n`;
    const system = PREAMBLE + (voice ? '\n음성 답변이므로 한두 문장으로 말하고 마크다운은 쓰지 마세요.' : '');
    return role('system', system) + history.map(pair => role('user', clean(pair.user)) +
        role('assistant', clean(pair.assistant))).join('') + role('user', clean(prompt)) +
        `<|im_start|>assistant\n${THINK_END}`;
}

function textOnly(value, maximum) {
    const text = clean(value.split(/<\/?think\s*>/i)[0]).slice(0, maximum);
    const last = [...text.matchAll(/[.!?。！？](?=\s|$)/gu)].at(-1);
    return last && last.index > text.length * .52 ? text.slice(0, last.index + 1) : text;
}

export async function openModel(filename = process.env.NEWAI_GGUF || path.join(
    process.env.NEWAI_MODEL_ROOT || path.join(os.homedir(), '.cache', 'newai'), 'context', 'Qwen3-1.7B-Q4_K_M.gguf')) {
    const gpu = process.env.NEWAI_GPU === 'auto' ? 'auto' : false;
    const llama = await getLlama({ gpu, build: 'never', logLevel: LlamaLogLevel.error });
    const model = await llama.loadModel({ modelPath: filename });
    const threads = Number(process.env.NEWAI_THREADS) || 2;
    const context = await model.createContext({ contextSize: 2048, threads: Math.min(16, Math.max(1, threads)) });
    const sequence = context.getSequence();
    const completion = new LlamaCompletion({ contextSequence: sequence });
    return {
        close: async () => { await context.dispose(); await model.dispose(); await llama.dispose(); },
        async answer(request, seed) {
            const prompt = clean(request.prompt);
            if (!prompt) throw new Error('invalid prompt');
            const history = (Array.isArray(request.history) ? request.history.slice(-3) : [])
                .flatMap(pair => clean(pair?.user) && clean(pair?.assistant) ? [{ user: clean(pair.user), assistant: clean(pair.assistant) }] : []);
            const guarded = policy(prompt, history);
            if (guarded) return { text: guarded, guarded: true };
            let tokens;
            do {
                tokens = model.tokenize(formatPrompt(prompt, history, request.voice === true), true);
            } while (tokens.length > 1650 && history.shift());
            if (tokens.length > 1650) return { text: '질문에서 가장 궁금한 부분을 한두 문장으로 좁혀 줄래요?', guarded: true };
            const signal = AbortSignal.timeout(request.voice ? DEADLINE.voice : DEADLINE.text);
            try {
                const text = await completion.generateCompletion(tokens, { maxTokens: request.voice ? 75 : 130,
                    seed, temperature: .7, topP: .8, topK: 20, repeatPenalty: { penalty: 1.1 },
                    customStopTriggers: STOP, disableContextShift: true, signal });
                return { text: textOnly(text, request.voice ? 180 : 800), guarded: false };
            } catch (error) {
                if (!signal.aborted) throw error;
                return { text: '', guarded: false, timed_out: true };
            } finally {
                // No prior user's prompt is retained as reusable inference context.
                await sequence.clearHistory();
            }
        },
    };
}

async function main() {
    const [command = 'serve', arg] = process.argv.slice(2);
    const model = await openModel();
    const show = item => console.log(JSON.stringify(item));
    try {
        if (command === 'ask') {
            const result = await model.answer({ prompt: arg }, 42);
            console.log(result.text);
        } else if (command === 'evaluate') {
            const cases = JSON.parse(await readFile(new URL('./context_eval.json', import.meta.url), 'utf8'));
            for (const [index, row] of cases.entries()) {
                const start = performance.now();
                const result = await model.answer(row, 42 + index);
                show({ ...row, ...result, elapsed_ms: Math.round(performance.now() - start) });
            }
        } else {
            show({ type: 'ready' });
            const lines = createInterface({ input: process.stdin });
            for await (const line of lines) {
                try {
                    const result = await model.answer(JSON.parse(line));
                    show(result.timed_out ? { type: 'answer', error: 'generation_timed_out' } :
                        { type: 'answer', text: result.text });
                } catch {
                    show({ type: 'answer', error: 'generation_failed' });
                }
            }
        }
    } finally { await model.close(); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    main().catch(error => { console.error(`로컬 GGUF 실행 실패: ${error.message}`); process.exitCode = 1; });
}
