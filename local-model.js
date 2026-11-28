const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');
const path = require('node:path');

class ModelProcess {
    constructor(script, args, { python = process.env.NEWAI_PYTHON || (process.platform === 'win32' ? 'python' : 'python3'),
        startupTimeout = 30_000, executable = python } = {}) {
        const { DISCORD_BOT_TOKEN: _token, ...modelEnv } = process.env;
        this.child = spawn(executable, [path.join(__dirname, 'model', script), ...args],
            { env: modelEnv, stdio: ['pipe', 'pipe', 'inherit'], windowsHide: true });
        this.ready = this.waitFor('ready', startupTimeout);
        this.lines = createInterface({ input: this.child.stdout });
        this.lines.on('line', (line) => {
            try {
                const message = JSON.parse(line);
                if (!this.pending || message.type !== this.pending.type) throw new Error('예상하지 못한 모델 응답이에요.');
                const error = message.error === 'generation_timed_out'
                    ? Object.assign(new Error('로컬 생성 시간이 초과됐어요.'), { code: 'TIMEOUT' })
                    : message.error ? new Error('로컬 추론에 실패했어요.') : null;
                this.finish(error, message.text);
            } catch (error) {
                this.close(error);
            }
        });
        this.child.on('error', () => this.close(new Error('로컬 모델 실행 파일을 열지 못했어요.')));
        this.child.stdin.on('error', () => this.close());
        this.child.on('close', () => this.close(new Error('로컬 모델 실행이 종료됐어요.')));
    }

    waitFor(type, milliseconds) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => this.close(new Error('로컬 모델 응답 시간이 초과됐어요.')), milliseconds);
            this.pending = { type, resolve, reject, timer };
        });
    }

    finish(error, value) {
        const pending = this.pending;
        if (!pending) return;
        this.pending = null;
        clearTimeout(pending.timer);
        if (error) pending.reject(error);
        else pending.resolve(value);
    }

    async request(message, type, timeout = 30_000) {
        await this.ready;
        if (this.closed) throw new Error('로컬 모델을 사용할 수 없어요.');
        if (this.pending) throw Object.assign(new Error('모델이 사용 중이에요.'), { code: 'BUSY' });
        const result = this.waitFor(type, timeout);
        this.child.stdin.write(`${JSON.stringify(message)}\n`);
        return result;
    }

    close(error = new Error('로컬 모델 연결을 닫았어요.')) {
        if (this.closed) return;
        this.closed = true;
        this.finish(error);
        this.lines.close();
        this.child.stdin.destroy();
        this.child.kill();
    }
}

class LocalModel extends ModelProcess {
    constructor({ python, engine = process.env.NEWAI_ENGINE || 'daily',
        checkpoint = process.env.NEWAI_CHECKPOINT || 'checkpoints/daily-final/model.json' } = {}) {
        if (!['daily', 'context', 'research', 'ua'].includes(engine)) {
            throw new Error('NEWAI_ENGINE은 daily, context, research, ua 중 하나로 지정해 주세요.');
        }
        if (engine === 'context' || engine === 'ua') {
            super('gguf-context.mjs', [engine === 'ua' ? 'serve-ua' : 'serve'],
                { executable: process.execPath, startupTimeout: 120_000 });
        } else if (engine === 'research') {
            const args = ['serve'];
            for (const [setting, option] of [['NEWAI_CONTEXT_BASE', '--base'], ['NEWAI_ADAPTER', '--adapter'],
                ['NEWAI_CONTEXT_DEVICE', '--device']]) {
                if (process.env[setting]) args.push(option, process.env[setting]);
            }
            super('context_model.py', args, { python, startupTimeout: 120_000 });
        } else {
            const script = path.extname(checkpoint).toLowerCase() === '.json' ? 'daily_model.py' : 'newai.py';
            super(script, ['serve', '--checkpoint', path.resolve(checkpoint)], { python });
        }
        this.engine = engine;
    }

    ask(prompt, { history = [], voice = false } = {}) {
        const timeout = ['context', 'ua'].includes(this.engine) ? (voice ? 55_000 : 130_000) :
            this.engine === 'research' ? 60_000 : 30_000;
        return this.request({ prompt, history, voice }, 'answer', timeout);
    }
}

class LocalSpeech extends ModelProcess {
    constructor({ python } = {}) { super('speech_runtime.py', [], { python, startupTimeout: 90_000 }); }

    transcript(pcm) { return this.request({ operation: 'stt', pcm: pcm.toString('base64') }, 'speech'); }

    async synth(text) {
        const pcm = await this.request({ operation: 'tts', text }, 'speech', 75_000);
        return Buffer.from(pcm, 'base64');
    }
}

module.exports = { LocalModel, LocalSpeech };
