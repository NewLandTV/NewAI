const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');
const path = require('node:path');

class LocalModel {
    constructor({ python = process.env.NEWAI_PYTHON || (process.platform === 'win32' ? 'python' : 'python3'),
        checkpoint = process.env.NEWAI_CHECKPOINT || 'checkpoints/daily-final/model.json' } = {}) {
        const { DISCORD_BOT_TOKEN: _token, ...modelEnv } = process.env;
        const script = path.extname(checkpoint).toLowerCase() === '.json' ? 'daily_model.py' : 'newai.py';
        this.child = spawn(python, [path.join(__dirname, 'model', script), 'serve',
            '--checkpoint', path.resolve(checkpoint)], { env: modelEnv, stdio: ['pipe', 'pipe', 'inherit'] });
        this.ready = this.waitFor('ready', 30_000);
        this.lines = createInterface({ input: this.child.stdout });
        this.lines.on('line', (line) => {
            try {
                const message = JSON.parse(line);
                if (!this.pending || message.type !== this.pending.type) throw new Error('예상하지 못한 모델 응답이에요.');
                this.finish(message.error ? new Error('로컬 추론에 실패했어요.') : null, message.text);
            } catch (error) {
                this.close(error);
            }
        });
        this.child.on('error', () => this.close(new Error('파이썬을 실행하지 못했어요.')));
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

    async ask(prompt) {
        await this.ready;
        if (this.closed) throw new Error('로컬 모델을 사용할 수 없어요.');
        if (this.pending) throw Object.assign(new Error('모델이 사용 중이에요.'), { code: 'BUSY' });
        const result = this.waitFor('answer', 30_000);
        this.child.stdin.write(`${JSON.stringify({ prompt })}\n`);
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

module.exports = { LocalModel };
