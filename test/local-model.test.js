const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { mkdtempSync, mkdirSync, rmSync, writeFileSync } = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');
const { LocalModel } = require('../local-model');

test('eight shards → checkpoint → resume → test metrics → local process answer', { timeout: 90_000 }, async () => {
    const fixture = mkdtempSync(path.join(os.tmpdir(), 'newai-test-'));
    const data = path.join(fixture, 'dataset');
    const output = path.join(fixture, 'checkpoints');
    const python = process.env.NEWAI_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
    const checkpoint = path.join(output, 'best.pt');
    const run = (...args) => execFileSync(python, [path.resolve(__dirname, '../model/newai.py'), ...args], {
        encoding: 'utf8', timeout: 25_000,
    }).trim().split('\n').map((line) => JSON.parse(line));

    let model;
    try {
        mkdirSync(data);
        for (let shard = 0; shard < 8; shard += 1) {
            const lines = Array.from({ length: 40 }, (_, n) => {
                const prompt = `합성 질문 ${shard}-${n}`;
                return JSON.stringify(n % 2 ? { user: prompt, assistant: '반가워요!' }
                    : { question: prompt, answer: '반가워요!' });
            });
            writeFileSync(path.join(data, `train_part_${shard}.jsonl`), lines.join('\n'));
        }
        const [split, first] = run('train', '--data', data, '--out', output, '--epochs', '1', '--width', '32',
            '--layers', '1', '--context', '48', '--learning-rate', '0.01');
        assert.equal(split.train + split.validation + split.test, 320);
        assert.equal(first.epoch, 1);
        assert.equal(first.best_saved, true);
        const [, second] = run('train', '--data', data, '--out', output, '--epochs', '1', '--resume', path.join(output, 'last.pt'));
        assert.equal(second.epoch, 2);
        const [report] = run('evaluate', '--data', data, '--checkpoint', checkpoint);
        assert.equal(report.split, 'test');
        assert.equal(report.examples, split.test);
        assert.ok(Number.isFinite(report.perplexity));

        model = new LocalModel({ python, checkpoint });
        await model.ready;
        const reply = model.ask('합성 질문 새 예시');
        await assert.rejects(model.ask('또 다른 합성 질문'), { code: 'BUSY' });
        assert.equal(await reply, '반가워요!');
        assert.equal(await model.ask('합성 질문 다시'), '반가워요!');
    } finally {
        model?.close();
        rmSync(fixture, { recursive: true, force: true });
    }
});
