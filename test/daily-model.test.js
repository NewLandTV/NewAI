const assert = require('node:assert/strict');
const { existsSync, readFileSync } = require('node:fs');
const { test } = require('node:test');
const { LocalModel } = require('../local-model');

const checkpoint = 'checkpoints/daily-final/model.json';

test('selected daily model gives reviewed replies through the Discord bridge', {
    skip: !existsSync(checkpoint), timeout: 20_000,
}, async () => {
    const model = new LocalModel({ checkpoint });
    const reviewed = JSON.parse(readFileSync('model/daily_dialogues.json', 'utf8'));
    try {
        await model.ready;
        assert.equal(await model.ask('안녕!'), reviewed.greeting.response);
        assert.equal(await model.ask('안녕하세요! 반가워요. 오늘 처음 인사해요.'), reviewed.greeting.response);
        assert.equal(await model.ask('이 두통약을 같이 먹어도 되나요?'), reviewed.fallback.response);
        assert.equal(await model.ask('점심으로 빨리 먹을 수 있는 게 뭐가 있죠?'), reviewed.food.response);
    } finally {
        model.close();
    }
});
