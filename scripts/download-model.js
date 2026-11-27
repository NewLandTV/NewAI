const { spawnSync } = require('node:child_process');
const { existsSync, mkdtempSync, mkdirSync, readFileSync, renameSync, rmSync } = require('node:fs');
const path = require('node:path');

function download(runId) {
    if (!/^\d+$/.test(runId || '')) throw new Error('사용법: npm run model:download -- <GitHub Actions 실행 번호>');
    const parent = path.resolve('checkpoints');
    const output = path.join(parent, 'daily-final');
    if (existsSync(output)) throw new Error('checkpoints/daily-final이 이미 있어요. 기존 파일을 옮긴 뒤 다시 시도해 주세요.');
    mkdirSync(parent, { recursive: true });
    const staging = mkdtempSync(path.join(parent, '.daily-download-'));
    try {
        const result = spawnSync('gh', ['run', 'download', runId, '--repo', 'NewLandTV/NewAI',
            '--name', 'newai-daily-model', '--dir', staging], { stdio: 'inherit' });
        if (result.error?.code === 'ENOENT') throw new Error('GitHub CLI(gh)를 먼저 설치하고 로그인해 주세요.');
        if (result.error || result.status !== 0) throw new Error('다운로드하지 못했어요. GitHub CLI 로그인과 성공한 실행 번호를 확인해 주세요.');
        const model = JSON.parse(readFileSync(path.join(staging, 'model.json'), 'utf8'));
        if (model.version !== 1 || !Array.isArray(model.prototypes) || typeof model.responses?.fallback !== 'string') {
            throw new Error('지원하지 않는 모델 파일이에요.');
        }
        if (existsSync(output)) throw new Error('모델 폴더가 이미 생겨서 설치를 중단했어요.');
        renameSync(staging, output);
        console.log('체크포인트를 checkpoints/daily-final에 내려받았어요.');
    } finally {
        rmSync(staging, { recursive: true, force: true });
    }
}

try {
    download(process.argv[2]);
} catch (error) {
    console.error(error.message);
    process.exitCode = 1;
}
