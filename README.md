# NewAI — 로컬 챗봇 프로토타입

[2023년 시연 영상: 디스코드 챗봇 제작 (오늘의 개발노트 p.340)](https://www.youtube.com/watch?v=0cbJv22SdaA)

NewAI는 내 컴퓨터의 모델로 답하는 디스코드 봇이에요. 설치와 모델 다운로드가 끝나면 텍스트 생성·음성 인식·음성 합성에 외부 AI 서비스나 키가 필요 없어요. 디스코드에 접속할 때는 인터넷과 봇 토큰이 필요해요.

**기본값은 [일상 대화 검색 모델](docs/daily-model.md) 그대로예요.** 새로운 문맥 생성과 음성은 아래처럼 직접 켤 수 있어요. [출처·평가·학습 재현 보고서](docs/context-voice.md)에는 두 모델의 실제 응답과 실패 사례를 함께 실었어요.

## 새 문맥·음성 모드 실행

Node.js 22.12 이상과 Python 3.11을 준비하고 저장소 폴더에서 실행하세요. 먼저 텍스트만 확인할 수 있어요. 이 모드에는 PyTorch가 필요 없어요.

```sh
npm ci
python model/fetch_runtime.py context
node model/gguf-context.mjs ask "달에 있는 빵집 이름을 세 개 지어 줘요."
```

음성도 쓰려면 Python 가상 환경을 만들고, 아래 [준비](#1-준비)처럼 활성화한 다음 실행하세요.

```sh
python -m pip install -r model/requirements-voice.txt
python model/fetch_runtime.py voice
```

모델은 기본적으로 홈 폴더의 `.cache/newai`에 받아요(문맥 모델 약 1.1 GB, 음성 모델 합계 약 260 MB). 다른 곳을 쓰려면 다운로드와 실행 전에 `NEWAI_MODEL_ROOT`를 같은 절대 경로로 설정하세요. 모델이나 녹음은 Git 저장소에 포함하지 않아요.

아래 [디스코드 실행](#3-디스코드-실행)대로 로컬에서 토큰을 설정한 뒤, 다음 중 사용하는 터미널에 맞는 명령을 실행하세요. 텍스트만 쓰려면 `NEWAI_VOICE`를 설정하는 줄은 빼세요.

```powershell
# Windows PowerShell
$env:NEWAI_ENGINE = 'context'
$env:NEWAI_VOICE = '1'
npm start
```

```sh
# macOS/Linux
export NEWAI_ENGINE=context
export NEWAI_VOICE=1
npm start
```

- **텍스트:** 채널에서 실제 봇을 선택해 멘션하고 `내 화분 이름은 초롱이에요`, 이어서 `화분 이름이 뭐였죠?`를 각각 보내세요. 같은 사용자·채널의 최근 문답 최대 3쌍을 메모리에만 유지해요. 15분 동안 쓰지 않거나 `@NewAI 대화 초기화`를 보내면 지워요. 디스코드에 보낸 메시지 자체는 삭제하지 않아요.
- **음성:** 테스트 서버에서 봇에 **연결·말하기** 권한을 주고 일반 음성 채널에 들어가세요. 텍스트 채널에서 `@NewAI 음성 시작`을 보내면 봇이 먼저 듣고, `@NewAI 음성 시작 인사`를 보내면 합성 음성으로 먼저 인사해요. 답변은 채널의 다른 사람에게도 들려요. 처음에는 혼자 있거나 모두가 동의한 테스트 채널에서 짧게 말해 보세요.
- **종료:** `@NewAI 음성 종료`를 보내거나 음성 채널에서 나오세요. 봇은 한 번에 세션 하나에서 시작한 사람의 음성만 처리하고, 녹음과 음성 대화 내용을 파일에 남기거나 학습에 쓰지 않아요. 듣기와 말하기는 번갈아 진행해요. 실제 디스코드 통화는 아래 보고서의 점검 순서로 확인해 주세요.

## 일상 대화 모델 받기

이 프로젝트의 **파일 탭**에서 `newai-daily-model.zip`을 다운로드하거나, GitHub의 PR에서 성공한 **NewAI daily model** 실행을 열고 **Artifacts → newai-daily-model**을 받으세요. 압축 파일 내용은 `checkpoints/daily-final/`에 풉니다. GitHub 아티팩트는 로그인 후 30일간 받을 수 있습니다. CLI를 사용한다면 실행 주소 `/actions/runs/` 뒤의 숫자를 복사해 넣으세요.

```sh
npm run model:download -- <실행_번호>
python model/daily_model.py ask "안녕!"
```

다운로드 명령은 GitHub CLI에 로그인한 상태에서 사용하고, 기존 모델 폴더는 덮어쓰지 않습니다. 아티팩트가 만료됐으면 [로컬 생성 방법](docs/daily-model.md#재현)으로 다시 만들 수 있습니다. 디스코드에서는 `npm ci` 후 `DISCORD_BOT_TOKEN`을 설정하고 `npm start`를 실행하세요. 이전 실험용 `NEWAI_CHECKPOINT`를 설정했다면 해제하거나 `checkpoints/daily-final/model.json`으로 바꿔야 새 모델이 사용됩니다. 서버 텍스트 채널에서 자동완성으로 실제 봇을 선택해 **`@NewAI 안녕!`**을 보내세요.

## 1. 준비

Node.js 22.12 이상, Python 3.11~3.13을 사용하세요. 아래 가상 환경과 PyTorch 설치는 생성형 비교 실험을 실행할 때만 필요합니다.

```sh
npm ci
python -m venv .venv
```

- Windows PowerShell: `& .\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```sh
python -m pip install -r model/requirements.txt
```

Linux/Windows에서 CPU 전용 패키지를 설치하려면 위 설치 명령에 `--index-url https://download.pytorch.org/whl/cpu`를 추가하세요. 패키지 다운로드에는 인터넷이 필요합니다.

## 2. 학습 데이터와 체크포인트

학습에 사용한 [UA-N1-Queen 데이터](https://github.com/NewLandTV/UA-N1-Queen/tree/5a653644a2452583a16a29d766e01e1d20a5d180/dataset)의 버전은 `5a65364`이며, 원본 저장소의 [라이선스는 Apache 2.0](https://github.com/NewLandTV/UA-N1-Queen/blob/5a653644a2452583a16a29d766e01e1d20a5d180/LICENSE)입니다. 원본 `train_part_0.jsonl`부터 `train_part_7.jsonl`까지는 이 저장소에 포함하지 않았습니다. 아래 명령으로 `dataset/`에 내려받을 수 있으며, 이미 있는 파일과 내용이 다르면 덮어쓰지 않습니다.

```sh
python model/fetch_dataset.py
```

원본은 다음 예시의 첫 줄처럼 `user`와 `assistant`를 사용합니다. 나머지 형식도 읽을 수 있습니다.

```jsonl
{"user":"안녕?", "assistant":"안녕하세요!"}
{"question":"안녕?", "answer":"안녕하세요!"}
{"prompt":"인사해 줘", "response":"반가워요."}
{"instruction":"문장을 고쳐 줘", "input":"안녕 하세요", "output":"안녕하세요."}
{"messages":[{"role":"user","content":"반가워"},{"role":"assistant","content":"저도 반가워요."}]}
```

`messages`는 순서대로 인접한 사용자–어시스턴트 쌍으로 나눕니다. 대화에 맞지 않는 행은 건너뛴 개수를 알리고, 깨진 JSON은 파일명과 줄 번호를 표시합니다. 같은 질문은 대소문자와 공백을 정규화한 뒤 한 분할에만 두며, 목표 비율은 학습 80%, 검증 10%, 테스트 10%입니다. 각 분할에 서로 다른 질문이 충분히 있어야 합니다.

```sh
# 파일당 앞의 2,000줄만 읽는 기본 프로토타입 학습
python model/newai.py train --epochs 5

# 같은 데이터 설정으로 last.pt부터 2회 더 학습
python model/newai.py train --resume checkpoints/last.pt --epochs 2

# 검증 손실이 가장 낮은 best.pt를 미사용 테스트 분할로 평가
python model/newai.py evaluate

# 디스코드 없이 로컬 응답 확인
python model/newai.py ask "안녕하세요!"
```

전체 행을 읽으려면 새 학습에서 `--limit 0 --out checkpoints/full`을 지정하세요. 전체 데이터를 메모리에 올리므로, 데이터가 크면 작은 범위로 먼저 확인하세요. 별도 출력 폴더를 썼다면 평가·질문의 `--checkpoint`와 디스코드 실행의 `NEWAI_CHECKPOINT`에도 해당 경로를 지정하세요. 학습 때 다른 데이터 폴더를 사용했다면 평가에도 같은 `--data`를 지정하세요.

UA-N1-Queen 학습을 같은 설정으로 재현하려면 아래 명령을 사용하세요.

```sh
python model/newai.py train --out checkpoints/ua-n1-5a65364 --limit 0 --context 96 --epochs 7
python model/newai.py train --out checkpoints/ua-n1-5a65364 --resume checkpoints/ua-n1-5a65364/last.pt --epochs 7
python model/newai.py evaluate --checkpoint checkpoints/ua-n1-5a65364/best.pt
python model/newai.py ask --checkpoint checkpoints/ua-n1-5a65364/best.pt "안녕하세요!"
```

평가는 답변 토큰 손실·퍼플렉서티와 알려진 문자에 대한 다음 문자 정확도를 출력합니다. 이 지표는 자연스러움이나 사실 정확도를 보장하지 않습니다. 맥락이 짧고, 매 질문을 독립적으로 처리하며, 학습하지 않은 주제에 잘못된 답을 만들 수 있습니다. 로컬 체크포인트는 직접 만든 신뢰할 수 있는 파일만 사용하세요.

### UA-N1-Queen 평가 결과

원본 811행 중 질문이 비어 있는 1행과 중복 질문·답변 쌍 2행을 제외했습니다. 나머지 808쌍은 학습 639쌍, 검증 98쌍, 테스트 71쌍으로 나눴습니다. 7회차 체크포인트에서 학습을 재개해 14회까지 진행했고, 검증 손실이 가장 낮은 10회차를 골랐습니다.

| 체크포인트 | 테스트 손실 | 테스트 퍼플렉서티 | 알려진 문자 정확도 |
| --- | ---: | ---: | ---: |
| `best.pt` (10회차) | 3.283 | 26.65 | 34.7% |

위 정확도는 정답의 앞부분을 보여 준 상태에서 다음 문자를 맞힌 비율이며, 질문에 맞는 답을 생성한 비율이 아닙니다. 직접 질문에서도 한국어가 출력됐지만, 문장이 깨지거나 질문과 관계없는 답이 나왔습니다. 현재 모델은 로컬 학습 흐름을 확인하기 위한 실험용입니다.

체크포인트 묶음에서 `best.pt`와 `last.pt`를 `checkpoints/ua-n1-5a65364/`에 풀면 위 직접 질문·평가 명령으로 확인할 수 있습니다. 디스코드에서 사용하려면 `NEWAI_CHECKPOINT`를 `checkpoints/ua-n1-5a65364/best.pt`로 지정하세요.

### 일상 대화 확장 실험

[Youngsook Song의 Chatbot_data_for_Korean v1.0](https://github.com/songys/Chatbot_data/tree/4cf20d13fc46f5037fd1c531cd566e2dd9f72974) ([MIT](https://github.com/songys/Chatbot_data/blob/4cf20d13fc46f5037fd1c531cd566e2dd9f72974/LICENSE))에서 일상 라벨(`0`)만 선택했습니다. 이별과 사랑 라벨은 제외했습니다. 원본과 추가 데이터에서 길이 제한, 빈 텍스트, 한글 없는 문답, 연락처·링크 패턴, 중복 및 짧은 되풀이 답변을 걸러냅니다. 추가 데이터에서는 기존 질문과 겹치는 질문을 빼고, 같은 답변이 최대 12번만 남도록 제한합니다. 이 규칙은 사람이 문맥·개인정보·안전성을 완전히 검수한 것을 대신하지 않습니다.

`dataset/daily/`에 UA-N1-Queen과 같은 `user`/`assistant` 구조의 JSONL 8개와 필터 집계 `preparation.json`을 생성합니다. 기존 UA-N1-Queen 파일이나 체크포인트는 덮어쓰지 않습니다. 이 실험에서는 UA-N1-Queen 800쌍과 추가 일상 대화 5,202쌍, 총 6,002쌍이 남았습니다.

```sh
python model/fetch_dataset.py
python model/prepare_daily.py
python model/newai.py train --data dataset/daily --out checkpoints/daily-v2 --limit 0 --context 96 --group-normalized --epochs 10
python model/newai.py evaluate --data dataset/daily --checkpoint checkpoints/daily-v2/best.pt
python model/compare_daily.py
```

확장 실험의 `--group-normalized`는 공백이나 문장부호만 다른 질문을 같은 분할에 둡니다. `compare_daily.py`는 기존 모델과 확장 모델에 같은 질문을 주고 응답을 나란히 보여 줍니다. 공백·문장부호 등을 제외한 질문이 양쪽 중 어느 한 모델의 학습·검증 데이터와 완전히 일치하면 해당 질문은 비교에서 건너뜁니다. 각 모델의 자체 테스트 퍼플렉서티는 분할과 문자 사전이 달라 직접 비교하지 않습니다. 모델 구조는 두 실험 모두 마스킹된 자기 어텐션을 쓰는 문자 단위 Transformer이며, 이전 대화를 함께 넘기지 않으므로 여러 턴에 걸친 맥락은 여전히 처리할 수 없습니다.

직접 응답은 (1) 인사에 인사, 감사에 감사 수용처럼 사용자의 말에 맞게 이어지는지, (2) 사용자가 말한 감정이나 상황을 엉뚱하게 바꾸지 않는지, (3) 추천 요청에 실제로 할 수 있는 행동을 제안하는지로 확인합니다. 문장처럼 보이기만 하거나 질문을 되풀이하는 답은 통과로 보지 않습니다. 12개로 한정한 탐색 점검이며 사람의 판단이 필요한 서비스 품질 검증을 대신하지 않습니다.

## 3. 디스코드 실행

디스코드 개발자 포털에서 봇을 만든 후, 테스트 서버에 초대하고 채널 보기·메시지 보내기·메시지 기록 읽기 권한을 주세요. 토큰은 코드나 채팅에 넣지 말고 로컬 터미널의 `DISCORD_BOT_TOKEN` 환경 변수로만 설정하세요.

```powershell
# Windows PowerShell
$env:DISCORD_BOT_TOKEN = '<내 봇 토큰>'
npm start
```

```sh
# macOS/Linux
export DISCORD_BOT_TOKEN='<내 봇 토큰>'
npm start
```

서버 채널에서 `@봇이름 질문`으로 호출하세요. 봇을 직접 멘션한 메시지에만 답합니다. `npm start`의 기본 체크포인트는 `checkpoints/daily-final/model.json`입니다. `NEWAI_ENGINE=daily`로 돌아가면 이 기본 동작을 쓸 수 있습니다. 다른 Python 실행 파일을 쓰려면 `NEWAI_PYTHON`에 그 경로를, 기존 문자 단위 비교용 `.pt`를 사용하려면 `NEWAI_CHECKPOINT`에 그 파일 경로를 지정하세요.

## 개발 점검

PyTorch가 설치된 Python 환경에서 `npm test`를 실행하면 기존 모델 학습·재개와 문맥 분리, 음성 패킷 처리를 점검합니다. 모델 다운로드 없이 문맥·음성 기본 점검만 실행하려면 `node --test test/context-voice.test.js`를 쓰세요. 음성 왕복을 확인하는 방법은 [보고서](docs/context-voice.md#음성-검증)에 있습니다.
