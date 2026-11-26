# NewAI — 로컬 챗봇 프로토타입

[2023년 시연 영상: 디스코드 챗봇 제작 (오늘의 개발노트 p.340)](https://www.youtube.com/watch?v=0cbJv22SdaA)

2023년판과 달리 AI 응답은 외부 API나 API 키 없이 내 컴퓨터의 체크포인트에서 생성합니다. PyTorch로 구현한 소형 문자 단위 Transformer를 사전 학습 가중치 없이 처음부터 학습합니다. 디스코드 연결에는 인터넷과 디스코드 봇 토큰이 필요하지만, 모델 학습·평가·직접 질문은 설치 후 오프라인에서 할 수 있습니다.

## 1. 준비

Node.js 22.12 이상, Python 3.11~3.13을 사용하세요. Python 가상 환경을 만든 후 활성화합니다.

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

서버 채널에서 `@봇이름 질문`으로 호출하세요. 봇을 직접 멘션한 메시지에만 답하며 대화 기록을 저장하지 않습니다. `npm start`는 활성화된 Python 환경을 사용합니다. 다른 Python 실행 파일을 쓰려면 `NEWAI_PYTHON`에 그 경로를, 다른 체크포인트를 쓰려면 `NEWAI_CHECKPOINT`를 설정하세요.

## 개발 점검

Python 환경을 활성화한 상태에서 `npm test`를 실행하면, 합성 데이터로 학습·재개·테스트 평가·로컬 응답을 점검합니다. 디스코드 접속이나 실제 답변 품질을 검사하는 테스트는 아닙니다.
