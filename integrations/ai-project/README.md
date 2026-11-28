# 유아(U.A.) · Ai-project 연동

이 모듈은 NewAI의 로컬 유아 모델을 Ai-project에서 사용하기 위한 선택형 연동이에요. Ai-project의 기본 모델과 Studio 설정, 기존 방송 코드, `commands.ini`는 바꾸지 않아요.

## 1. 비공개 리허설

1. NewAI에서 `npm ci`와 `python model/fetch_runtime.py context`를 실행해 Qwen3를 준비하세요. Node.js 22.12 이상과 Python 3.11을 사용하세요.
2. NewAI 폴더에서 `python integrations/ai-project/install.py "<Ai-project 폴더>"`를 실행하세요. `src/newai_ua_bridge.py`, `src/ua_broadcast.py`, `src/commands/newai_ua/__init__.py` 세 파일만 추가해요. 내용이 다른 파일이 이미 있으면 덮어쓰지 않아요.
3. 환경 변수 `NEWAI_ROOT`에 NewAI 폴더의 절대 경로를 설정하세요. Windows PowerShell에서는 `$env:NEWAI_ROOT = 'C:\경로\NewAI'`, macOS/Linux에서는 `export NEWAI_ROOT="/경로/NewAI"` 형식이에요. Node.js가 실행 경로에 없으면 `NEWAI_NODE`에 실행 파일의 절대 경로를 지정하세요.
4. 같은 터미널에서 Ai-project CLI를 열고 `/run newai_ua`를 입력하세요. 또는 Ai-project 폴더에서 `python src/ua_broadcast.py`를 실행하세요. `/reset`은 리허설 문맥을 지우고 `/exit`은 원래 화면으로 돌아가거나 릴레이를 닫아요.

리허설은 최근 문답 최대 3쌍을 메모리에 유지하고, 15분 동안 쉬었다 돌아오면 이전 문답을 지워요. 답변이나 음성을 파일에 저장하지 않아요.

## 2. OBS와 선택형 음성

Ai-project 폴더에서 다음 중 필요한 모드만 켜세요.

```sh
# 운영자가 직접 질문하고 OBS 자막 확인
python src/ua_broadcast.py --obs
# YouTube 라이브의 !ua 질문을 받아 OBS 자막과 로컬 음성으로 리허설
python src/ua_broadcast.py --video-id <영상의_11자_ID> --obs --voice
```

- `--obs`는 Ai-project의 `obs_host`, `obs_port`, `obs_password` 설정을 사용해 **`Answer Text`** 자막 소스의 글자만 바꿔요. 해당 이름의 텍스트 소스를 OBS에 먼저 추가하세요. 방송을 시작·종료하거나 장면을 전환하지 않아요.
- `--voice`는 운영자 컴퓨터에서 Canora `seoyeon` 합성 음성을 재생해요. NewAI 쪽에서 `python -m pip install -r model/requirements-voice.txt`, `python model/fetch_runtime.py voice`로 준비하고 `ffplay`를 실행 경로에 넣으세요. 별도 가상 환경을 사용했다면 `NEWAI_PYTHON`에 그 Python 실행 파일의 절대 경로를 설정하세요. OBS로 보내려면 운영자가 녹음·송출 대상 오디오를 별도로 지정해야 해요.
- 릴레이는 초안을 먼저 콘솔에 표시해요. YouTube 질문, OBS 자막 또는 합성 음성을 켠 상태에서는 운영자가 **`공개`**라고 입력한 답만 내보내요. 다른 입력은 건너뛰어요. YouTube 채팅에 답글을 쓰지는 않아요.
- 라이브는 시청자의 일반 메시지나 `/run` 명령을 실행하지 않아요. `!ua 질문`만 받아 최근 수신 묶음의 마지막 적격 질문 하나를 제시하고, 각 검토 뒤 약 8초 동안 들어온 질문은 건너뛰어요. 사용자 간 대화 문맥은 이어가지 않아요.

먼저 비공개 화면에서 자막·음성 각각의 승인과 거절을 확인하세요. 기존 Ai-project의 자동 방송 모드와 이 릴레이를 동시에 같은 자막에 연결하지 마세요. 실제 방송에서 사람이 검토하는 절차를 대신하지 않아요.

## 3. 후속 모델로 교체할 때

`NewAIUABridge.get_response(message)`는 Ai-project의 기존 응답 객체와 같은 문자열 순회 방식을 제공해요. 현재는 답변을 완성한 뒤 한 덩어리로 넘겨요. 후속 연동 코드는 `NewAIUABridge(worker_command=[실행_파일, 스크립트_경로])`로 호환되는 로컬 모델을 지정할 수 있어요. 모델은 준비 시 `{"type":"ready"}`를 보내고, `prompt`, `history`, `voice`가 담긴 UTF-8 JSON 한 줄을 받을 때마다 `{"type":"answer","text":"…"}`를 한 줄 보내야 해요. 라이브에서는 교체 후에도 기존 자동 응답 루프에 직접 끼우지 말고, 운영자 승인이 있는 릴레이를 사용하세요. UA-N2-Queen의 실제 연결이나 초기화 방식은 이번 범위에 포함하지 않아요.
