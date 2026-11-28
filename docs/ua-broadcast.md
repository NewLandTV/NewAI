# 유아(U.A.)의 캐릭터 학습과 방송 리허설

## 왜 유아를 추가했나요?

[「방송하는 AI 버튜버 만들기 (Ai project)」 영상 설명](https://www.youtube.com/watch?v=LApZVb76Qj4)은 직접 AI 버튜버를 만드는 시도를 채널의 새 출발로 기록하려는 취지를 밝힙니다. [Ai-project README](https://github.com/NewLandTV/Ai-project/blob/e981415aa4bf604d736b6a799b27f12583e7355a/README.md)는 목표를 ‘누구나 자신이 원하는 AI 버튜버를 만들어 실제 방송까지 하기’로 설명합니다. 유아를 NewAI에 추가한 이유는 디스코드 응답을 넘어, 같은 캐릭터로 시청자를 맞이하고 운영자가 답변을 공개하기 전에 확인할 수 있는 리허설을 만들기 위해서입니다.

- **캐릭터:** `develop`의 [모델 개요](../ua-activity/newai-model-overview.pdf), [일러스트](../ua-activity/ua-character-concept.png), [사용자가 제공한 설정 초안](../ua-activity/ua-character-profile.md)을 참고했습니다. 텍스트 설정의 계보는 [UA-N1-Queen 원본](https://github.com/NewLandTV/UA-N1-Queen/blob/main/prompt.txt)과 [Ai-project의 유아 프롬프트](https://github.com/NewLandTV/Ai-project/blob/e981415aa4bf604d736b6a799b27f12583e7355a/prompt.txt)입니다. 빨간 리본의 용도는 비밀로 남기고, 게임 속 점수를 실제 결제나 후원 기능으로 연결하지 않았습니다. PDF·이미지를 모델 데이터로 넣은 것은 아닙니다.
- **후속 연구 방향:** [UA-N2-Queen README](https://github.com/NewLandTV/UA-N2-Queen/blob/7c9371ca005a38cabf42c133e91ce992cd681a8f/README.md)는 N1 시연의 문맥·유창성 한계, 더 좋은 데이터, 로컬에서 모델을 직접 설계하고 학습하려는 배경을 설명합니다. 이 실험은 그 문제의식을 참고하지만 UA-N2-Queen의 가중치·훈련·성적은 포함하지 않습니다. UA-N1-Queen 시연이나 사용자가 별도로 수행한 코드 작성 테스트와도 정량 비교하지 않았습니다.
- **연동:** [Ai-project의 커맨드 로더](https://github.com/NewLandTV/Ai-project/blob/e981415aa4bf604d736b6a799b27f12583e7355a/src/commands/__init__.py)와 [답변 순회 방식](https://github.com/NewLandTV/Ai-project/blob/e981415aa4bf604d736b6a799b27f12583e7355a/src/chatbot.py)을 사용합니다. 외부 설치 도구가 필요한 파일만 추가하고 기존 코드를 유지합니다. 리허설과 공개 승인 릴레이를 분리했습니다. [설치·운영 방법](../integrations/ai-project/README.md).

## 무엇을 어떻게 학습했나요?

| 실험 | 방법과 목적 |
| --- | --- |
| Qwen2.5 학습 | [Qwen2.5-0.5B-Instruct `7ae5576`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775), Apache 2.0. CPU에서 고정 int8 본체의 은닉 상태를 한 번 뽑은 뒤, 마지막 어휘 점수 층에만 [LoRA](https://arxiv.org/abs/2106.09685) 방식의 델타를 학습했습니다. 어텐션이나 임베딩은 학습하지 않았습니다. |
| 선택형 유아 리허설 | [기존 Qwen3-1.7B Q4_K_M `d7f544e`](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/tree/d7f544eead698dbd1f15126ef60b45a1e1933222), [Apache 2.0](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/blob/90862c4b9d2787eaed51d12237eafdfe7c5f6077/LICENSE). 이전 `context`와 가중치와 생성 설정이 같고 시스템 프롬프트만 다릅니다. 학습 가중치를 얹은 버전은 아닙니다. |

[학습 대화 36개](../model/ua_training.jsonl), [검증 대화 6개](../model/ua_validation.jsonl), [생성 평가 20개](../model/ua_eval.json)를 별도로 작성했습니다. 질문은 정규화 후 동일 문장이 없음을 확인했습니다. 설정이 같으므로 주제나 정답의 일부는 겹칩니다. 채팅 로그나 음성 녹음은 사용하지 않았습니다.

동결된 모델의 상위 128개 어휘 후보와 정답 토큰을 모아, 정답 문장을 한 토큰씩 이어 쓸 때의 손실을 줄였습니다. Rank 4, alpha 8, 초기화 시드 19, 학습 변수 611,328개, 최대 8 epoch로 설정했습니다. 검증 후보 손실이 가장 작은 가중치를 남겼습니다. 이 손실은 전체 어휘 perplexity가 아니고, 실제로 문장을 생성했을 때의 유창성이나 정확도도 아닙니다.

## 비교 방법

20문항에 같은 질문·이전 문답·판정 조건을 넣었습니다. 캐릭터 설정·말투 10문항, 계산·공감·문맥 6문항, 실제 기능과 세계관 경계 4문항입니다. Qwen2.5는 동일 프롬프트와 greedy 생성에서 어댑터를 끈 상태와 켠 상태를 비교했습니다. Qwen3 두 모드는 동일한 샘플링 설정과 시드 19를 썼습니다. 두 계열의 가중치·크기·양자화 방식·생성 설정은 서로 다르므로 계열 간 점수 차이를 LoRA 효과로 해석할 수 없습니다.

통과는 지정된 표현, 금지 표현, 답변 길이를 함께 확인하는 자동 규칙입니다. 예를 들어 정정 질문은 `토요일`을 포함하고 `목요일`을 포함하지 않아야 합니다. 키워드가 맞아도 틀린 설명을 붙일 수 있고, 자연스러운 동의어를 놓칠 수도 있습니다. 독립 심사자 평가, 공개 벤치마크, 영상·채팅별 방송 품질, 범용 자연어 이해나 코드 작성 실력을 측정한 점수가 아닙니다.

### 실제 생성 결과

[문항별 응답 원문과 학습 기록](ua-eval/)을 함께 남겼습니다. 응답 원문에는 모델의 오답과 지어낸 정보도 들어 있습니다.

| 모드 | 캐릭터 (10) | 과제·문맥 (6) | 기능 경계 (4) | 전체 (20) |
| --- | ---: | ---: | ---: | ---: |
| 기존 Qwen3 `context` ([PR #2](https://github.com/NewLandTV/NewAI/pull/2)) | 0 | 4 | 0 | 4 |
| 같은 Qwen3 + 유아 설정 (가중치 변경 없음) | 2 | 5 | 0 | 7 |
| Qwen2.5 per-tensor int8, 같은 유아 프롬프트, 학습 전 | 0 | 2 | 0 | 2 |
| Qwen2.5 per-tensor int8, 출력층 LoRA 학습 후 | 0 | 0 | 0 | 0 |
| Qwen2.5 per-channel int8, 같은 유아 프롬프트, 학습 전 | 0 | 0 | 1 | 1 |
| Qwen2.5 per-channel int8, 출력층 LoRA 학습 후 | 0 | 0 | 0 | 0 |

- **Qwen3:** 설정을 적용하면 **4/20(20%) → 7/20(35%)**로 3문항 늘었습니다. 키의 두 기준과 65,536의 결제 여부에 답했고, `9 × 8 = 72`를 맞혔습니다. 다만 키 답변에 무관한 운영 권한 얘기를 덧붙였고, 자기소개는 고정 시드 실행에서 비었습니다. 반대로 “뭐를 고쳐 줄 수 있어요?”처럼 의미상 되물었지만 지정 문구가 아니라 실패한 사례도 있습니다. 기능 경계는 양쪽 모두 **0/4**로, 자동 방송 대신 답변별 승인을 유지해야 합니다.
- **Qwen2.5:** 첫 시도의 검증 후보 손실은 **5.1082 → 4.2976**(8 epoch 중 2번째), per-channel 재시도에서는 **5.1192 → 4.5280**(8 epoch 중 3번째)이었습니다. 실제 생성을 고정 20문항으로 확인하니 각각 **2/20 → 0/20**, **1/20 → 0/20**으로 나빠졌습니다. 첫 어댑터는 실제 제어가 불가능한데도 “OSB에서 실제 버튼을 눌렀어요.”라고 답했습니다. 정답 토큰을 넣어 측정한 손실은 이미 생성한 토큰을 토대로 이어 쓰는 상황과 다릅니다. per-channel 재시도는 첫 실패를 보고 추가한 탐색 실험이며 방송 모드에는 두 어댑터 모두 적용하지 않았습니다.
- **정밀도 진단:** 원본 BF16에서 같은 계산·방송 인사 질문을 포함한 3개 예를 점검했을 때 일부 문장이 int8보다 자연스러웠지만, 지정 길이·캐릭터 조건까지 안정적으로 맞춘 것은 아닙니다. 이 확인은 별도 정밀도 점검이며 20문항 비교나 LoRA 성능에 합산하지 않았습니다.

## 재현

```sh
# 모델 다운로드는 처음 한 번만 필요
python -m pip install -r model/requirements-ua.txt
python model/fetch_runtime.py qwen
python model/train_ua.py --out checkpoints/ua-head-v1
python model/evaluate_ua.py --adapter checkpoints/ua-head-v1/adapter.safetensors

# 같은 출력층 학습을 채널별 int8 가중치에서 재검증
python model/train_ua.py --out checkpoints/ua-head-per-channel --quantization per-channel
python model/evaluate_ua.py --adapter checkpoints/ua-head-per-channel/adapter.safetensors --quantization per-channel

# 같은 20문항에서 이전 Qwen3와 캐릭터 설정을 넣은 Qwen3 비교
npm ci
python model/fetch_runtime.py context
python model/evaluate_ua.py --prior
python model/evaluate_ua.py --ua-qwen3
```

모델과 학습된 가중치는 저장소에 올리지 않았습니다. `train_ua.py`는 기존 학습 폴더를 덮어쓰지 않고, 데이터 체크섬과 손실·선택 시점을 `training.json`에 남깁니다. 학습·평가 결과는 하드웨어와 실행 환경에 따라 달라질 수 있습니다.

## 방송 연동 검증 범위

- Ai-project 고정 버전의 실제 커맨드 로더를 복제한 시험용 폴더에서 `/run newai_ua`로 모듈을 열고, 실제 모델이 생성한 자기소개를 띄운 뒤 `/exit`으로 돌아오는 것을 확인했습니다. Ai-project 자체에는 변경을 올리지 않았습니다.
- 실제 로컬 Qwen3를 연결한 브리지에서 “유아는 어떤 AI야?”에 “유아는 백발 단발, 회색 눈, 빨간 리본을 쓴 마인크래프트 플레이어 AI예요.”로 시작하는 답을 받았습니다. 2 vCPU 환경에서 다른 평가도 동시에 실행하던 중의 단건 확인으로, 준비 후 답변 생성에 약 48초가 걸렸습니다. 즉각적인 실시간 응답을 보장하지 않습니다.
- 오프라인 테스트는 파일 덮어쓰기 방지, 외부 모델과의 메시지 규격, 라이브 문맥 미공유, 승인 전 무출력, 승인 후 자막·음성 순서를 확인합니다.
- 모의 OBS에는 승인 후 `Answer Text`를 갱신했고, 기존 Canora `seoyeon`이 합성한 오디오는 가상 재생 장치에서 확인했습니다. YouTube의 실제 라이브, OBS 연결, 실방송 소음 속 음질 및 오디오 라우팅을 검증한 결과는 아닙니다. 첫 사용은 비공개 리허설을 권장합니다.
