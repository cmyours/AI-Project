# Natural Language AI Assistant

**자연어 요청을 분석해 외부 기능을 실행하는 AI 어시스턴트**

기존 언어 모델을 QLoRA 방식으로 파인튜닝하고, 모델이 생성한 구조화된 JSON 결과를 프로그램의 기능 실행 로직과 연결한 개인 프로젝트입니다. 일반 대화와 기능 실행 요청을 구분하고, 외부 기능의 결과를 다시 자연어 응답으로 전달하도록 설계했습니다.

## 주요 기능

| 기능 | 설명 | Action |
| --- | --- | --- |
| 일반 대화 | 별도 기능 실행 없이 응답 생성 | `NONE` |
| 날씨 조회 | 지역명 추출 후 Open-Meteo API 조회 | `GET_WEATHER` |
| 날짜·시간 확인 | 현재 시간, 날짜, 요일 조회 | `GET_CURRENT_TIME` |
| 타이머 | 자연어로 지정한 시간 길이 해석 및 타이머 설정 | `SET_TIMER` |
| 일정 추가 | 일정 정보 추출 및 로컬 JSON 저장 | `ADD_SCHEDULE` |
| 일정 조회 | 날짜 범위에 맞춰 저장된 일정 조회 | `LIST_SCHEDULE` |

텍스트 입력·출력을 기본으로 사용하며, ElevenLabs STT/TTS와 Edge TTS를 이용한 음성 입출력 코드도 포함되어 있습니다. 음성 기능은 실행 환경과 API 설정에 따라 별도 설정이 필요합니다.

## 시스템 구성

```text
사용자 입력 (텍스트 / 음성)
          |
          v
파인튜닝된 언어 모델 (요청 분류 및 인자 추출)
          |
          v
   JSON 출력 및 검증
          |
     +----+----+
     |         |
   일반 대화   기능 실행 (날씨 / 시간 / 타이머 / 일정)
     |         |
     |     외부 API 또는 로컬 처리
     |         |
     +----+----+
          |
          v
  최종 자연어 응답 (텍스트 / 음성)
```

모델의 판단만으로 기능을 즉시 실행하지 않고, 프로그램에서 `action_trigger`와 `action_argument`를 확인·보정한 뒤 기능을 호출합니다. 예를 들어 날씨 질문에 지역이 없으면 API를 호출하지 않고 지역을 다시 묻도록 처리합니다.

모델 출력 데이터의 기본 형식은 다음과 같습니다.

```json
{
  "output": "서울 날씨 확인해볼게.",
  "action_required": true,
  "action_trigger": "GET_WEATHER",
  "action_argument": "서울"
}
```

위 JSON은 데이터 형식을 설명하기 위한 예시이며, 실제 생성 문장은 입력 및 모델 상태에 따라 달라질 수 있습니다.

## 사용 기술

- **언어 모델:** [Llama 3 Korean Bllossom 8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)
- **파인튜닝:** QLoRA, PEFT, Transformers, bitsandbytes, PyTorch
- **학습 데이터 생성·검증:** Python, Gemini API, Pydantic
- **외부 기능:** Open-Meteo API, Python 표준 라이브러리, JSON 파일 기반 일정 저장
- **음성 기능:** ElevenLabs STT/TTS, Edge TTS
- **개발 환경:** Google Colab (GPU)

## 학습 데이터와 모델 학습

Gemini API 등을 사용해 자연어 요청과 응답 데이터를 생성하고, 필수 필드·문장 길이·말투·중복 여부 및 일부 기능 실행 규칙을 검사하는 필터를 구현했습니다.

저장된 **Bllossom LoRA 이어학습 설정**은 다음과 같습니다.

| 항목 | 설정 |
| --- | --- |
| 베이스 모델 | `MLP-KTLim/llama-3-Korean-Bllossom-8B` |
| 학습 데이터 | 총 1,456개 (학습 1,384개 / 평가 72개) |
| 학습 방식 | 기존 LoRA 어댑터를 불러온 뒤 추가 학습 |
| 양자화 | 4-bit NF4, double quantization |
| LoRA | rank 8, alpha 16, dropout 0.05 |
| 적용 모듈 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| 이어학습 설정 | 1 epoch, learning rate `1e-4`, 최대 시퀀스 길이 512 |

`notebooks/Llama_Retraining.ipynb`는 **기존 LoRA 어댑터를 이어학습하는 코드**입니다. 베이스 모델을 처음부터 사전학습하거나, 최초 LoRA 어댑터를 생성하는 전체 과정을 담은 노트북은 아닙니다.

## 프로젝트 구조

```text
AI-Function-Calling-Assistant/
├── README.md
├── requirements.txt
├── .gitignore
├── aiproject.py                 # 모델 로드 및 어시스턴트 실행
├── make_dataset.py              # 데이터 생성 및 검증
├── notebooks/
│   └── Llama_Retraining.ipynb   # LoRA 이어학습
├── prompts/
│   ├── sys_instruction.txt
│   └── schedule_request_prompt.txt
├── dataset/
│   └── dataset.json             # 공개용 학습 데이터
└── model/
    └── blossom_LoRA/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

`prompts/`의 두 파일과 `dataset/dataset.json`이 있어야 데이터 생성 스크립트를 사용할 수 있습니다. 모델 실행에는 베이스 모델과 LoRA 어댑터가 필요합니다.

## 실행 방법

### 1. 준비

**CUDA를 지원하는 NVIDIA GPU 환경**을 권장합니다. 이 프로젝트는 4비트 양자화를 사용하므로 실행 환경에 맞는 PyTorch/CUDA/bitsandbytes 구성이 필요합니다. Google Colab GPU 런타임에서도 실행할 수 있습니다.

프로젝트 폴더에서 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

### 2. 모델 실행

`aiproject.py`의 `BASE_MODEL_ID`가 Bllossom 8B를 가리키는지 확인하고, LoRA 로딩 경로가 `model/blossom_LoRA`를 가리키도록 설정합니다.

텍스트 입출력으로 실행하려면 다음 설정을 사용합니다.

```python
INPUT_MODE = "keyboard"
OUTPUT_MODE = "console"
```

```bash
python aiproject.py
```

최초 실행 시 Hugging Face에서 베이스 모델을 받아야 하므로 네트워크 연결 및 충분한 저장 공간이 필요합니다. 이 저장소에는 **베이스 모델 전체 가중치가 아니라 LoRA 어댑터만** 포함됩니다.

### 3. 학습 데이터 추가 생성 (선택)

`make_dataset.py`는 Gemini API를 호출하므로 `GEMINI_API_KEY` 환경변수가 필요합니다.

```bash
python make_dataset.py
```

이 스크립트는 `prompts/`에서 프롬프트를 읽고 `dataset/dataset.json`에 결과를 누적하므로, 실행 전 기존 데이터 파일을 백업하는 것을 권장합니다. API 키를 소스코드에 직접 기록하거나 Git에 커밋하지 마세요.

### 4. LoRA 이어학습 (선택)

Google Colab에서 `notebooks/Llama_Retraining.ipynb`를 열고, `PROJECT_DIR`과 `CONTINUE_FROM_LORA_PATH`를 실제 프로젝트 위치 및 기존 어댑터 경로로 설정한 뒤 실행합니다. 이어학습에는 GPU와 원본 모델 접근이 필요합니다.

## 개발 과정에서 다룬 문제

1. **불완전한 기능 요청:** 날씨 조회에 지역 정보가 없을 때 임의로 답하거나 기능을 실행하지 않도록 요청 조건을 분리했습니다.
2. **기능 실행 판단 오류:** 일반 대화와 기능 실행 요청을 구분하기 위해 구조화된 데이터와 실행 단계 검증 로직을 사용했습니다.
3. **생성 데이터 품질:** 말투·형식·길이·중복 검사 등 자동 필터를 추가했습니다.

## 한계 및 개선 과제

- 학습 데이터의 형식 검증만으로 모든 의미적 오류를 찾을 수는 없습니다. 일부 일정 예시에는 날짜와 요일이 일치하지 않는 사례가 확인되었습니다.
- 기능별 정확도 및 응답 성능을 나타내는 정량 평가 결과는 이 저장소에 포함되어 있지 않습니다.
- 학습·실행 코드 일부는 Google Colab을 중심으로 작성되었으므로 환경에 맞는 경로 및 패키지 설정이 필요합니다.
- 일정 데이터는 로컬 JSON 파일에 저장되며, 별도의 사용자 인증이나 클라우드 일정 동기화는 구현되어 있지 않습니다.

## 모델 및 라이선스 안내

이 프로젝트의 LoRA 어댑터는 제3자가 공개한 **Llama 3 Korean Bllossom 8B**를 기반으로 합니다. 베이스 모델과 관련 구성 요소의 이용·재배포 조건은 원본 모델 페이지 및 적용 라이선스를 확인해야 합니다. 별도 라이선스 파일이 없는 경우 이 저장소의 소스코드에 대한 사용 권한이 자동으로 부여되는 것은 아닙니다.
