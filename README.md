# 😃 KoBERT 기반 한국어 감정 및 광고성 문장 분류 프로젝트

이 프로젝트는 SKT KoBERT 모델을 활용하여 한국어 문장의 감정(8가지)과 게시글의 광고성 여부를 분류하는 텍스트 분류 모델입니다.

* 감정 분류: 주어진 문장을 '우울한', '기쁜', '화가 나는', '슬픈', '편안한', '걱정스러운', '신이 난', '충만한' 중 하나로 예측합니다.
* 광고성 분류: 주어진 문장이 '광고'인지 '정상'인지 예측합니다.
* KoBERT(Huggingface 기반)를 사용하여 구현했습니다.
* 파이토치(PyTorch)와 transformers 라이브러리 기반으로 구성됐습니다.

---

## 🚀 실행 방법 (How to Run)

아래 실행 명령어에서 `--task` 옵션으로 분류 작업을 선택할 수 있습니다. (`emotion` 또는 `ad` 중 선택)

### 1. 모델 학습 (Train)

```bash
# 감정 분류 모델 학습
python main.py --task emotion --mode train

# 광고성 분류 모델 학습
python main.py --task ad --mode train
````

* 학습이 끝나면 각각의 task에 따라 모델 파일이 생성됩니다.

    * 감정 분류: `saved_model_emotion.pt`
    * 광고성 분류: `saved_model_ad.pt`

---

### 2. 모델 평가 (Eval)

```bash
# 감정 분류 모델 평가
python main.py --task emotion --mode eval

# 광고성 분류 모델 평가
python main.py --task ad --mode eval
```

* 반드시 **먼저 train**을 해서 해당 모델 파일이 생성되어 있어야 합니다.

---

### 3. 문장 예측 (Predict)

```bash
# 감정 분류 예측
python main.py --task emotion --mode predict --text "오늘 하루 너무 힘들었어"

# 광고성 분류 예측
python main.py --task ad --mode predict --text "지금 클릭하면 무료 증정!"
```

* 예시 결과:

```
[감정 분류]
입력: 오늘 하루 너무 힘들었어
예측 결과: 우울한

[광고성 분류]
입력: 지금 클릭하면 무료 증정!
예측 결과: 광고
```

* **주의:** `--text` 옵션을 반드시 입력해야 합니다.

---

## 전체 명령 예시

| 목적        | 명령어 예시                                                                |
|-----------|-----------------------------------------------------------------------|
| 감정 모델 학습  | `python main.py --task emotion --mode train`                          |
| 광고성 모델 학습 | `python main.py --task ad --mode train`                               |
| 감정 모델 평가  | `python main.py --task emotion --mode eval`                           |
| 광고성 모델 평가 | `python main.py --task ad --mode eval`                                |
| 감정 문장 예측  | `python main.py --task emotion --mode predict --text "오늘 하루 너무 힘들었어"` |
| 광고성 문장 예측 | `python main.py --task ad --mode predict --text "최신폰 무료 증정 이벤트!"`     |

---

## ⚙️ 개발 환경 및 개선 기록

### 1. Python & PyTorch 환경

* Python 버전: **3.11.9**
* PyTorch 버전: **2.8.0+cu128** (CUDA 12.8 지원)
* GPU: **NVIDIA GeForce RTX 5060 Ti**
* 설치 예시:

```bash
pip install torch==2.8.0+cu128 torchvision==0.19.0+cu128 torchaudio==2.5.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

### 2. GPU 인식 확인

```python
import torch

print("CUDA available?  :", torch.cuda.is_available())
print("GPU name         :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

출력 예시:

```
CUDA available?  : True
GPU name         : NVIDIA GeForce RTX 5060 Ti
Tensor device    : cuda:0
```

---

### 3. 전처리 개선

* 학습(`train`)과 예측(`predict`) 모두 동일 전처리 적용
* 학습 데이터: 문장을 512 이하 토큰 단위로 분리하여 정보 손실 최소화
* 예측 데이터: `truncation=True` 유지 (실시간성 확보)
* 토큰 길이 통계 로그 예시:

```
[STATS] Train (raw) | n=844 | mean=111.8 | p50=88 | p90=217 | p95=293 | max=901 | >512=1.8%
[INFO] Train samples: raw=844 -> after_split=866
```

---

### 4. 학습 최적화

* **MAX\_LEN**: 512 → **320** (p95 기준으로 메모리 효율 확보)
* **BATCH\_SIZE**: VRAM 고려해 16\~32 권장
* **Gradient Checkpointing**로 메모리 절약:

```python
if DEVICE.type == "cuda":
    model.bert.gradient_checkpointing_enable()
```

* **혼합정밀(AMP)** 적용 시 속도 및 메모리 절감 가능

---

### 5. CUDA 결정론 오류 해결

**문제**

```
RuntimeError: Deterministic behavior was enabled...
```

**원인**: `set_seed()`에서 결정론 모드가 활성화돼 있고, CUDA 연산(cuBLAS)이 비결정적

**해결 방법**

1. 환경변수 설정 (결정론 유지)

```bash
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

---

다음처럼 `README.md` 테스트 설명을 작성하면 됩니다.
스모크 케이스 선정 이유, 전체 배치 테스트 역할, 실행 방법을 모두 포함해 두면 팀원이 바로 이해할 수 있습니다.

---

## 테스트 구조 안내

### 1) 개요

이 프로젝트의 광고/정상 문장 분류 모델은 **단건 예측 검증**과 **전체 케이스 일괄 검증** 두 가지 방식으로 테스트합니다.

- **`test_ad_batch_predictions`**  
  단건 스모크 케이스만 검증해 **중복 실패 로그를 방지**하고, 예측 경로와 기본 동작을 빠르게 확인합니다.

- **`test_ad_batch_predictions_all`**  
  모든 케이스를 한 번에 배치 예측해 **정확도 전수 검증**을 수행하고, 실패 시 표 형태로 상세 내역을 출력합니다.

---

### 2) 스모크 케이스 선정 기준

스모크 테스트는 다음 기준을 만족하는 **소수의 대표 케이스**만 사용합니다.

1. **라벨 균형**: 광고/정상 각각 최소 1개 이상
2. **다양성 확보**:
    - 광고: `무료`, `클릭` 등 주요 상업 키워드
    - 정상: 일상 대화, 짧은 인사 등 모델이 혼동할 가능성이 낮은 문장
3. **속도 최적화**: 최소한의 케이스로 예측 경로만 확인
4. **회귀 방지**: 과거에 오분류가 발생했던 문장이면 우선 포함

---

### 3) 전체 케이스 검증

`test_ad_batch_predictions_all`는 `CASES`에 정의된 **모든 문장**을 배치 예측으로 검증합니다.

- 실패 시 다음과 같이 출력됩니다:

```
예측 불일치 목록:
[15] 반가워요  (기대: 광고 → 예측: 정상)
[27] 좋은 상품이에요  (기대: 광고 → 예측: 정상)
```

---

### 4) 실행 방법

#### 전체 테스트 실행

```bash
pytest -q
````

#### 특정 테스트 실행

```bash
pytest -k "test_ad_batch_predictions"
pytest -k "test_ad_batch_predictions_all"
```

---

### 5) 참고

* **단건 테스트**는 실패 시 메시지에 입력/예측/기대 값을 출력해 디버깅이 빠릅니다.
* **전체 테스트**는 표 형태로 한 번만 실패 로그를 남겨, 불필요한 중복 출력을 제거합니다.
* 케이스 추가/변경 시 `CASES`와 `SMOKE_IDX`를 함께 업데이트하세요.

---
이 문서의 **"개발 환경 및 개선 기록"** 섹션을 참고하면
새로운 환경에서도 GPU 활용 및 최적화된 학습을 문제없이 진행할 수 있습니다.

