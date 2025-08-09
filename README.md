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
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

2. 또는 결정론 모드 해제:

```python
torch.use_deterministic_algorithms(False)
```

---

이 문서의 **"개발 환경 및 개선 기록"** 섹션을 참고하면
새로운 환경에서도 GPU 활용 및 최적화된 학습을 문제없이 진행할 수 있습니다.

