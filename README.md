# 😃 KoBERT 기반 한국어 감정 분류 프로젝트

이 프로젝트는 SKT KoBERT 모델을 활용하여 한국어 문장의 감정(8가지)을 분류하는 텍스트 분류(감성 분석) 모델입니다.  
주어진 문장을 입력하면 모델이 '우울한', '기쁜', '화가 나는', '슬픈', '편안한', '걱정스러운', '신이 난', '충만한' 중 하나로 예측합니다.

- KoBERT(Huggingface 기반)를 사용하여 구현
- 학습 데이터: 감성대화말뭉치(긍부정)_Training.xlsx (한국어 감정 대화 데이터)
- 파이토치(PyTorch)와 transformers 라이브러리 기반

---

## 🚀 실행 방법 (How to Run)

### 1. 모델 학습 (Train)

처음 사용하거나, 모델을 새로 학습하고 싶을 때 사용합니다.

```bash
python main.py --mode train
````

* 데이터셋을 불러와서 모델을 학습합니다.
* 학습이 끝나면 `kobert_emotion.pt` 파일이 생성됩니다.

---

### 2. 모델 평가 (Eval)

학습된 모델의 테스트셋 성능(정확도, 손실 등)을 평가합니다.

```bash
python main.py --mode eval
````

* 반드시 **먼저 train**을 해서 `kobert_emotion.pt`가 생성되어 있어야 합니다.
* 테스트셋 기준 모델 성능(정확도 등)을 출력합니다.

---

### 3. 문장 감정 예측 (Predict)

한 문장에 대해 감정 예측 결과를 확인할 때 사용합니다.

```bash
python main.py --mode predict --text "오늘 하루 너무 힘들었어"
```

* 예시 결과:

  ```
  입력: 오늘 하루 너무 힘들었어
  예측 감정: 우울한
  ```

* **참고:** `--text` 뒤에 예측하고 싶은 문장을 반드시 입력해야 합니다.

---

### ❗️에러 주의사항

* `kobert_emotion.pt`가 없을 경우, `--mode train`으로 먼저 학습해야 `eval` 또는 `predict`가 동작합니다.
* `predict`에서 `--text` 옵션을 빠뜨리면 오류가 발생합니다.

---

## 전체 명령 예시

| 목적       | 명령어 예시                                                 |
|----------|--------------------------------------------------------|
| 모델 학습    | `python main.py --mode train`                          |
| 모델 평가    | `python main.py --mode eval`                           |
| 문장 감정 예측 | `python main.py --mode predict --text "오늘 하루 너무 힘들었어"` |
