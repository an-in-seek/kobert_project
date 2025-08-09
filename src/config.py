# src/config.py
# 하이퍼파라미터 및 환경설정 (최종 개선본)

import os

import torch

# 프로젝트 최상위 폴더 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 공통 하이퍼파라미터
MAX_LEN = 320  # KoBERT 최대 토큰 길이
BATCH_SIZE = 32  # MAX_LEN 상승 시 메모리 여유 없으면 32/16으로 조정
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1
LOG_INTERVAL = 200
RANDOM_SEED = 42
NUM_WORKERS = 0

# 안전 가드
assert MAX_LEN <= 512, f"KoBERT 최대 토큰은 512입니다. 현재 MAX_LEN={MAX_LEN}"

# 디바이스 설정 및 실행 환경 로그
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 저장 경로 패턴 (task별로 다르게 저장)
MODEL_SAVE_PATH_PATTERN = os.path.join(BASE_DIR, "saved_model_{task}.pt")

# 분류 작업별 config 집합
TASK_CONFIGS = {
    "emotion": {
        "label2str": {
            0: "우울한",
            1: "기쁜",
            2: "화가나는",
            3: "슬픈",
            4: "편안한",
            5: "걱정스러운",
        },
        "str2label": {
            "우울한": 0,
            "기쁜": 1,
            "화가나는": 2,
            "슬픈": 3,
            "편안한": 4,
            "걱정스러운": 5,
        },
        "num_classes": 6,
        "data_path": os.path.join(BASE_DIR, "data", "emotion_ko_dataset.csv"),
        "text_col": "Sentence",
        "label_col": "Emotion",
        "model_save_path": MODEL_SAVE_PATH_PATTERN.format(task="emotion"),
    },
    "ad": {
        "label2str": {
            0: "정상",
            1: "광고",
        },
        "str2label": {
            "정상": 0,
            "광고": 1,
        },
        "num_classes": 2,
        "data_path": os.path.join(BASE_DIR, "data", "ad_classification.csv"),
        "text_col": "Sentence",
        "label_col": "Label",
        "model_save_path": MODEL_SAVE_PATH_PATTERN.format(task="ad"),
    },
    # 추가 작업 확장 가능
    # "topic": {...}
}
