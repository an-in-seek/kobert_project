# src/config.py
# 하이퍼파라미터 및 환경설정

import os

import torch

# 데이터 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "감성대화말뭉치(긍부정)_Training.xlsx")

# 모델 저장 경로
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "kobert_emotion.pt")

# 하이퍼파라미터
MAX_LEN = 64
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1
LOG_INTERVAL = 200

# 디바이스 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 감정 라벨 맵핑 (숫자 → 한글)
LABEL2EMOTION = {
    0: "우울한",
    1: "기쁜",
    2: "화가 나는",
    3: "슬픈",
    4: "편안한",
    5: "걱정스러운",
    6: "신이 난",
    7: "충만한",
}

EMOTION2LABEL = {v: k for k, v in LABEL2EMOTION.items()}  # 한글 → 숫자 매핑

# 기타 설정 (필요에 따라 추가)
RANDOM_SEED = 42
NUM_WORKERS = 2  # DataLoader 병렬 처리
