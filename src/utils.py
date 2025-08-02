# src/utils.py
# 유틸 함수(라벨 매핑 등)

import random

import numpy as np
import torch

from src.config import LABEL2EMOTION, EMOTION2LABEL


def set_seed(seed: int = 42):
    """
    실험 재현성을 위한 랜덤 시드 고정
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_to_emotion(label: int) -> str:
    """
    숫자 라벨을 감정 문자열로 변환
    """
    return LABEL2EMOTION.get(label, "Unknown")


def emotion_to_label(emotion: str) -> int:
    """
    감정 문자열을 숫자 라벨로 변환
    """
    return EMOTION2LABEL.get(emotion, -1)


def print_model_info(model):
    """
    모델 파라미터 및 레이어 수 요약 출력
    """
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {n_params:,}")
    print(model)
