# src/utils.py
# 유틸 함수(시드, 라벨 매핑, 모델 요약)

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    실험 재현성을 위한 랜덤 시드 고정
    - Python/NumPy/PyTorch 시드
    - 결정론 모드 활성화
    - CUDA/CPU 모두 안전
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cuDNN 결정론/튜닝 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 연산 전반에 결정론 강제 (가능한 경우)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # 일부 환경/버전에서 미지원일 수 있음 → 무시
        pass


def label_to_str(label: int, label2str: dict) -> str:
    """숫자 레이블을 문자열로 변환"""
    return label2str.get(label, "Unknown")


def str_to_label(s: str, str2label: dict) -> int:
    """문자열을 숫자 레이블로 변환"""
    return str2label.get(s, -1)


def print_model_info(model):
    """
    모델 파라미터 요약 출력
    - 총 파라미터 수
    - 학습 가능한 파라미터 수
    - 모델 구조
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: total={total_params:,} | trainable={trainable_params:,}")
    print(model)
