# src/utils.py
# 유틸 함수(라벨 매핑 등)

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    실험 재현성을 위한 랜덤 시드 고정 (모든 분류 작업 공통)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_to_str(label: int, label2str: dict) -> str:
    """
    숫자 레이블을 문자열로 변환 (공통 라벨 매핑)
    """
    return label2str.get(label, "Unknown")


def str_to_label(s: str, str2label: dict) -> int:
    """
    문자열을 숫자 레이블로 변환 (공통 라벨 매핑)
    """
    return str2label.get(s, -1)


def print_model_info(model):
    """
    모델 파라미터 및 레이어 수 요약 출력 (모든 분류 모델 공통)
    """
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {n_params:,}")
    print(model)
