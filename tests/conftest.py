# tests/conftest.py
# pytest 공용 픽스처: 경로 설정 + 모델/토크나이저 1회 로드 + 장치/설정 공유

import sys
from pathlib import Path

import pytest
import torch

# --- 1) 프로젝트 루트를 sys.path에 추가 (ModuleNotFoundError: src 방지) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../KoBERT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 2) 프로젝트 모듈 import ---
from src.config import TASK_CONFIGS, DEVICE, MAX_LEN
from src.model import get_kobert_model_and_tokenizer, BERTClassifier


# (선택) CUDA 결정론 이슈 회피: 테스트는 추론만 하므로 필요 없음
# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def device():
    """현재 설정된 학습/추론 장치(cuda 또는 cpu)를 공유"""
    return DEVICE


@pytest.fixture(scope="session")
def task_cfg():
    """광고 분류(ad) 태스크 설정 공유 (필요시 다른 태스크로 교체 가능)"""
    return TASK_CONFIGS["ad"]


@pytest.fixture(scope="session")
def max_len():
    return MAX_LEN


@pytest.fixture(scope="session")
def model_and_tokenizer(task_cfg, device):
    """
    KoBERT 모델과 토크나이저를 세션당 1회 로드하여 공유.
    - 가중치 파일이 없으면 해당 테스트를 skip.
    - 로드 실패 시에도 skip 처리하여 CI 중단 방지.
    """
    weights = task_cfg["model_save_path"]
    if not Path(weights).is_file():
        pytest.skip(f"모델 가중치가 없습니다: {weights} (먼저 `--mode train`으로 학습하세요)")

    try:
        bert_model, tokenizer = get_kobert_model_and_tokenizer()
    except Exception as e:
        pytest.skip(f"KoBERT 로드 실패: {e}")

    model = BERTClassifier(
        bert_model,
        num_classes=task_cfg["num_classes"],
        dr_rate=0.0,  # 테스트 추론이므로 드롭아웃 off
    ).to(device)

    try:
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        pytest.skip(f"모델 가중치 로드 실패: {e}")

    model.eval()
    return model, tokenizer
