# tests/test_ad_predictions.py
from typing import List, Tuple

import pytest

from src.config import TASK_CONFIGS
from src.predict import predict_sentence

CASES: List[Tuple[str, str]] = [
    ("지금 클릭하면 무료 증정!", "광고"),
    ("안녕하세요", "정상"),
    ("지금 구매하시면 1+1 혜택!", "광고"),
    ("오늘 날씨 참 좋네요.", "정상"),
    # 필요 시 케이스 추가
]
IDS = [f"case_{i}_{'ad' if e == '광고' else 'normal'}" for i, (_, e) in enumerate(CASES)]


@pytest.mark.parametrize("text,expected", CASES, ids=IDS)
def test_ad_single_predictions(model_and_tokenizer, max_len, device, text, expected):
    model, tokenizer = model_and_tokenizer
    label2str = TASK_CONFIGS["ad"]["label2str"]

    pred = predict_sentence(
        model=model,
        tokenizer=tokenizer,
        sentence=text,
        max_len=max_len,
        device=device,
        label2str=label2str,
    )
    assert pred == expected, (
        f"입력: {text}\n예측: {pred}\n기대: {expected}"
    )
