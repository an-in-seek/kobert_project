# src/predict.py

from typing import List

import torch

from src.textproc import clean_text, count_tokens


@torch.no_grad()
def predict_sentence(
    model,
    tokenizer,
    sentence: str,
    max_len: int,
    device,
    label2str: dict,
):
    """
    한 문장에 대해 예측 결과(레이블 문자열) 반환
    - 학습과 동일 전처리(clean_text) 적용
    - 512/ max_len 초과 시 truncation=True로 안전 절단
    """
    model.eval()
    sentence = clean_text(sentence)
    enc = tokenizer(
        sentence,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    logits = model(input_ids, attention_mask, token_type_ids)
    pred_label = int(logits.argmax(dim=1).item())
    return label2str.get(pred_label, "Unknown")


@torch.no_grad()
def predict_batch(
    model,
    tokenizer,
    sentences: List[str],
    max_len: int,
    device,
    label2str: dict,
    log_stats: bool = True,
):
    """
    여러 문장 예측
    - 공통 전처리(clean_text) 일괄 적용
    - 토크나이저 배치 인코딩으로 효율 개선
    - 선택적으로 토큰 길이 통계 로그 출력
    """
    model.eval()
    cleaned = [clean_text(s) for s in sentences]

    if log_stats and len(cleaned) > 0:
        lengths = [count_tokens(s, tokenizer) for s in cleaned]
        import numpy as np
        arr = np.array(lengths)
        over = (arr > max_len).mean() * 100.0
        print(
            f"[STATS] Predict Batch | n={len(arr)} | mean={arr.mean():.1f} "
            f"| p90={np.percentile(arr, 90):.0f} | p95={np.percentile(arr, 95):.0f} "
            f"| max={arr.max():.0f} | >{max_len}={over:.1f}%"
        )

    enc = tokenizer(
        cleaned,
        truncation=True,  # 실시간 예측: 길면 절단
        padding="max_length",  # 모델 입력 고정 길이 유지
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    logits = model(input_ids, attention_mask, token_type_ids)
    preds = logits.argmax(dim=1).tolist()
    return [label2str.get(int(p), "Unknown") for p in preds]
