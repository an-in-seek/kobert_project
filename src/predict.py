# src/predict.py

import torch
from torch.utils.data import DataLoader

from src.config import LABEL2EMOTION, BATCH_SIZE, DEVICE
from src.dataset import BERTDataset


def predict_sentence(
    model,
    tokenizer,
    sentence,
    max_len,
    device=DEVICE,
):
    """
    한 문장에 대해 감정 예측 결과 반환
    """
    data = [[sentence, 0]]  # label은 dummy
    dataset = BERTDataset(data, 0, 1, tokenizer, max_len, pad=True, pair=False)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, label in data_loader:
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            logits = outputs.detach().cpu().numpy()[0]
            pred_label = int(logits.argmax())
            emotion = LABEL2EMOTION.get(pred_label, "Unknown")
    return emotion


def predict_batch(
    model,
    tokenizer,
    sentences,
    max_len,
    device=DEVICE,
):
    """
    여러 문장에 대해 감정 예측 결과 리스트 반환
    """
    data = [[s, 0] for s in sentences]
    dataset = BERTDataset(data, 0, 1, tokenizer, max_len, pad=True, pair=False)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    model.eval()
    results = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, label in data_loader:
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            logits = outputs.detach().cpu().numpy()
            preds = logits.argmax(axis=1)
            emotions = [LABEL2EMOTION.get(int(p), "Unknown") for p in preds]
            results.extend(emotions)
    return results
