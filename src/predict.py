# src/predict.py

import torch
from torch.utils.data import DataLoader

from src.dataset import GenericBERTDataset


def predict_sentence(
    model,
    tokenizer,
    sentence,
    max_len,
    device,
    label2str,
):
    """
    한 문장에 대해 예측 결과(레이블 문자열) 반환
    """
    data = [[sentence, 0]]  # label은 dummy
    dataset = GenericBERTDataset(data, 0, 1, tokenizer, max_len, pad=True, pair=False)
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
            pred_str = label2str.get(pred_label, "Unknown")
    return pred_str


def predict_batch(
    model,
    tokenizer,
    sentences,
    max_len,
    device,
    label2str,
    batch_size=32,
):
    """
    여러 문장에 대해 예측 결과(레이블 문자열 리스트) 반환
    """
    data = [[s, 0] for s in sentences]
    dataset = GenericBERTDataset(data, 0, 1, tokenizer, max_len, pad=True, pair=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

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
            preds_str = [label2str.get(int(p), "Unknown") for p in preds]
            results.extend(preds_str)
    return results
