# src/bert_transform.py
import numpy as np

from src.textproc import clean_text


class BERTSentenceTransform:
    """
    입력 문장을 transformers용 KoBERT tokenizer로 변환
    (input_ids, attention_mask, token_type_ids) 반환
    """

    def __init__(self, tokenizer, max_seq_length, pad=True, pair=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad = pad
        self.pair = pair

    def __call__(self, line):
        if self.pair and len(line) == 2:
            text_a, text_b = clean_text(line[0]), clean_text(line[1])
            encoding = self.tokenizer.encode_plus(
                text_a, text_b,
                max_length=self.max_seq_length,
                padding='max_length' if self.pad else None,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True
            )
        else:
            text_a = clean_text(line[0])
            encoding = self.tokenizer.encode_plus(
                text_a,
                max_length=self.max_seq_length,
                padding='max_length' if self.pad else None,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True
            )
        input_ids = np.array(encoding['input_ids'], dtype='int64')
        attention_mask = np.array(encoding['attention_mask'], dtype='int64')
        token_type_ids = np.array(encoding['token_type_ids'], dtype='int64')
        return input_ids, attention_mask, token_type_ids
