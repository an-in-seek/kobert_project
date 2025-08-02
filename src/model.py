# src/model.py
# KoBERT 분류기 (HuggingFace Transformers + PyTorch + Local Model 경로 안전화)

import os

from kobert_tokenizer import KoBERTTokenizer
from torch import nn
from transformers import BertModel


def get_kobert_model_and_tokenizer():
    """
    로컬에 저장된 kobert-base-v1-local 폴더에서 KoBERT 모델과 토크나이저를 로드
    """
    # model.py가 src/에 있다고 가정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'kobert-base-v1-local')
    tokenizer = KoBERTTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)
    return model, tokenizer


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=8, dr_rate=None):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dr_rate) if dr_rate else nn.Identity()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output  # transformers >=4.8.2
        out = self.dropout(pooled)
        return self.classifier(out)
