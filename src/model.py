# src/model.py
# KoBERT 분류기 (HuggingFace Transformers + PyTorch + Local Model 경로 안전화)

import os

from kobert_tokenizer import KoBERTTokenizer
from torch import nn
from transformers import BertModel


def get_kobert_model_and_tokenizer():
    """
    로컬 디렉토리에서 KoBERT 모델과 토크나이저를 로드
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'kobert-base-v1-local')
    tokenizer = KoBERTTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)
    return model, tokenizer


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=2, dr_rate=None):
        """
        bert: 사전학습 KoBERT 모델
        hidden_size: KoBERT 출력 차원(기본 768)
        num_classes: 분류 클래스 수 (task별로 다름)
        dr_rate: Dropout 확률 (None이면 Dropout 미사용)
        """
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
        pooled = outputs.pooler_output  # [batch, hidden]
        out = self.dropout(pooled)
        return self.classifier(out)
