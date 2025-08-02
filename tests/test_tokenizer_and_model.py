# test_tokenizer_and_model.py

import os

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel


def test_kobert_tokenizer_and_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'kobert-base-v1-local')
    tokenizer = KoBERTTokenizer.from_pretrained(model_dir)
    encoded = tokenizer.encode("한국어 모델 테스트")
    print("인코딩 결과:", encoded)
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    model = BertModel.from_pretrained(model_dir)
    print("모델 일부 파라미터:", list(model.named_parameters())[0])
    assert hasattr(model, "forward")


if __name__ == "__main__":
    test_kobert_tokenizer_and_model()
