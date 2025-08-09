# test_tokenizer_and_model.py

import os
import sys

import torch  # ← torch import 추가
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel


def test_kobert_tokenizer_and_model():
    # 프로젝트 최상위 폴더 자동 탐색
    current_path = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.dirname(current_path)
    model_dir = os.path.join(base_dir, 'kobert-base-v1-local')

    # 토크나이저 테스트
    try:
        tokenizer = KoBERTTokenizer.from_pretrained(model_dir)
        encoded = tokenizer.encode("한국어 모델 테스트")
        print("인코딩 결과:", encoded)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
    except Exception as e:
        print(f"[ERROR] KoBERTTokenizer 테스트 실패: {e}")
        sys.exit(1)

    # 모델 테스트
    try:
        model = BertModel.from_pretrained(model_dir)
        print("모델 일부 파라미터:", list(model.named_parameters())[0])
        assert hasattr(model, "forward")
        # 임의 텍스트 인코딩 → 입력 텐서 → 모델 패스 테스트
        inputs = tokenizer.encode_plus(
            "한국어 모델 테스트",
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # token_type_ids가 없을 수도 있음
        model_inputs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
        }
        if 'token_type_ids' in inputs:
            model_inputs["token_type_ids"] = inputs['token_type_ids']

        with torch.no_grad():
            outputs = model(**model_inputs)
        assert hasattr(outputs, "pooler_output")
        print("모델 출력 pooler_output shape:", outputs.pooler_output.shape)
    except Exception as e:
        print(f"[ERROR] KoBERTModel 테스트 실패: {e}")
        sys.exit(1)

    print("[OK] KoBERT 토크나이저/모델 모두 정상 작동!")


if __name__ == "__main__":
    test_kobert_tokenizer_and_model()
