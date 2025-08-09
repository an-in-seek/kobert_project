# tests/test.py

import collections
import os

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from src.config import TASK_CONFIGS, MAX_LEN, BATCH_SIZE
from src.dataset import load_data, get_dataloaders


def main():
    # 프로젝트 최상위 폴더 자동 탐색
    current_path = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.dirname(current_path)
    model_dir = os.path.join(base_dir, 'kobert-base-v1-local')

    # 1. KoBERT 토크나이저/모델 정상 로드 확인
    print("=== KoBERT 토크나이저/모델 로딩 테스트 ===")
    tokenizer = KoBERTTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer.encode("테스트 문장입니다.")
    print("인코딩 결과:", encoded)
    model = BertModel.from_pretrained(model_dir, local_files_only=True)
    print("모델 일부 파라미터:", list(model.named_parameters())[0])

    # 2. 감정 분류 config 사용
    task_cfg = TASK_CONFIGS["emotion"]

    # 3. 데이터 로드 및 라벨 분포
    train_data, test_data = load_data(
        data_path=task_cfg["data_path"],
        text_col=task_cfg["text_col"],
        label_col=task_cfg["label_col"],
        str2label=task_cfg["str2label"]
    )
    print("\n=== 라벨 분포 (train+test) ===")
    all_labels = [x[1] for x in train_data + test_data]
    print(collections.Counter(all_labels))

    # 4. train 데이터 샘플 출력
    print("\n=== train 데이터 예시 ===")
    for x in train_data[:10]:
        print(x)

    # 5. DataLoader 배치 체크 (Windows 호환)
    print("\n=== DataLoader 첫 배치 샘플 ===")
    train_loader, _ = get_dataloaders(train_data, test_data, tokenizer, MAX_LEN, BATCH_SIZE)
    for i, batch in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids, label = batch
        print("input_ids shape:", input_ids.shape)
        print("labels:", label.tolist())
        if i > 0:
            break

    # 6. 토크나이저로 주요 단어 토큰화 직접 체크
    test_words = ["좋아", "짜증나", "우울해", "편안한"]
    for word in test_words:
        print(f"'{word}' 인코딩:", tokenizer.encode(word))


if __name__ == "__main__":
    main()
