# main.py
# 전체 파이프라인 실행 (학습/평가/예측)

# --mode train
# --mode predict --text "오늘 하루 너무 힘들었어"

import argparse

import torch

from src.config import (
    DATA_PATH,
    MODEL_SAVE_PATH,
    DEVICE,
    MAX_LEN,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from src.dataset import load_data, get_dataloaders
from src.model import BERTClassifier, get_kobert_model_and_tokenizer
from src.predict import predict_sentence
from src.train import train, evaluate


def main():
    parser = argparse.ArgumentParser(description="KoBERT Emotion Classification")
    parser.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True, help='실행 모드 선택')
    parser.add_argument('--text', type=str, help='예측할 문장 (predict 모드에서 사용)')
    args = parser.parse_args()

    # KoBERT 모델, 토크나이저 불러오기 (vocab 제거)
    bert_model, tokenizer = get_kobert_model_and_tokenizer()
    model = BERTClassifier(bert_model, dr_rate=0.5).to(DEVICE)

    if args.mode == 'train':
        print("[INFO] 데이터 로딩 및 전처리...")
        train_data, test_data = load_data(DATA_PATH)
        train_loader, test_loader = get_dataloaders(train_data, test_data, tokenizer, MAX_LEN, BATCH_SIZE)
        print("[INFO] 학습 시작")
        train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            model_save_path=MODEL_SAVE_PATH,
            device=DEVICE
        )

    elif args.mode == 'eval':
        print("[INFO] 모델 로드 및 평가...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        train_data, test_data = load_data(DATA_PATH)
        _, test_loader = get_dataloaders(train_data, test_data, tokenizer, MAX_LEN, BATCH_SIZE)
        evaluate(
            model=model,
            test_loader=test_loader,
            device=DEVICE
        )

    elif args.mode == 'predict':
        if not args.text:
            print("[ERROR] --text 옵션에 예측할 문장을 입력하세요.")
            return
        print("[INFO] 모델 로드 및 예측")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        result = predict_sentence(
            model=model,
            tokenizer=tokenizer,
            sentence=args.text,
            max_len=MAX_LEN,
            device=DEVICE
        )
        print(f"입력: {args.text}")
        print(f"예측 감정: {result}")


if __name__ == "__main__":
    main()
