# main.py
# 전체 파이프라인 실행 (학습/평가/예측)

import argparse

import torch

from src.config import (
    TASK_CONFIGS,
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
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="KoBERT Multi-Task Classification")
    parser.add_argument('--task', choices=TASK_CONFIGS.keys(), required=True, help='분류 작업 종류 선택 (emotion, ad 등)')
    parser.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True, help='실행 모드 선택')
    parser.add_argument('--text', type=str, help='예측할 문장 (predict 모드에서 사용)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (기본값 42)')
    args = parser.parse_args()

    # 랜덤 시드 고정
    set_seed(args.seed)

    task_cfg = TASK_CONFIGS[args.task]

    # KoBERT 모델 및 토크나이저 로드
    bert_model, tokenizer = get_kobert_model_and_tokenizer()
    model = BERTClassifier(
        bert_model,
        num_classes=task_cfg['num_classes'],
        dr_rate=0.5
    ).to(DEVICE)

    if args.mode in ['train', 'eval']:
        # 데이터 로드
        train_data, test_data = load_data(
            data_path=task_cfg["data_path"],
            text_col=task_cfg["text_col"],
            label_col=task_cfg["label_col"],
            str2label=task_cfg["str2label"]
        )
        train_loader, test_loader = get_dataloaders(
            train_data, test_data, tokenizer, MAX_LEN, BATCH_SIZE
        )

    if args.mode == 'train':
        print("[INFO] 학습 시작")
        train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            model_save_path=task_cfg["model_save_path"],
            device=DEVICE
        )

    elif args.mode == 'eval':
        print("[INFO] 평가 모드: 모델 로드 및 테스트셋 평가")
        model.load_state_dict(torch.load(task_cfg["model_save_path"], map_location=DEVICE))
        evaluate(
            model=model,
            test_loader=test_loader,
            device=DEVICE
        )

    elif args.mode == 'predict':
        if not args.text:
            print("[ERROR] --text 옵션에 예측할 문장을 입력하세요.")
            return
        print("[INFO] 예측 모드: 모델 로드 및 문장 분류")
        model.load_state_dict(torch.load(task_cfg["model_save_path"], map_location=DEVICE))
        result = predict_sentence(
            model=model,
            tokenizer=tokenizer,
            sentence=args.text,
            max_len=MAX_LEN,
            device=DEVICE,
            label2str=task_cfg["label2str"]
        )
        print(f"입력: {args.text}")
        print(f"예측 결과: {result}")


if __name__ == "__main__":
    main()
