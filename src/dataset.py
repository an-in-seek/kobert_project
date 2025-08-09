# src/dataset.py
import numpy as np
import numpy as onp
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from src.bert_transform import BERTSentenceTransform
from src.config import RANDOM_SEED, NUM_WORKERS
from src.textproc import split_to_under_max_tokens, count_tokens


def load_data(data_path, text_col, label_col, str2label, test_size=0.25):
    """엑셀/CSV 로드 및 train/test 분할"""
    df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].map(str2label)
    data_list = [[row[text_col], int(row[label_col])] for _, row in df.iterrows()]
    train_data, test_data = train_test_split(
        data_list, test_size=test_size, random_state=RANDOM_SEED, shuffle=True
    )
    return train_data, test_data


def _log_token_stats(dataset, tokenizer, max_len, title: str):
    """토큰 길이 분포/초과 비율 로그"""
    lengths = [count_tokens(text, tokenizer) for text, _ in dataset]
    if not lengths:
        print(f"[STATS] {title}: empty dataset")
        return
    arr = onp.array(lengths)
    over = (arr > max_len).mean() * 100.0
    print(
        f"[STATS] {title} | n={len(arr)} | mean={arr.mean():.1f} | p50={onp.percentile(arr, 50):.0f} "
        f"| p90={onp.percentile(arr, 90):.0f} | p95={onp.percentile(arr, 95):.0f} "
        f"| max={arr.max():.0f} | >{max_len}={over:.1f}%"
    )


class GenericBERTDataset(Dataset):
    """
    모든 분류 작업에 공통 사용 가능한 PyTorch Dataset
    긴 텍스트는 문장 단위로 분할하여 512(또는 MAX_LEN) 이하 청크로 확장
    """

    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad=True, pair=False, split_long=True):
        self.transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        samples = []
        for item in dataset:
            text, label = item[sent_idx], int(item[label_idx])
            if split_long:
                for chunk in split_to_under_max_tokens(text, bert_tokenizer, max_len):
                    samples.append((chunk, label))
            else:
                samples.append((text, label))
        # 전처리 후(분할 포함) 길이 통계도 보고 싶으면 여기서 한 번 더 찍을 수 있음
        self.sentences = [self.transform([t]) for t, _ in samples]
        self.labels = [lbl for _, lbl in samples]

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = self.sentences[idx]
        label = self.labels[idx]
        return (
            np.array(input_ids),
            np.array(attention_mask),
            np.array(token_type_ids),
            np.int64(label)
        )

    def __len__(self):
        return len(self.labels)


def get_dataloaders(train_data, test_data, tokenizer, max_len, batch_size):
    # 원본 길이 분포 로그
    _log_token_stats(train_data, tokenizer, max_len, title="Train (raw)")
    _log_token_stats(test_data, tokenizer, max_len, title="Test  (raw)")

    # 학습 데이터: 길이 보존을 위해 분할 활성화
    train_dataset = GenericBERTDataset(train_data, 0, 1, tokenizer, max_len, pad=True, pair=False, split_long=True)
    # 평가 데이터: 일반적으로 문장 단위 분할까지는 필요 없지만, 필요 시 True로
    test_dataset = GenericBERTDataset(test_data, 0, 1, tokenizer, max_len, pad=True, pair=False, split_long=False)

    # 분할/전처리 후 샘플이 어떻게 늘었는지 간단 로그
    print(f"[INFO] Train samples: raw={len(train_data)} -> after_split={len(train_dataset)}")
    print(f"[INFO] Test  samples: raw={len(test_data)}  -> after_split={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader
