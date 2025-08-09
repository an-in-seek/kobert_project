# src/dataset.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from src.bert_transform import BERTSentenceTransform
from src.config import RANDOM_SEED, NUM_WORKERS


def load_data(data_path, text_col, label_col, str2label, test_size=0.25):
    """
    엑셀/CSV 데이터 로드 및 train/test 분할 (task별 text/label 컬럼명과 라벨 매핑 사용)
    """
    df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].map(str2label)
    data_list = [[row[text_col], int(row[label_col])] for _, row in df.iterrows()]
    train_data, test_data = train_test_split(
        data_list, test_size=test_size, random_state=RANDOM_SEED, shuffle=True
    )
    return train_data, test_data


class GenericBERTDataset(Dataset):
    """
    모든 분류 작업에 공통 사용 가능한 PyTorch Dataset
    """

    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad=True, pair=False):
        self.transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [self.transform([i[sent_idx]]) for i in dataset]
        self.labels = [int(i[label_idx]) for i in dataset]

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
    train_dataset = GenericBERTDataset(train_data, 0, 1, tokenizer, max_len, pad=True, pair=False)
    test_dataset = GenericBERTDataset(test_data, 0, 1, tokenizer, max_len, pad=True, pair=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader
