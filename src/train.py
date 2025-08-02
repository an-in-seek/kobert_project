# src/train.py
import torch
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

from src.config import WARMUP_RATIO, MAX_GRAD_NORM, NUM_EPOCHS, LOG_INTERVAL, DEVICE


def calc_accuracy(logits, labels):
    max_indices = torch.argmax(logits, dim=1)
    acc = (max_indices == labels).sum().item() / labels.size(0)
    return acc


def train(
    model,
    train_loader,
    test_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=5e-5,
    model_save_path="kobert_emotion.pt",
    device=DEVICE,
):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_loader) * num_epochs
    warmup_steps = int(t_total * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    print("[INFO] Training start.")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch_id, (input_ids, attention_mask, token_type_ids, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_acc += calc_accuracy(out, label)

            if (batch_id + 1) % LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch + 1} | Batch {batch_id + 1} | "
                    f"Loss {loss.item():.4f} | Train Acc {train_acc / (batch_id + 1):.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")

        # 평가
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for batch_id, (input_ids, attention_mask, token_type_ids, label) in enumerate(test_loader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                label = label.to(device)
                out = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(out, label)
                test_loss += loss.item()
                test_acc += calc_accuracy(out, label)

        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)
        print(f"Epoch {epoch + 1} | Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f}")

        torch.save(model.state_dict(), model_save_path)
        print(f"[INFO] Model saved to {model_save_path}")


def evaluate(model, test_loader, device=DEVICE):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (input_ids, attention_mask, token_type_ids, label) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)
            out = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(out, label)
            test_loss += loss.item()
            test_acc += calc_accuracy(out, label)
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    print(f"[EVAL] Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f}")
    return avg_test_loss, avg_test_acc
