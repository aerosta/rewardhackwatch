"""Train DistilBERT classifier for reward hack detection."""

import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer, get_linear_schedule_with_warmup


class TrajectoryDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Pre-tokenize all data for speed
        print(f"  Tokenizing {len(data)} samples...", flush=True)
        self.encodings = []
        for i, t in enumerate(tqdm(data, desc="Tokenizing")):
            text = t["text"][:10000]  # Truncate very long texts
            enc = tokenizer(
                text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt"
            )
            self.encodings.append(
                {
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "label": torch.tensor(1 if t["is_hack"] else 0),
                }
            )
        print("  Done tokenizing.", flush=True)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


class Classifier(nn.Module):
    def __init__(self, hidden=256, dropout=0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Sequential(
            nn.Linear(768, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )

    def forward(self, input_ids, attention_mask):
        return self.fc(
            self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        )


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labels.extend(batch["label"].numpy())
    return {
        "acc": accuracy_score(labels, preds),
        "p": precision_score(labels, preds, zero_division=0),
        "r": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "cm": confusion_matrix(labels, preds).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test with 500 samples")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-len", type=int, default=256, help="Max token length")
    args = parser.parse_args()

    # Force MPS (Apple GPU) with fallback
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if device.type == "mps":
        print("✅ Running on Apple GPU (MPS)", flush=True)
    else:
        print("⚠️ Falling back to CPU", flush=True)
    print(f"Using device: {device}", flush=True)

    print("Loading data...", flush=True)
    with open("data/training/train.json") as f:
        train_data = json.load(f)
    with open("data/training/val.json") as f:
        val_data = json.load(f)
    with open("data/training/test.json") as f:
        test_data = json.load(f)

    # Quick mode: use subset with stratified sampling
    if args.quick:
        print("🚀 QUICK MODE: Using 500 train, 100 val, 100 test samples", flush=True)
        # Stratified sample to include hacks
        train_hacks = [x for x in train_data if x["is_hack"]]
        train_clean = [x for x in train_data if not x["is_hack"]]
        val_hacks = [x for x in val_data if x["is_hack"]]
        val_clean = [x for x in val_data if not x["is_hack"]]
        test_hacks = [x for x in test_data if x["is_hack"]]
        test_clean = [x for x in test_data if not x["is_hack"]]
        print(
            f"  Full data - Train hacks: {len(train_hacks)}, Val hacks: {len(val_hacks)}, Test hacks: {len(test_hacks)}",
            flush=True,
        )
        # Take proportional samples
        train_data = train_hacks[:50] + train_clean[:450]
        val_data = val_hacks[:10] + val_clean[:90]
        test_data = test_hacks[:10] + test_clean[:90]
        import random

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    print("Creating datasets...", flush=True)
    train_ds = TrajectoryDataset(train_data, tokenizer, max_len=args.max_len)
    val_ds = TrajectoryDataset(val_data, tokenizer, max_len=args.max_len)
    test_ds = TrajectoryDataset(test_data, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print("Loading model...", flush=True)
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, len(train_loader), len(train_loader) * args.epochs
    )
    criterion = nn.CrossEntropyLoss()
    Path("models").mkdir(exist_ok=True)

    print(f"\n🏋️ Starting training for {args.epochs} epochs...\n", flush=True)
    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, Val F1={val_metrics['f1']:.4f}, Val Acc={val_metrics['acc']:.4f}",
            flush=True,
        )
        if val_metrics["f1"] >= best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  💾 Saved best model (F1={best_f1:.4f})", flush=True)

    print("\n📊 Evaluating on test set...", flush=True)
    if Path("models/best_model.pt").exists():
        model.load_state_dict(torch.load("models/best_model.pt", weights_only=True))
    else:
        print("  (Using final model - no best model saved)", flush=True)
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n{'=' * 50}")
    print("  TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {test_metrics['acc']:.4f}")
    print(f"  Precision: {test_metrics['p']:.4f}")
    print(f"  Recall:    {test_metrics['r']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"{'=' * 50}\n")

    with open("models/results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Results saved to models/results.json", flush=True)


if __name__ == "__main__":
    main()
