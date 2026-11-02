#!/usr/bin/env python3
"""
Trains the 2D MRI slice classifier used when a picture (JPG/PNG) is uploaded
to the dashboard. Pools every labeled image under "MRI DS/Combined Dataset",
drops exact duplicates, and makes a stratified 80/20 train/val split.

Usage:
    python scripts/06_train_slice2d.py
"""
import sys
import hashlib
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.models.slice2d import SliceClassifier2D, image_to_tensor, FOLDER_TO_CLASS, CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "MRI DS" / "Combined Dataset"
OUT_PATH = ROOT / "outputs" / "checkpoints" / "slice2d_model.pt"
EPOCHS = 70
SEED = 42


def collect():
    seen, items = set(), []
    for f in sorted(DATA_DIR.glob("*/*/*")):
        label = FOLDER_TO_CLASS.get(f.parent.name)
        if label is None or f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        items.append((f, label))
    return items


class SliceDataset(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("L")
        if self.train:
            img = img.rotate(random.uniform(-10, 10), resample=Image.BILINEAR)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = image_to_tensor(img)[0]
        if self.train:
            x = x * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return x, label


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return (preds == labels).mean(), recall.mean(), cm


if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    items = collect()
    train_items, val_items = [], []
    for c in range(3):
        cls_items = [it for it in items if it[1] == c]
        random.shuffle(cls_items)
        n_val = int(0.2 * len(cls_items))
        val_items += cls_items[:n_val]
        train_items += cls_items[n_val:]
    counts = np.bincount([y for _, y in train_items], minlength=3)
    print(f"[data] unique={len(items)} train={len(train_items)} val={len(val_items)} "
          f"train per class={dict(zip(CLASSES, counts.tolist()))}")

    train_dl = DataLoader(SliceDataset(train_items, True), batch_size=32, shuffle=True)
    val_dl = DataLoader(SliceDataset(val_items, False), batch_size=64)

    model = SliceClassifier2D().to(device)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * len(train_dl))

    best_bal = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step(); sched.step()
            total += loss.item() * len(y)
        acc, bal, cm = evaluate(model, val_dl, device)
        print(f"epoch {epoch:02d} loss={total / len(train_items):.4f} val_acc={acc:.3f} val_bal_acc={bal:.3f}")
        if bal > best_bal:
            best_bal = bal
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.cpu().state_dict(), "classes": CLASSES,
                        "val_acc": float(acc), "val_balanced_acc": float(bal), "confusion": cm.tolist()}, OUT_PATH)
            model.to(device)

    ckpt = torch.load(OUT_PATH)
    print(f"\nBest val balanced acc={ckpt['val_balanced_acc']:.3f} acc={ckpt['val_acc']:.3f}")
    print("Confusion (rows=true CN/MCI/AD, cols=pred):")
    for row in ckpt["confusion"]:
        print("  ", row)
    print(f"Saved -> {OUT_PATH}")
