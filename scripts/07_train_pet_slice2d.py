#!/usr/bin/env python3
"""
Trains the 2D PET slice classifier used when a PET picture (JPG/PNG) is uploaded
to the dashboard. Uses the labeled axial PET slices in data/real/pet
(real-pet-{cn,mci,ad}-NNN.npy), starts from the MRI slice model's weights, and
reports 5-fold cross-validated accuracy before training the final model on all data.

Usage:
    python scripts/06_train_slice2d.py      # MRI model first (used as initialisation)
    python scripts/07_train_pet_slice2d.py
"""
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from matplotlib import colormaps
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.models.slice2d import SliceClassifier2D, image_to_tensor, pet_intensity, PET_CMAPS, CLASSES

ROOT = Path(__file__).resolve().parents[1]
PET_DIR = ROOT / "data" / "real" / "pet"
MRI_INIT = ROOT / "outputs" / "checkpoints" / "slice2d_model.pt"
OUT_PATH = ROOT / "outputs" / "checkpoints" / "pet_slice2d_model.pt"
CODE_TO_CLASS = {"cn": 0, "mci": 1, "ad": 2}
EPOCHS = 40
FOLDS = 5
SEED = 42


def collect():
    items = []
    for f in sorted(PET_DIR.glob("real-pet-*-*.npy")):
        code = f.stem.split("-")[2]
        if code in CODE_TO_CLASS:
            items.append((f, CODE_TO_CLASS[code]))
    return items


def slice_image(path):
    vol = np.load(path).astype(np.float32)
    s = vol[vol.shape[0] // 2]  # these volumes are one axial slice stacked along axis 0
    lo, hi = np.percentile(s, 1), np.percentile(s, 99.5)
    return np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)


class PetDataset(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        self.slices = [slice_image(p) for p, _ in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, label = self.slices[i], self.items[i][1]
        if self.train and random.random() < 0.5:
            # Colour-map then invert, exactly as an uploaded colour PET picture is processed
            img = Image.fromarray((colormaps[random.choice(PET_CMAPS)](s)[..., :3] * 255).astype(np.uint8))
            img = pet_intensity(img)
        else:
            img = Image.fromarray((s * 255).astype(np.uint8))
        if self.train:
            w, h = img.size
            sc = random.uniform(0.85, 1.0)
            cw, ch = int(w * sc), int(h * sc)
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x0, y0, x0 + cw, y0 + ch)).rotate(random.uniform(-10, 10), resample=Image.BILINEAR)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = image_to_tensor(img)[0]
        if self.train:
            x = x * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return x, label


def train_one(train_items, device, log=None):
    model = SliceClassifier2D()
    if MRI_INIT.exists():
        model.load_state_dict(torch.load(MRI_INIT, map_location="cpu")["model_state"])
    model.to(device)
    dl = DataLoader(PetDataset(train_items, True), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * len(dl))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step(); sched.step()
            total += loss.item() * len(y)
        if log and epoch % 10 == 0:
            print(f"  epoch {epoch} loss={total / len(train_items):.4f}")
    return model.cpu().eval()


def predict(model, items):
    dl = DataLoader(PetDataset(items, False), batch_size=64)
    with torch.no_grad():
        return torch.cat([model(x).argmax(1) for x, _ in dl]).numpy()


if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    items = collect()
    print(f"[data] {len(items)} PET slices, per class={np.bincount([y for _, y in items]).tolist()}")

    # Stratified k-fold cross-validation for an honest accuracy estimate
    folds = [[] for _ in range(FOLDS)]
    for c in range(3):
        cls = [it for it in items if it[1] == c]
        random.shuffle(cls)
        for i, it in enumerate(cls):
            folds[i % FOLDS].append(it)
    cm = np.zeros((3, 3), dtype=int)
    for k in range(FOLDS):
        train_items = [it for j, f in enumerate(folds) if j != k for it in f]
        model = train_one(train_items, device)
        for (_, t), p in zip(folds[k], predict(model, folds[k])):
            cm[t, p] += 1
        print(f"fold {k + 1}: acc={np.mean([t == p for (_, t), p in zip(folds[k], predict(model, folds[k]))]):.3f}")
    cv_acc = cm.diagonal().sum() / cm.sum()
    cv_bal = (cm.diagonal() / cm.sum(1)).mean()
    print(f"\n{FOLDS}-fold CV acc={cv_acc:.3f} balanced={cv_bal:.3f}")
    print("Confusion (rows=true CN/MCI/AD, cols=pred):")
    for row in cm:
        print("  ", row.tolist())

    print("\nTraining final model on all PET slices")
    model = train_one(items, device, log=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": CLASSES, "cv_acc": float(cv_acc),
                "cv_balanced_acc": float(cv_bal), "confusion": cm.tolist()}, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")
