from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ..utils import RunningAverage, save_json


def _class_weights(dataset, num_classes, device):
    labels = dataset.df["label"].values
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    loss_meter = RunningAverage()
    correct, total = 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            mri = batch["mri"].to(device)
            pet = batch["pet"].to(device)
            clinical = batch["clinical"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(mri, pet, clinical, mask)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), n=labels.size(0))
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_meter.avg, correct / max(total, 1)


def train_model(cfg, model, train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                               num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=cfg.train.num_workers)

    weights = _class_weights(train_ds, cfg.data.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_left = cfg.train.early_stopping_patience

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[epoch {epoch:03d}/{cfg.train.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_left = cfg.train.early_stopping_patience
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_acc": val_acc}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early stopping] no val_loss improvement for "
                      f"{cfg.train.early_stopping_patience} epochs, stopping at epoch {epoch}")
                break

    save_json(history, Path(cfg.paths.results_dir) / "training_history.json")
    _plot_history(history, Path(cfg.paths.results_dir) / "training_curves.png")

    # reload best weights before returning
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[train] restored best checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.3f})")
    return model, history


def _plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
