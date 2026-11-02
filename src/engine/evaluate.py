from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from ..utils import save_json


@torch.no_grad()
def collect_predictions(model, dataset, device, batch_size=8):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_labels, all_ids = [], [], []
    for batch in loader:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        clinical = batch["clinical"].to(device)
        mask = batch["mask"].to(device)
        logits = model(mri, pet, clinical, mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
        all_ids.extend(batch["subject_id"])
    return np.concatenate(all_probs), np.concatenate(all_labels), all_ids


def _specificity_per_class(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    specs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specs.append(tn / (tn + fp + 1e-8))
    return np.array(specs)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, class_names: list) -> dict:
    preds = probs.argmax(axis=1)
    n_classes = len(class_names)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(n_classes)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    specificity = _specificity_per_class(cm)

    try:
        labels_onehot = np.eye(n_classes)[labels]
        auc_macro = roc_auc_score(labels_onehot, probs, average="macro", multi_class="ovr")
        auc_per_class = roc_auc_score(labels_onehot, probs, average=None, multi_class="ovr") \
            if n_classes > 2 else [roc_auc_score(labels_onehot[:, 1], probs[:, 1])]
    except ValueError:
        auc_macro, auc_per_class = float("nan"), [float("nan")] * n_classes

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision.mean()),
        "macro_recall_sensitivity": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_specificity": float(specificity.mean()),
        "macro_auc_roc": float(auc_macro),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall_sensitivity": float(recall[i]),
                "specificity": float(specificity[i]),
                "f1": float(f1[i]),
                "auc_roc": float(auc_per_class[i]),
                "support": int(support[i]),
            } for i in range(n_classes)
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_roc_curves(probs, labels, class_names, out_path):
    n_classes = len(class_names)
    labels_onehot = np.eye(n_classes)[labels]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
            auc = roc_auc_score(labels_onehot[:, i], probs[:, i])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC (one-vs-rest)"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_prediction_probability(probs_row, class_names, out_path, title="Prediction Probability"):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    colors = ["#3B6FA0", "#7FA8D9", "#1C3A5E"]
    ax.bar(class_names, probs_row, color=colors[: len(class_names)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate_and_report(cfg, model, dataset, device, split_name="test"):
    class_names = cfg.data.class_names
    probs, labels, ids = collect_predictions(model, dataset, device, cfg.train.batch_size)
    metrics = compute_metrics(probs, labels, class_names)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, results_dir / f"metrics_{split_name}.json")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                           results_dir / f"confusion_matrix_{split_name}.png")
    plot_roc_curves(probs, labels, class_names, results_dir / f"roc_curves_{split_name}.png")

    print(f"[{split_name}] accuracy={metrics['accuracy']:.3f}  "
          f"macro_f1={metrics['macro_f1']:.3f}  macro_auc={metrics['macro_auc_roc']:.3f}  "
          f"macro_specificity={metrics['macro_specificity']:.3f}")
    return metrics, probs, labels, ids
