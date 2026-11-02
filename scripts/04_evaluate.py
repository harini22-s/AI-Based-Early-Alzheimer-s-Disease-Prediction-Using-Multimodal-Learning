#!/usr/bin/env python3
"""
Loads the best checkpoint and reports the full metrics suite (accuracy,
precision, recall/sensitivity, specificity, F1, AUC-ROC, confusion matrix)
on the validation and test splits.

Usage:
    python scripts/04_evaluate.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.config import load_config
from src.utils import set_seed, ensure_dirs, get_device
from src.data.dataset import make_datasets
from src.models.network import MultimodalAlzheimerNet
from src.engine.evaluate import evaluate_and_report, plot_prediction_probability

if __name__ == "__main__":
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    device = get_device(cfg.train.device)

    train_ds, val_ds, test_ds = make_datasets(cfg)
    model = MultimodalAlzheimerNet(cfg).to(device)
    ckpt_path = Path(cfg.paths.checkpoint_dir) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[checkpoint] loaded epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")

    evaluate_and_report(cfg, model, val_ds, device, split_name="val")
    metrics, probs, labels, ids = evaluate_and_report(cfg, model, test_ds, device, split_name="test")

    # example: per-sample prediction-probability bar chart for the first test subject
    out_dir = Path(cfg.paths.results_dir)
    plot_prediction_probability(probs[0], cfg.data.class_names,
                                 out_dir / f"prediction_probability_{ids[0]}.png",
                                 title=f"Prediction Probability — {ids[0]}")
    print(f"\nSaved metrics + plots to {out_dir}")
