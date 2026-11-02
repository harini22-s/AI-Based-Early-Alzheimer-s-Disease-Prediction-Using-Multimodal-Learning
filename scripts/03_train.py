#!/usr/bin/env python3
"""
Trains the attention-fusion multimodal network (3D CNN + 3D Transformer +
clinical MLP -> attention fusion -> classifier).

Usage:
    python scripts/03_train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import set_seed, ensure_dirs, get_device
from src.data.dataset import make_datasets
from src.models.network import MultimodalAlzheimerNet
from src.engine.train import train_model

if __name__ == "__main__":
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    device = get_device(cfg.train.device)
    print(f"[device] {device}")

    train_ds, val_ds, test_ds = make_datasets(cfg)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model = MultimodalAlzheimerNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] mri_backbone={cfg.model.mri_backbone} pet_backbone={cfg.model.pet_backbone} "
          f"params={n_params:,}")

    model, history = train_model(cfg, model, train_ds, val_ds, device)
    print("\nDone. Next: python scripts/04_evaluate.py")
