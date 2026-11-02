#!/usr/bin/env python3
"""
Generates a synthetic multimodal MRI+PET+clinical dataset so the full
pipeline can run today. SKIP this script once you have real data —
just populate data/raw/MRI, data/raw/PET, data/raw/clinical.csv and
data/raw/labels.csv yourself (see DATASETS.md), and go straight to
02_preprocess.py.

Usage:
    python scripts/01_generate_synthetic_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import set_seed, ensure_dirs
from src.data.synthetic import generate_synthetic_dataset

if __name__ == "__main__":
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    generate_synthetic_dataset(cfg)
    print("\nDone. Next: python scripts/02_preprocess.py")
