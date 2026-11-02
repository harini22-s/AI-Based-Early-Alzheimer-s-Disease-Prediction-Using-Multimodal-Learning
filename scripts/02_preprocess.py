#!/usr/bin/env python3
"""
Builds the manifest (train/val/test split), runs the MRI/PET preprocessing
pipeline once and caches results as .npy, and fits the clinical
preprocessor. Works identically for synthetic or real data — it just reads
whatever is in data/processed/raw_manifest.csv (written by either
01_generate_synthetic_data.py or --real-data below).

Usage:
    python scripts/02_preprocess.py               # uses existing raw_manifest.csv
    python scripts/02_preprocess.py --real-data    # (re)builds raw_manifest.csv from data/raw/*
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import set_seed, ensure_dirs
from src.data.dataset import build_raw_manifest_from_real_data, build_manifest, precompute_cache, \
    fit_clinical_preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", action="store_true",
                         help="Rebuild raw_manifest.csv from data/raw/{MRI,PET,labels.csv,clinical.csv}")
    args = parser.parse_args()

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    if args.real_data:
        build_raw_manifest_from_real_data(cfg)

    manifest = build_manifest(cfg)
    manifest = precompute_cache(cfg, manifest)
    fit_clinical_preprocessor(cfg, manifest)

    print("\nDone. Next: python scripts/03_train.py")
