#!/usr/bin/env python3
"""
Runs the entire pipeline end-to-end with synthetic data:
  generate -> preprocess -> train -> evaluate -> explain

This is the fastest way to confirm the whole project works in your
environment before pointing it at real MRI/PET scans.

Usage:
    python scripts/00_run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "scripts/01_generate_synthetic_data.py",
    "scripts/02_preprocess.py",
    "scripts/03_train.py",
    "scripts/04_evaluate.py",
    "scripts/05_explain.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'=' * 70}\nRUNNING {step}\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(ROOT / step)], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\n[FAILED] {step} exited with code {result.returncode}")
            sys.exit(result.returncode)
    print("\nAll steps completed. See outputs/results and outputs/explainability.")
