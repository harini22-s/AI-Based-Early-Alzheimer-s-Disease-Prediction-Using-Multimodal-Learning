#!/usr/bin/env python3
"""
Generates the explainability panel from the architecture diagram:
  - Important Brain Regions: Grad-CAM for CNN-backbone modalities,
    attention maps for transformer-backbone modalities.
  - Important Clinical Features: SHAP bar chart.

Usage:
    python scripts/05_explain.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.utils import set_seed, ensure_dirs, get_device
from src.data.dataset import make_datasets
from src.models.network import MultimodalAlzheimerNet
from src.explain.visual_explain import grad_cam_3d, vit_attention_map, visualize_map
from src.explain.shap_explain import compute_clinical_shap, plot_clinical_importance

if __name__ == "__main__":
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    device = get_device(cfg.train.device)

    train_ds, val_ds, test_ds = make_datasets(cfg)
    model = MultimodalAlzheimerNet(cfg).to(device)
    ckpt = torch.load(Path(cfg.paths.checkpoint_dir) / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    explain_dir = Path(cfg.paths.explain_dir)
    explain_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1) Important Brain Regions (MRI + PET), a few test-set samples
    # ---------------------------------------------------------------- #
    n_show = min(cfg.explain.n_gradcam_samples, len(test_ds))
    for idx in range(n_show):
        sample = test_ds[idx]
        mri = sample["mri"].unsqueeze(0).to(device)
        pet = sample["pet"].unsqueeze(0).to(device)
        clinical = sample["clinical"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        sid = sample["subject_id"]
        pred_class = model(mri, pet, clinical, mask).argmax(dim=1).item()
        class_name = cfg.data.class_names[pred_class]

        for modality, vol_tensor, backbone in [
            ("mri", mri, cfg.model.mri_backbone),
            ("pet", pet, cfg.model.pet_backbone),
        ]:
            if sample["mask"][{"mri": 0, "pet": 1}[modality]].item() == 0:
                continue  # skip explaining a missing modality
            if backbone == "cnn":
                cam = grad_cam_3d(model, mri, pet, clinical, mask, pred_class, modality=modality)[0]
                tag = "gradcam"
            else:
                cam = vit_attention_map(model, mri, pet, clinical, mask, modality=modality)[0]
                tag = "attnmap"

            volume = vol_tensor[0, 0].detach().cpu().numpy()
            out_path = explain_dir / f"{tag}_{modality}_{sid}_pred-{class_name}.png"
            visualize_map(volume, cam, f"{modality.upper()} {tag} — {sid} (pred: {class_name})", out_path)
            print(f"[explain] wrote {out_path}")

    # ---------------------------------------------------------------- #
    # 2) Important Clinical Features (SHAP)
    # ---------------------------------------------------------------- #
    background = train_ds._clinical_matrix
    sample_matrix = test_ds._clinical_matrix
    feature_names = train_ds.clinical_pp.selected_features_

    importance, _ = compute_clinical_shap(
        model, tuple(cfg.data.image_size), device, feature_names,
        background_matrix=background, sample_matrix=sample_matrix,
        n_background=cfg.explain.n_shap_background, n_samples=cfg.explain.n_shap_samples,
    )
    out_path = explain_dir / "shap_clinical_importance.png"
    plot_clinical_importance(feature_names, importance, out_path)
    print(f"[explain] wrote {out_path}")

    print(f"\nAll explainability artifacts saved to {explain_dir}")
