"""
Synthetic multimodal data generator.

WHY THIS EXISTS
----------------
Real ADNI / OASIS-3 / OpenNeuro PET+MRI data requires a data-use agreement
and a multi-GB authenticated download that cannot happen inside a sandboxed
build environment. This module generates class-conditional, brain-shaped
3D MRI and PET volumes plus correlated clinical scores, so the *entire*
pipeline (preprocessing -> feature extraction -> fusion -> classification ->
explainability -> evaluation) can be run and verified end-to-end today.

Swap this out for real data by pointing `scripts/01_preprocess_real_data.py`
at your own folders of NIfTI files — nothing else in the pipeline changes.
See DATASETS.md for where to legally obtain real MRI+PET data.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import gaussian_filter


def _ellipsoid_mask(shape, center, radii):
    zz, yy, xx = np.indices(shape)
    val = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    return val <= 1.0


def _make_brain_mri(shape, label, rng):
    """
    label: 0=CN, 1=MCI, 2=AD
    Simulates cortical thinning + ventricle enlargement with disease severity,
    which is the real structural biomarker Grad-CAM is meant to recover.
    """
    center = np.array(shape) / 2
    brain_radii = np.array(shape) * 0.42
    brain = _ellipsoid_mask(shape, center, brain_radii).astype(np.float32)

    # cortical shell thins as disease severity increases
    shell_shrink = 1.0 - 0.10 * label
    inner_radii = brain_radii * (0.80 + 0.05 * label) * shell_shrink
    inner = _ellipsoid_mask(shape, center, inner_radii).astype(np.float32)
    cortex = brain - inner
    cortex = np.clip(cortex, 0, 1)

    # lateral ventricles enlarge with severity (a classic AD biomarker)
    vent_radii = np.array(shape) * (0.05 + 0.035 * label)
    vent_offset = np.array(shape) * 0.10
    vent_l = _ellipsoid_mask(shape, center - [0, 0, vent_offset[2]], vent_radii)
    vent_r = _ellipsoid_mask(shape, center + [0, 0, vent_offset[2]], vent_radii)
    ventricles = (vent_l | vent_r).astype(np.float32)

    vol = 0.55 * inner + 0.85 * cortex
    vol[ventricles > 0] = 0.05  # CSF-filled ventricles are dark on T1
    vol = gaussian_filter(vol, sigma=0.8)
    vol += rng.normal(0, 0.03, size=shape)  # scanner noise
    vol = np.clip(vol, 0, None)
    return vol.astype(np.float32)


def _make_brain_pet(shape, label, rng):
    """
    Simulates FDG/amyloid-style hypometabolism in temporo-parietal regions
    that deepens with disease severity — this is what Grad-CAM should light up.
    """
    center = np.array(shape) / 2
    brain_radii = np.array(shape) * 0.40
    brain = _ellipsoid_mask(shape, center, brain_radii).astype(np.float32)
    uptake = brain * (1.0 - rng.uniform(0, 0.05, size=shape))

    hypo_radii = np.array(shape) * (0.14 + 0.05 * label)
    hypo_offset = np.array(shape) * 0.18
    hypo_l = _ellipsoid_mask(shape, center - [hypo_offset[0] * 0.3, hypo_offset[1], 0], hypo_radii)
    hypo_r = _ellipsoid_mask(shape, center + [hypo_offset[0] * 0.3, hypo_offset[1], 0], hypo_radii)
    hypo_mask = (hypo_l | hypo_r).astype(np.float32)

    reduction = 0.15 * label  # 0 for CN, up to 0.30 for AD
    uptake = uptake * (1 - hypo_mask * reduction)
    uptake = gaussian_filter(uptake, sigma=1.0)
    uptake += rng.normal(0, 0.02, size=shape)
    uptake = np.clip(uptake, 0, None)
    return uptake.astype(np.float32)


def _make_clinical(label, age_rng, rng):
    """Clinical scores correlated with diagnosis, with realistic overlap/noise."""
    age = np.clip(age_rng.normal(65 + 6 * label, 7), 50, 95)
    gender = rng.choice(["Male", "Female"])
    education = np.clip(rng.normal(15 - 1.2 * label, 3), 4, 22)
    mmse = np.clip(rng.normal(29 - 5.5 * label, 2.2), 0, 30)
    cdr_choices = {0: [0, 0.5], 1: [0.5, 1.0], 2: [1.0, 2.0]}[label]
    cdr = rng.choice(cdr_choices, p=[0.7, 0.3])
    return dict(Age=round(float(age), 1), Gender=gender,
                Education=round(float(education), 1),
                MMSE=round(float(mmse), 1), CDR=float(cdr))


def generate_synthetic_dataset(cfg, out_root: Path | None = None) -> pd.DataFrame:
    """
    Writes NIfTI MRI/PET volumes + clinical.csv + labels.csv + manifest.csv
    under cfg.paths.raw_mri_dir / raw_pet_dir, following the same layout that
    real data would use, so downstream code never needs to know the data was
    synthetic.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.synthetic.n_subjects
    shape = tuple(cfg.data.image_size)
    class_names = cfg.data.class_names
    probs = cfg.synthetic.class_balance

    mri_dir = Path(cfg.paths.raw_mri_dir)
    pet_dir = Path(cfg.paths.raw_pet_dir)
    mri_dir.mkdir(parents=True, exist_ok=True)
    pet_dir.mkdir(parents=True, exist_ok=True)

    labels = rng.choice([0, 1, 2], size=n, p=probs)
    rows = []
    clinical_rows = []
    affine = np.eye(4)

    for i in range(n):
        sid = f"sub-{i+1:04d}"
        label = int(labels[i])
        has_mri = rng.random() > cfg.synthetic.missing_mri_prob
        has_pet = rng.random() > cfg.synthetic.missing_pet_prob

        mri_path = ""
        pet_path = ""
        if has_mri:
            vol = _make_brain_mri(shape, label, rng)
            mri_path = str(mri_dir / f"{sid}_T1w.nii.gz")
            nib.save(nib.Nifti1Image(vol, affine), mri_path)
        if has_pet:
            vol = _make_brain_pet(shape, label, rng)
            pet_path = str(pet_dir / f"{sid}_pet.nii.gz")
            nib.save(nib.Nifti1Image(vol, affine), pet_path)

        clin = _make_clinical(label, rng, rng)
        clinical_rows.append(dict(subject_id=sid, **clin))
        rows.append(dict(
            subject_id=sid,
            diagnosis=class_names[label],
            label=label,
            mri_path=mri_path,
            pet_path=pet_path,
            mri_available=has_mri,
            pet_available=has_pet,
        ))

    manifest = pd.DataFrame(rows)
    clinical = pd.DataFrame(clinical_rows)

    Path(cfg.paths.clinical_csv).parent.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(cfg.paths.clinical_csv, index=False)
    manifest[["subject_id", "diagnosis", "label"]].to_csv(cfg.paths.labels_csv, index=False)
    manifest.to_csv(Path(cfg.paths.processed_dir) / "raw_manifest.csv", index=False)

    print(f"[synthetic] wrote {n} subjects "
          f"({manifest.mri_available.sum()} MRI, {manifest.pet_available.sum()} PET) to "
          f"{mri_dir} / {pet_dir}")
    print(f"[synthetic] class balance: "
          f"{dict(zip(class_names, np.bincount(labels, minlength=3)))}")
    return manifest
