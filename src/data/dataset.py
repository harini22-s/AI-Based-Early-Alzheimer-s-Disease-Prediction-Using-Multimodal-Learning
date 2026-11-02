"""
Turns raw files (or the synthetic generator's output) into:
  1) a single manifest.csv with train/val/test splits
  2) cached preprocessed .npy volumes (so preprocessing runs once, not every epoch)
  3) a PyTorch Dataset that serves (MRI, PET, clinical, modality_mask, label)

Missing-modality handling: if a subject has no MRI or no PET, we feed a
zero volume for that branch and set its mask bit to 0. The fusion module
(src/models/fusion.py) uses that mask to exclude the modality from
attention instead of letting a zero-vector silently masquerade as data.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import preprocessing as prep
from .clinical import ClinicalPreprocessor


def build_raw_manifest_from_real_data(cfg) -> pd.DataFrame:
    """
    For REAL data: expects
      data/raw/labels.csv    -> columns [subject_id, diagnosis]  (diagnosis in cfg.data.class_names)
      data/raw/clinical.csv  -> columns [subject_id, <cfg.data.clinical_features...>]
      data/raw/MRI/          -> any file whose name starts with the subject_id (.nii/.nii.gz)
      data/raw/PET/          -> same, in data/raw/PET/
    and writes data/processed/raw_manifest.csv in the same schema the
    synthetic generator produces, so every downstream script is identical
    whether the data is real or synthetic.
    """
    labels = pd.read_csv(cfg.paths.labels_csv)
    name_to_id = {name: i for i, name in enumerate(cfg.data.class_names)}
    labels["label"] = labels["diagnosis"].map(name_to_id)
    if labels["label"].isna().any():
        bad = labels.loc[labels["label"].isna(), "diagnosis"].unique()
        raise ValueError(f"labels.csv has diagnosis values not in cfg.data.class_names: {bad}")

    mri_dir, pet_dir = Path(cfg.paths.raw_mri_dir), Path(cfg.paths.raw_pet_dir)
    mri_files = {f.name: f for f in mri_dir.glob("*") if f.suffix in (".gz", ".nii")}
    pet_files = {f.name: f for f in pet_dir.glob("*") if f.suffix in (".gz", ".nii")}

    def find_file(files_by_name, subject_id):
        matches = [f for name, f in files_by_name.items() if name.startswith(subject_id)]
        return str(matches[0]) if matches else ""

    rows = []
    for _, row in labels.iterrows():
        sid = row["subject_id"]
        mri_path = find_file(mri_files, sid)
        pet_path = find_file(pet_files, sid)
        rows.append(dict(
            subject_id=sid, diagnosis=row["diagnosis"], label=int(row["label"]),
            mri_path=mri_path, pet_path=pet_path,
            mri_available=bool(mri_path), pet_available=bool(pet_path),
        ))
    manifest = pd.DataFrame(rows)
    Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(cfg.paths.processed_dir) / "raw_manifest.csv", index=False)
    print(f"[real data] matched {manifest.mri_available.sum()}/{len(manifest)} MRI, "
          f"{manifest.pet_available.sum()}/{len(manifest)} PET scans")
    return manifest


def build_manifest(cfg) -> pd.DataFrame:
    raw_manifest_path = Path(cfg.paths.processed_dir) / "raw_manifest.csv"
    manifest = pd.read_csv(raw_manifest_path)
    clinical = pd.read_csv(cfg.paths.clinical_csv)
    manifest = manifest.merge(clinical, on="subject_id", how="left")

    train_val, test = train_test_split(
        manifest, test_size=cfg.data.test_split, random_state=cfg.seed,
        stratify=manifest["label"],
    )
    val_frac_of_remainder = cfg.data.val_split / (1 - cfg.data.test_split)
    train, val = train_test_split(
        train_val, test_size=val_frac_of_remainder, random_state=cfg.seed,
        stratify=train_val["label"],
    )
    manifest["split"] = "train"
    manifest.loc[val.index, "split"] = "val"
    manifest.loc[test.index, "split"] = "test"

    manifest.to_csv(cfg.paths.manifest_csv, index=False)
    print(f"[manifest] train={len(train)} val={len(val)} test={len(test)} "
          f"-> {cfg.paths.manifest_csv}")
    return manifest


def precompute_cache(cfg, manifest: pd.DataFrame) -> pd.DataFrame:
    """Runs the MRI/PET preprocessing pipeline once and caches results as .npy."""
    target_shape = tuple(cfg.data.image_size)
    mri_cache_dir = Path(cfg.paths.processed_dir) / "MRI"
    pet_cache_dir = Path(cfg.paths.processed_dir) / "PET"
    mri_cache_dir.mkdir(parents=True, exist_ok=True)
    pet_cache_dir.mkdir(parents=True, exist_ok=True)

    mri_cache_col, pet_cache_col = [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="preprocessing"):
        sid = row["subject_id"]

        mri_cache = mri_cache_dir / f"{sid}.npy"
        if bool(row["mri_available"]) and row["mri_path"] and str(row["mri_path"]) != "nan":
            if not mri_cache.exists():
                vol = prep.preprocess_mri(row["mri_path"], target_shape)
                np.save(mri_cache, vol)
            mri_cache_col.append(str(mri_cache))
        else:
            mri_cache_col.append("")

        pet_cache = pet_cache_dir / f"{sid}.npy"
        if bool(row["pet_available"]) and row["pet_path"] and str(row["pet_path"]) != "nan":
            if not pet_cache.exists():
                vol = prep.preprocess_pet(row["pet_path"], target_shape)
                np.save(pet_cache, vol)
            pet_cache_col.append(str(pet_cache))
        else:
            pet_cache_col.append("")

    manifest["mri_cache"] = mri_cache_col
    manifest["pet_cache"] = pet_cache_col
    manifest.to_csv(cfg.paths.manifest_csv, index=False)
    return manifest


def fit_clinical_preprocessor(cfg, manifest: pd.DataFrame) -> ClinicalPreprocessor:
    numeric = [c for c in cfg.data.clinical_features if c not in cfg.data.categorical_features]
    categorical = list(cfg.data.categorical_features)
    train_df = manifest[manifest.split == "train"]
    pp = ClinicalPreprocessor(numeric, categorical)
    pp.fit(train_df[numeric + categorical], train_df["label"].values,
           top_k=len(numeric + categorical))
    pp.save(Path(cfg.paths.processed_dir) / "clinical_preprocessor.json")
    return pp


class MultimodalAlzheimerDataset(Dataset):
    """
    Returns a dict per sample:
      mri: FloatTensor [1, D, H, W]
      pet: FloatTensor [1, D, H, W]
      clinical: FloatTensor [F]
      mask: FloatTensor [3]   -> (mri_present, pet_present, clinical_present)
      label: LongTensor []
      subject_id: str
    """

    def __init__(self, manifest: pd.DataFrame, split: str, cfg,
                 clinical_pp: ClinicalPreprocessor, train_mode: bool | None = None):
        self.df = manifest[manifest.split == split].reset_index(drop=True)
        self.cfg = cfg
        self.shape = tuple(cfg.data.image_size)
        self.clinical_pp = clinical_pp
        self.train_mode = (split == "train") if train_mode is None else train_mode
        self.augment = bool(cfg.data.augment) and self.train_mode
        self.rng = np.random.default_rng(cfg.seed)

        clin_cols = clinical_pp.numeric_features + clinical_pp.categorical_features
        self._clinical_matrix = clinical_pp.transform(self.df[clin_cols])

    def __len__(self):
        return len(self.df)

    def _load_volume(self, cache_path) -> tuple[np.ndarray, float]:
        if cache_path is None or (isinstance(cache_path, float) and np.isnan(cache_path)):
            cache_path = ""
        cache_path = str(cache_path).replace("\\", "/")  # manifest may hold Windows-style paths
        if cache_path and cache_path.lower() != "nan" and Path(cache_path).exists():
            vol = np.load(cache_path)
            if vol.shape != self.shape:
                vol = prep.resize_volume(vol, self.shape)
            if self.augment:
                vol = prep.augment_volume(vol, self.rng)
            return vol.astype(np.float32), 1.0
        return np.zeros(self.shape, dtype=np.float32), 0.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mri, mri_mask = self._load_volume(row.get("mri_cache", ""))
        pet, pet_mask = self._load_volume(row.get("pet_cache", ""))
        clinical = self._clinical_matrix[idx]

        return {
            "mri": torch.from_numpy(mri).unsqueeze(0),
            "pet": torch.from_numpy(pet).unsqueeze(0),
            "clinical": torch.from_numpy(clinical),
            "mask": torch.tensor([mri_mask, pet_mask, 1.0], dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": row["subject_id"],
        }


def make_datasets(cfg):
    manifest = pd.read_csv(cfg.paths.manifest_csv)
    clinical_pp = ClinicalPreprocessor.load(Path(cfg.paths.processed_dir) / "clinical_preprocessor.json")
    train_ds = MultimodalAlzheimerDataset(manifest, "train", cfg, clinical_pp)
    val_ds = MultimodalAlzheimerDataset(manifest, "val", cfg, clinical_pp)
    test_ds = MultimodalAlzheimerDataset(manifest, "test", cfg, clinical_pp)
    return train_ds, val_ds, test_ds
