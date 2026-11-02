"""
Preprocessing steps mirroring the project's system-architecture diagram:

  MRI  : skull stripping -> registration -> resizing -> normalization -> augmentation
  PET  : co-registration -> resizing -> smoothing -> normalization -> augmentation

Notes on scope
--------------
Clinical-grade skull stripping / inter-subject registration normally uses
FSL BET, ANTs, or FreeSurfer. Those are large, separately-licensed C++
toolkits that can't be pulled in here. This module implements lightweight,
dependency-free equivalents (Otsu threshold + largest-connected-component
for skull stripping; affine resampling to a fixed grid for "registration")
that are good enough for a research/course-project pipeline and for
validating the whole system end-to-end. Swap in `nipype` wrappers around
real FSL/ANTs binaries for a thesis-grade pipeline — the function
signatures below (`skull_strip`, `register_to_template`) are the seams to
replace.
"""
from __future__ import annotations
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_nifti(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return np.nan_to_num(data)


# --------------------------------------------------------------------------- #
# MRI-specific
# --------------------------------------------------------------------------- #
def skull_strip(volume: np.ndarray) -> np.ndarray:
    """Otsu threshold + largest connected component -> approximate brain mask."""
    v = volume.copy()
    if v.max() <= 0:
        return v
    thresh = threshold_otsu(v[v > 0]) if np.any(v > 0) else 0
    mask = v > (0.3 * thresh)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    labeled = cc_label(mask)
    if labeled.max() > 0:
        biggest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        mask = labeled == biggest
    return v * mask


def register_to_template(volume: np.ndarray, target_shape) -> np.ndarray:
    """Isotropic resample to a fixed grid size ('registration' to a common template space)."""
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, zoom_factors, order=1)


# --------------------------------------------------------------------------- #
# PET-specific
# --------------------------------------------------------------------------- #
def coregister_pet_to_mri(pet_volume: np.ndarray, mri_shape) -> np.ndarray:
    """Resample PET onto the MRI's voxel grid (stand-in for real rigid coregistration)."""
    return register_to_template(pet_volume, mri_shape)


def smooth_pet(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return ndimage.gaussian_filter(volume, sigma=sigma)


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
def resize_volume(volume: np.ndarray, target_shape) -> np.ndarray:
    if tuple(volume.shape) == tuple(target_shape):
        return volume
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, zoom_factors, order=1)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if mask.sum() < 10:
        return volume
    mu, sigma = volume[mask].mean(), volume[mask].std() + 1e-8
    out = (volume - mu) / sigma
    out[~mask] = out[mask].min()
    return out.astype(np.float32)


def preprocess_mri(path: str, target_shape) -> np.ndarray:
    vol = load_nifti(path)
    vol = skull_strip(vol)
    vol = register_to_template(vol, target_shape)
    vol = zscore_normalize(vol)
    return vol.astype(np.float32)


def preprocess_pet(path: str, target_shape) -> np.ndarray:
    vol = load_nifti(path)
    vol = coregister_pet_to_mri(vol, target_shape)
    vol = smooth_pet(vol, sigma=1.0)
    vol = zscore_normalize(vol)
    return vol.astype(np.float32)


# --------------------------------------------------------------------------- #
# Augmentation (applied on already-preprocessed, fixed-size volumes)
# --------------------------------------------------------------------------- #
def augment_volume(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vol = volume.copy()

    # random axis flips
    for axis in range(3):
        if rng.random() < 0.5:
            vol = np.flip(vol, axis=axis)

    # small random rotation in a random plane
    if rng.random() < 0.5:
        angle = rng.uniform(-8, 8)
        axes = tuple(rng.choice([0, 1, 2], size=2, replace=False))
        vol = ndimage.rotate(vol, angle, axes=axes, reshape=False, order=1, mode="nearest")

    # intensity jitter
    if rng.random() < 0.5:
        vol = vol * rng.uniform(0.95, 1.05) + rng.uniform(-0.05, 0.05)

    # gaussian noise
    if rng.random() < 0.3:
        vol = vol + rng.normal(0, 0.02, size=vol.shape)

    return np.ascontiguousarray(vol.astype(np.float32))
