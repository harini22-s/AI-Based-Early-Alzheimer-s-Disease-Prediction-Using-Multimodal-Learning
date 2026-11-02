"""
2D axial slice classifiers for picture uploads (JPG/PNG), one for MRI and one for PET.

The 3D multimodal network only understands 64^3 volumes, so a single 2D
picture fed to it is out of distribution and collapses to one class. These
small CNNs are trained directly on labeled 2D slices instead
(scripts/06_train_slice2d.py for MRI, scripts/07_train_pet_slice2d.py for PET)
and map them to CN / MCI / AD.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import colormaps
from PIL import Image

SLICE_SIZE = 128
CLASSES = ["CN", "MCI", "AD"]

# Source dataset folder -> 3-class label (CDR 0 / 0.5 / >=1)
FOLDER_TO_CLASS = {
    "No Impairment": 0,
    "Very Mild Impairment": 1,
    "Mild Impairment": 2,
    "Moderate Impairment": 2,
}


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SliceClassifier2D(nn.Module):
    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(_block(1, 32), _block(32, 64), _block(64, 128), _block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Grayscale, resize to 128x128, per-image z-score. Returns [1, 1, H, W]."""
    arr = np.asarray(img.convert("L").resize((SLICE_SIZE, SLICE_SIZE), Image.BILINEAR), dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return torch.from_numpy(arr)[None, None]


# Colour maps PET pictures are commonly rendered with
PET_CMAPS = ["jet", "turbo", "rainbow", "nipy_spectral", "hot", "gist_heat", "inferno", "plasma"]
_LUTS = {name: (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32) for name in PET_CMAPS}


def pet_intensity(img: Image.Image) -> Image.Image:
    """Undo a PET colour map: map each pixel back to its position on the best-fitting
    colour scale, giving the underlying uptake intensity. Grayscale pictures pass through."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 255)), img)
    rgb = np.asarray(img.convert("RGB").resize((SLICE_SIZE, SLICE_SIZE), Image.BILINEAR), dtype=np.float32)
    if (rgb.max(-1) - rgb.min(-1)).mean() < 12:  # essentially grayscale
        return Image.fromarray(rgb.mean(-1).astype(np.uint8))
    px = rgb.reshape(-1, 3)
    bright = px.max(-1) > 20  # ignore black background when picking the colour map
    best = None
    for lut in _LUTS.values():
        d = ((px[:, None, :] - lut[None]) ** 2).sum(-1)
        idx = d.argmin(1)
        err = d[np.arange(len(px)), idx][bright].mean() if bright.any() else 0.0
        if best is None or err < best[0]:
            best = (err, idx)
    inten = best[1].astype(np.float32)
    inten[~bright] = 0
    return Image.fromarray(inten.reshape(SLICE_SIZE, SLICE_SIZE).astype(np.uint8))


def load_slice_model(path: Path):
    if not path.exists():
        return None
    model = SliceClassifier2D()
    model.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_slice(model: SliceClassifier2D, payload: bytes, tta: bool = True, modality: str = "mri") -> np.ndarray:
    """Class probabilities [CN, MCI, AD] for raw image bytes."""
    img = Image.open(BytesIO(payload))
    x = image_to_tensor(pet_intensity(img) if modality == "pet" else img)
    probs = torch.softmax(model(x), dim=-1)
    if tta:
        probs = 0.5 * probs + 0.5 * torch.softmax(model(torch.flip(x, dims=[-1])), dim=-1)
    return probs[0].numpy()
