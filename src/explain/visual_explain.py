"""
"Important Brain Regions" explainability, matching the architecture diagram's
Grad-CAM panel — implemented for both backbone types actually used in this
project:

  - CNN backbone  -> true Grad-CAM (gradient-weighted class activation map
                     on the last conv feature map).
  - Transformer   -> CLS-token attention map from the last block, reshaped
    backbone         back into a 3D grid (a lightweight, single-layer stand-in
                     for full attention-rollout; extending to multi-layer
                     rollout is a straightforward loop over `model.*_encoder.blocks`).

Both return a low-resolution 3D saliency map that `visualize_map` upsamples
and overlays on axial/coronal/sagittal mid-slices of the input volume.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    denom = x.max() + 1e-8
    return x / denom


def grad_cam_3d(model, mri, pet, clinical, mask, target_class, modality="mri"):
    """Returns a [B, d, h, w] low-res CAM (pre-upsampling). `modality` must use a CNN backbone."""
    model.eval()
    backbone_type = getattr(model, f"{modality}_backbone_type")
    if backbone_type != "cnn":
        raise ValueError(f"grad_cam_3d requires a CNN backbone for '{modality}', got '{backbone_type}'. "
                          f"Use vit_attention_map for transformer backbones.")

    logits, extra = model(mri, pet, clinical, mask, return_intermediates=True)
    feat_map = extra[modality]["feat_map"]
    feat_map.retain_grad()

    target_class = torch.as_tensor(target_class, device=logits.device)
    if target_class.ndim == 0:
        target_class = target_class.expand(logits.shape[0])
    score = logits.gather(1, target_class.view(-1, 1)).sum()

    model.zero_grad(set_to_none=True)
    score.backward()

    grads = feat_map.grad                                  # [B, C, d, h, w]
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)       # global-average-pooled gradients
    cam = torch.relu((weights * feat_map).sum(dim=1))       # [B, d, h, w]
    return cam.detach().cpu().numpy()


def vit_attention_map(model, mri, pet, clinical, mask, modality="pet"):
    """Returns a [B, d, h, w] low-res attention map for a transformer-backbone modality."""
    model.eval()
    backbone_type = getattr(model, f"{modality}_backbone_type")
    if backbone_type != "transformer":
        raise ValueError(f"vit_attention_map requires a transformer backbone for '{modality}', "
                          f"got '{backbone_type}'. Use grad_cam_3d for CNN backbones.")

    with torch.no_grad():
        _, extra = model(mri, pet, clinical, mask, return_intermediates=True)
    attn = extra[modality]["attn"]          # [B, N+1, N+1], averaged over heads
    grid = extra[modality]["grid"]          # (gd, gh, gw)
    cls_to_patches = attn[:, 0, 1:]         # [B, N]  attention FROM the [CLS] token
    b = cls_to_patches.shape[0]
    maps = cls_to_patches.reshape(b, *grid).detach().cpu().numpy()
    return maps


def upsample_map(low_res_map: np.ndarray, target_shape) -> np.ndarray:
    zoom_factors = [t / s for t, s in zip(target_shape, low_res_map.shape)]
    return zoom(low_res_map, zoom_factors, order=1)


def visualize_map(volume: np.ndarray, heatmap: np.ndarray, title: str, out_path: str):
    """Overlay a saliency heatmap on axial / coronal / sagittal mid-slices and save a PNG."""
    heatmap = _normalize(upsample_map(heatmap, volume.shape))
    d, h, w = volume.shape
    slices = [
        ("Axial", volume[d // 2, :, :], heatmap[d // 2, :, :]),
        ("Coronal", volume[:, h // 2, :], heatmap[:, h // 2, :]),
        ("Sagittal", volume[:, :, w // 2], heatmap[:, :, w // 2]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, (name, base, hm) in zip(axes, slices):
        ax.imshow(base, cmap="gray")
        ax.imshow(hm, cmap="jet", alpha=0.45)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
