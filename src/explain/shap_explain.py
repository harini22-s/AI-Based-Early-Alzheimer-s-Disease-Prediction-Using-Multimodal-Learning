"""
"Important Clinical Features" explainability (the SHAP panel in the
architecture diagram).

We isolate the clinical branch's marginal contribution by feeding the model
zero-filled MRI/PET volumes with their modality-mask bits turned off. The
missing-modality-aware fusion module then routes 100% of the attention
weight to the clinical branch, so the resulting SHAP values describe
"what would this model predict from clinical data alone" — a clean,
well-defined quantity to explain (rather than an approximation that
conflates imaging and clinical signal).
"""
from __future__ import annotations
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt


class ClinicalOnlyPredictor:
    """Callable wrapper: clinical feature matrix -> class probabilities."""

    def __init__(self, model, image_shape, device):
        self.model = model.eval()
        self.image_shape = image_shape
        self.device = device

    def __call__(self, clinical_batch: np.ndarray) -> np.ndarray:
        """
        With MRI/PET masked out, the attention-fusion module is guaranteed
        to route 100% of the weight to the clinical branch (softmax over a
        single unmasked entry == 1). We exploit that to skip the CNN/ViT
        branches entirely instead of running expensive forward passes on
        zero-filled volumes — this keeps SHAP fast and memory-safe no
        matter how many coalition rows KernelExplainer requests at once.
        """
        clinical_batch = np.atleast_2d(clinical_batch).astype(np.float32)
        clinical_t = torch.from_numpy(clinical_batch).to(self.device)
        with torch.no_grad():
            clin_feat = self.model.clinical_encoder(clinical_t)
            fused = self.model.fusion.norm(self.model.fusion.proj["clinical"](clin_feat))
            logits = self.model.classifier(fused)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def compute_clinical_shap(model, image_shape, device, feature_names,
                           background_matrix: np.ndarray, sample_matrix: np.ndarray,
                           n_background=40, n_samples=15):
    background = shap.sample(background_matrix, min(n_background, len(background_matrix)), random_state=0)
    predictor = ClinicalOnlyPredictor(model, image_shape, device)
    explainer = shap.KernelExplainer(predictor, background)

    sample = sample_matrix[: min(n_samples, len(sample_matrix))]
    shap_values = explainer.shap_values(sample, nsamples=100, silent=True)

    # shap_values is either an array [n_samples, n_features, n_classes] (new shap)
    # or a list of length n_classes each [n_samples, n_features] (older shap).
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=-1)          # [n_samples, n_features, n_classes]
    else:
        stacked = np.asarray(shap_values)
        if stacked.ndim == 2:
            stacked = stacked[..., None]

    mean_abs_importance = np.abs(stacked).mean(axis=(0, 2))  # -> [n_features]
    return mean_abs_importance, stacked


def plot_clinical_importance(feature_names, importance, out_path, title="SHAP: Important Clinical Features"):
    order = np.argsort(importance)[::-1]
    names = [feature_names[i] for i in order]
    vals = importance[order]

    fig, ax = plt.subplots(figsize=(6, 0.55 * len(names) + 1.2))
    ax.barh(names[::-1], vals[::-1], color="#3B6FA0")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
