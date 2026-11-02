"""
End-to-end model: per-modality encoders -> attention fusion -> FC classifier.

forward(..., return_intermediates=True) also hands back everything the
explainability module needs (last conv feature maps for Grad-CAM, ViT
attention maps, and fusion attention weights) without any extra passes.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .backbones import make_image_encoder, ClinicalEncoder
from .fusion import AttentionFusion


class MultimodalAlzheimerNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        image_size = tuple(cfg.data.image_size)
        n_clinical = len(cfg.data.clinical_features)

        self.mri_backbone_type = m.mri_backbone
        self.pet_backbone_type = m.pet_backbone

        self.mri_encoder = make_image_encoder(m.mri_backbone, image_size, m)
        self.pet_encoder = make_image_encoder(m.pet_backbone, image_size, m)
        self.clinical_encoder = ClinicalEncoder(n_clinical, tuple(m.clinical_hidden), m.dropout)

        input_dims = {
            "mri": self.mri_encoder.out_dim,
            "pet": self.pet_encoder.out_dim,
            "clinical": self.clinical_encoder.out_dim,
        }
        self.fusion = AttentionFusion(input_dims, m.fusion_dim, m.fusion_heads, m.dropout)

        head = []
        d = m.fusion_dim
        for h in m.classifier_hidden:
            head += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(m.dropout)]
            d = h
        head.append(nn.Linear(d, cfg.data.num_classes))
        self.classifier = nn.Sequential(*head)

    def _encode_image(self, x, encoder, backbone_type, return_attn):
        if backbone_type == "cnn":
            pooled, feat_map = encoder.forward_features(x)
            return pooled, {"type": "cnn", "feat_map": feat_map}
        else:
            cls_out, patch_tokens, attn = encoder.forward_features(x, return_attn=return_attn)
            return cls_out, {"type": "transformer", "patch_tokens": patch_tokens,
                              "attn": attn, "grid": encoder.grid}

    def forward(self, mri, pet, clinical, mask, return_intermediates=False):
        mri_feat, mri_extra = self._encode_image(mri, self.mri_encoder, self.mri_backbone_type,
                                                   return_intermediates)
        pet_feat, pet_extra = self._encode_image(pet, self.pet_encoder, self.pet_backbone_type,
                                                   return_intermediates)
        clin_feat = self.clinical_encoder(clinical)

        fused, attn_weights, modality_names = self.fusion(
            {"mri": mri_feat, "pet": pet_feat, "clinical": clin_feat}, mask
        )
        logits = self.classifier(fused)

        if not return_intermediates:
            return logits

        return logits, {
            "fusion_weights": attn_weights,        # [B, 3] relative modality importance
            "modality_names": modality_names,
            "mri": mri_extra,
            "pet": pet_extra,
            "fused_features": fused,
        }
