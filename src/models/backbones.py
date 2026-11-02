"""
Per-modality feature extractors.

- CNN3DEncoder   : a 3D convolutional encoder (used for MRI and/or PET).
                   Exposes the last conv feature map for Grad-CAM.
- ViT3DEncoder   : a 3D Vision Transformer built from 3D patch embeddings
                   (used for MRI and/or PET). Exposes patch-token attention
                   for attention-rollout style explainability.
- ClinicalEncoder: a small MLP over tabular clinical/demographic features.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class CNN3DEncoder(nn.Module):
    """Stacked Conv3d-BN-ReLU-MaxPool blocks -> global average pooled feature vector."""

    def __init__(self, in_channels=1, channels=(16, 32, 64, 128), dropout=0.1):
        super().__init__()
        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            ))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = channels[-1]

    def forward_features(self, x):
        feat_map = x
        for block in self.blocks:
            feat_map = block(feat_map)
        pooled = self.dropout(self.pool(feat_map).flatten(1))
        return pooled, feat_map  # feat_map: [B, C, d, h, w] -> used for Grad-CAM

    def forward(self, x):
        pooled, _ = self.forward_features(x)
        return pooled


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w


class ViT3DEncoder(nn.Module):
    """
    3D Vision Transformer: non-overlapping cubic patches -> linear patch
    embedding (via strided Conv3d) -> [CLS] token + learned position
    embedding -> standard pre-norm transformer encoder.
    """

    def __init__(self, in_channels=1, image_size=(64, 64, 64), patch_size=8,
                 embed_dim=128, depth=4, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        assert all(s % patch_size == 0 for s in image_size), \
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        self.grid = tuple(s // patch_size for s in image_size)
        num_patches = math.prod(self.grid)

        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward_features(self, x, return_attn=False):
        b = x.shape[0]
        patches = self.patch_embed(x)                     # [B, E, d, h, w]
        patches = patches.flatten(2).transpose(1, 2)       # [B, N, E]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed

        attn_last = None
        for i, block in enumerate(self.blocks):
            need_w = return_attn and i == len(self.blocks) - 1
            tokens, attn_w = block(tokens, need_weights=need_w)
            if need_w:
                attn_last = attn_w  # [B, N+1, N+1]

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return cls_out, patch_tokens, attn_last

    def forward(self, x):
        cls_out, _, _ = self.forward_features(x)
        return cls_out


class ClinicalEncoder(nn.Module):
    """Small MLP encoder for tabular clinical/demographic features."""

    def __init__(self, in_dim, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


def make_image_encoder(backbone: str, image_size, cfg_model):
    if backbone == "cnn":
        return CNN3DEncoder(1, tuple(cfg_model.cnn_channels), cfg_model.dropout)
    elif backbone == "transformer":
        return ViT3DEncoder(
            in_channels=1, image_size=image_size, patch_size=cfg_model.vit_patch_size,
            embed_dim=cfg_model.vit_embed_dim, depth=cfg_model.vit_depth,
            heads=cfg_model.vit_heads, mlp_ratio=cfg_model.vit_mlp_ratio,
            dropout=cfg_model.dropout,
        )
    raise ValueError(f"Unknown backbone: {backbone}")
