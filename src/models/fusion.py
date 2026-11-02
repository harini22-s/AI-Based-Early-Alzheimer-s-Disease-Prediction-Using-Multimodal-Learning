"""
Attention-based multimodal fusion with explicit missing-modality masking.

Each modality's feature vector is projected into a shared space, then a
small multi-head scoring network learns how much to trust each modality
per-sample. Missing modalities are masked to -inf *before* the softmax so a
zero-filled placeholder volume can never be mistaken for real signal — this
is the "Modality Mask" -> "Attention-Based Fusion" path in the architecture
diagram.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, input_dims: dict, fusion_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.modality_names = list(input_dims.keys())
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim) for name, dim in input_dims.items()
        })
        self.norm = nn.LayerNorm(fusion_dim)
        self.score_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(fusion_dim, fusion_dim // 2), nn.Tanh(),
                          nn.Linear(fusion_dim // 2, 1))
            for _ in range(heads)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: dict, mask: torch.Tensor):
        """
        features: {name: [B, dim]}
        mask:     [B, M]  (M in the same order as self.modality_names), 1=present, 0=missing
        returns:  fused [B, fusion_dim], weights [B, M] (averaged across heads), modality_names
        """
        stacked = torch.stack(
            [self.norm(self.proj[name](features[name])) for name in self.modality_names], dim=1
        )  # [B, M, D]

        neg_inf = torch.finfo(stacked.dtype).min
        head_weights = []
        for head in self.score_heads:
            scores = head(stacked).squeeze(-1)               # [B, M]
            scores = scores.masked_fill(mask <= 0, neg_inf)
            head_weights.append(torch.softmax(scores, dim=1))
        weights = torch.stack(head_weights, dim=0).mean(0)    # [B, M]

        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # [B, D]
        fused = self.dropout(fused)
        return fused, weights, self.modality_names
