"""
Post-hoc analysis modules used by the dashboard's live prediction:

- CMCAD_Gate: cross-modal cross-attention with learned residual gates (MRI <-> PET).
- MVDICalculator: metabolic-volumetric dissociation index from MRI/PET volumes.
- EvidentialUncertaintyCalibrator: Dirichlet evidential uncertainty + deferral flag.
- NeuroSymbolicClinicalCopilot: rule-based rationale combining model output and clinical scores.

All public helpers return plain Python types so results are JSON-serialisable.
"""

import numpy as np
import torch
import torch.nn as nn


class CMCAD_Gate(nn.Module):
    """H_fused = LN(H_mri + g_mri * CrossAttn(H_mri, H_pet)) || LN(H_pet + g_pet * CrossAttn(H_pet, H_mri))"""

    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.attn_mri = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_pet = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate_mri = nn.Sequential(nn.Linear(2 * dim, 1), nn.Sigmoid())
        self.gate_pet = nn.Sequential(nn.Linear(2 * dim, 1), nn.Sigmoid())
        self.norm_mri = nn.LayerNorm(dim)
        self.norm_pet = nn.LayerNorm(dim)

    def forward(self, h_mri, h_pet):
        # h_*: [B, dim] -> treated as single-token sequences
        q_m, q_p = h_mri.unsqueeze(1), h_pet.unsqueeze(1)
        a_m, _ = self.attn_mri(q_m, q_p, q_p)
        a_p, _ = self.attn_pet(q_p, q_m, q_m)
        a_m, a_p = a_m.squeeze(1), a_p.squeeze(1)
        g_m = self.gate_mri(torch.cat([h_mri, a_m], dim=-1))
        g_p = self.gate_pet(torch.cat([h_pet, a_p], dim=-1))
        fused = torch.cat([self.norm_mri(h_mri + g_m * a_m), self.norm_pet(h_pet + g_p * a_p)], dim=-1)
        return fused, {"mri_gate": g_m.squeeze(-1), "pet_gate": g_p.squeeze(-1)}


class MVDICalculator:
    """MVDI = relative metabolic deficit - relative volumetric deficit, per region."""

    # Fractional slab boundaries along the axial axis (D) for coarse regions
    REGIONS = {
        "Mesial Temporal": (0.30, 0.50),
        "Temporoparietal": (0.50, 0.70),
        "Posterior Cingulate": (0.60, 0.80),
    }

    @staticmethod
    def _deficit(vol, lo, hi):
        d = vol.shape[0]
        region = vol[int(lo * d):int(hi * d)]
        brain = vol[vol > vol.min()] if np.any(vol > vol.min()) else vol
        ref = float(np.mean(np.abs(brain))) + 1e-6
        # Higher score = lower signal in the region relative to whole brain
        return float(np.clip(1.0 - np.mean(np.abs(region)) / ref, -1.0, 1.0))

    @classmethod
    def compute_mvdi(cls, mri_vol, pet_vol, clin):
        has_mri = mri_vol is not None and float(np.abs(mri_vol).sum()) > 0
        has_pet = pet_vol is not None and float(np.abs(pet_vol).sum()) > 0
        regions = []
        for name, (lo, hi) in cls.REGIONS.items():
            vol_def = cls._deficit(mri_vol, lo, hi) if has_mri else 0.0
            met_def = cls._deficit(pet_vol, lo, hi) if has_pet else 0.0
            regions.append({
                "region": name,
                "volumetric_deficit": round(vol_def, 3),
                "metabolic_deficit": round(met_def, 3),
                "mvdi": round(met_def - vol_def, 3),
            })
        global_mvdi = float(np.mean([r["mvdi"] for r in regions]))
        # Clinical prior: low MMSE / higher CDR raises concern for dissociation significance
        mmse = float(clin.get("mmse", 30))
        cdr = float(clin.get("cdr", 0))
        if not (has_mri and has_pet):
            interp = "MVDI requires both MRI and PET; one modality is missing."
            status = "Not Computable"
        elif global_mvdi > 0.05:
            interp = ("Metabolic deficit exceeds structural atrophy — functional decline may precede "
                      "visible atrophy (prodromal pattern).")
            status = "Metabolic-Predominant"
        elif global_mvdi < -0.05:
            interp = "Structural atrophy exceeds metabolic deficit — consistent with established neurodegeneration."
            status = "Structural-Predominant"
        else:
            interp = "Metabolic and structural changes are concordant."
            status = "Concordant"
        return {
            "global_mvdi": round(global_mvdi, 3),
            "status": status,
            "regions": regions,
            "interpretation": interp,
            "clinical_prior": {"mmse": mmse, "cdr": cdr},
        }


class EvidentialUncertaintyCalibrator:
    """e_k = max(0, f_k), alpha_k = e_k + 1, S = sum(alpha), u = K / S."""

    DEFER_THRESHOLD = 0.5

    @classmethod
    def calibrate_evidence(cls, logits, num_classes=3):
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)[:num_classes]
        # Shift so the weakest class has zero evidence; scale to a usable evidence range
        evidence = np.maximum(0.0, (logits - logits.min()) * 10.0)
        alpha = evidence + 1.0
        s = float(alpha.sum())
        probs = alpha / s
        uncertainty = num_classes / s
        defer = bool(uncertainty > cls.DEFER_THRESHOLD)
        return {
            "evidence": [round(float(e), 3) for e in evidence],
            "alpha": [round(float(a), 3) for a in alpha],
            "dirichlet_strength": round(s, 3),
            "expected_probabilities": [round(float(p), 4) for p in probs],
            "epistemic_uncertainty": round(float(uncertainty), 4),
            "uncertainty_pct": round(float(uncertainty) * 100, 1),
            "defer_to_specialist": defer,
            "recommendation": ("High uncertainty — defer to neurologist review and gather additional biomarkers."
                               if defer else "Evidence is sufficient for decision support."),
        }


class NeuroSymbolicClinicalCopilot:
    """Combines the network's staging with symbolic NIA-AA-style clinical rules."""

    @staticmethod
    def generate_symbolic_rationale(pred_code, confidence, clin, mvdi, edl):
        mmse = float(clin.get("mmse", 30))
        cdr = float(clin.get("cdr", 0))
        rules = []
        if mmse >= 25 and cdr == 0:
            clinical_stage = "CN"
            rules.append(f"MMSE {mmse:.1f} ≥ 25 and CDR 0 → clinically unimpaired.")
        elif mmse >= 18 and cdr <= 0.5:
            clinical_stage = "MCI"
            rules.append(f"MMSE {mmse:.1f} in 18–24 or CDR 0.5 → mild cognitive impairment range.")
        else:
            clinical_stage = "AD"
            rules.append(f"MMSE {mmse:.1f} < 18 or CDR {cdr:.1f} ≥ 1 → dementia range.")

        if mvdi.get("status") == "Metabolic-Predominant":
            rules.append("PET deficit exceeds MRI atrophy → supports early/prodromal neurodegeneration.")
        elif mvdi.get("status") == "Structural-Predominant":
            rules.append("MRI atrophy exceeds PET deficit → supports established neurodegeneration.")

        if edl.get("defer_to_specialist"):
            rules.append(f"Evidential uncertainty {edl.get('uncertainty_pct')}% exceeds threshold → specialist deferral.")

        agrees = clinical_stage == pred_code
        rules.append(
            f"Neural staging ({pred_code}) {'agrees' if agrees else 'disagrees'} with symbolic clinical staging ({clinical_stage})."
        )
        if agrees and not edl.get("defer_to_specialist"):
            verdict = "Concordant — neural and symbolic evidence align."
        elif agrees:
            verdict = "Concordant but uncertain — confirm with additional biomarkers."
        else:
            verdict = "Discordant — requires clinician adjudication."
        return {
            "neural_stage": pred_code,
            "symbolic_stage": clinical_stage,
            "concordant": agrees,
            "confidence_pct": round(float(confidence) * 100, 1),
            "rules_fired": rules,
            "verdict": verdict,
        }
