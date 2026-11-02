# AI-Based Early Alzheimer's Disease Prediction Using Multimodal Learning

An end-to-end multimodal deep learning pipeline that fuses **MRI**, **PET**,
and **clinical/demographic** data to classify subjects as **Cognitively
Normal (CN)**, **Mild Cognitive Impairment (MCI)**, or **Alzheimer's Disease
(AD)** — implementing the architecture in `docs/system_architecture.png` and
`docs/pipeline_overview.png`:

MRI/PET/Clinical → preprocessing → per-modality feature extraction
(3D CNN **and** 3D Vision Transformer) → missing-modality-aware
attention fusion → classifier → Grad-CAM / attention-map / SHAP
explainability → full evaluation suite.

## Read this first — about the data

Real ADNI / OASIS-3 / OpenNeuro PET+MRI datasets require signing a data-use
agreement and downloading tens of gigabytes from an authenticated portal —
something no automated build environment can do on your behalf. So this
project ships two ways to get data, and the **code is identical either way**:

1. **Synthetic generator (works today, no downloads).** `scripts/01_generate_synthetic_data.py`
   creates 600 class-conditional, brain-shaped MRI + PET volumes (with simulated
   cortical thinning / ventricle enlargement / hypometabolism that scales with
   disease severity) plus correlated clinical scores. This proves the whole
   pipeline end-to-end and is what the default config points at.
2. **Real data.** Once you've downloaded real scans (see `DATASETS.md`), drop
   them in `data/raw/MRI`, `data/raw/PET`, fill in `data/raw/labels.csv` and
   `data/raw/clinical.csv`, and run `python scripts/02_preprocess.py --real-data`.
   Nothing else changes.

**This has already been verified end-to-end** on 520 synthetic subjects
(495 MRI + 434 PET) at 48³ resolution on a single CPU core: 3 epochs reached
73% test accuracy / 0.976 macro AUC-ROC, and the explainability outputs are
sane (Grad-CAM lit up the simulated ventricle region for AD predictions;
SHAP correctly ranked MMSE and CDR as the most important clinical features
— exactly the features the synthetic generator actually drives severity
with). Sample outputs are in `docs/sample_outputs/`. See "Known limitations"
below for what's simplified vs. a thesis-grade pipeline.

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # or conda
pip install -r requirements.txt

python scripts/00_run_all.py       # generate -> preprocess -> train -> evaluate -> explain
```

Or run each stage yourself:

```bash
python scripts/01_generate_synthetic_data.py   # skip this once you have real data
python scripts/02_preprocess.py                # add --real-data to ingest data/raw/*
python scripts/03_train.py
python scripts/04_evaluate.py
python scripts/05_explain.py
```

Everything is controlled from **`config.yaml`** — image size, which backbone
(CNN vs. Transformer) each modality uses, training hyperparameters,
explainability sample counts, etc. Nothing else needs editing for normal use.

## Project layout

```
config.yaml                  <- every setting lives here
requirements.txt
DATASETS.md                  <- where to get real MRI+PET data
src/
  config.py                  <- YAML -> attr-dict loader
  utils.py                   <- seeding, device selection, small helpers
  data/
    synthetic.py              <- synthetic MRI/PET/clinical generator
    preprocessing.py          <- skull-strip, registration, smoothing, normalization, augmentation
    clinical.py                <- tabular imputation / encoding / normalization / feature selection
    dataset.py                  <- manifest building (synthetic OR real), caching, PyTorch Dataset
  models/
    backbones.py               <- CNN3DEncoder, ViT3DEncoder, ClinicalEncoder
    fusion.py                  <- missing-modality-aware attention fusion
    network.py                 <- full MultimodalAlzheimerNet
  explain/
    visual_explain.py          <- Grad-CAM (CNN) + attention maps (Transformer)
    shap_explain.py             <- SHAP importance for clinical features
  engine/
    train.py                   <- training loop, early stopping, checkpointing
    evaluate.py                 <- accuracy/precision/recall/specificity/F1/AUC-ROC/confusion matrix
scripts/                      <- thin CLI entry points, one per pipeline stage
data/                         <- raw/ and processed/ (empty until you run a script)
outputs/                      <- checkpoints/, results/, explainability/ (created at runtime)
docs/sample_outputs/          <- example plots from the verification run described above
```

## The model

- **MRI branch**: your choice of a 3D CNN (`Conv3d`-`BatchNorm`-`ReLU`-`MaxPool`
  stack) or a 3D Vision Transformer (cubic patch embedding + `[CLS]` token +
  transformer encoder blocks). Set via `model.mri_backbone` in `config.yaml`.
- **PET branch**: same choice, set via `model.pet_backbone`. The shipped
  config uses CNN for MRI and Transformer for PET, so both code paths are
  exercised by default — change either to `"cnn"` or `"transformer"` freely.
- **Clinical branch**: a small MLP over `Age`, `Gender`, `Education`, `MMSE`, `CDR`.
- **Fusion**: each modality is projected into a shared space; a multi-head
  scoring network learns per-sample "how much to trust this modality," and
  **missing modalities are masked to `-inf` before the softmax** — a
  zero-filled placeholder volume can never be mistaken for real signal.
- **Classifier**: fully-connected layers → 3-way softmax (CN / MCI / AD).

## Explainability

- **Grad-CAM** (`src/explain/visual_explain.py::grad_cam_3d`) for any
  modality using the CNN backbone — gradient-weighted class activation on
  the last conv feature map, overlaid on axial/coronal/sagittal slices.
- **Attention maps** (`vit_attention_map`) for any modality using the
  Transformer backbone — the `[CLS]` token's attention to patch tokens from
  the last block, reshaped back into 3D. This is a lightweight, single-layer
  stand-in for full attention-rollout; extending it to multi-layer rollout
  is a short loop over `model.*_encoder.blocks` (see the docstring).
- **SHAP** (`src/explain/shap_explain.py`) for clinical features — computed
  by masking out MRI/PET (which routes 100% of the fusion attention weight
  to the clinical branch), so it explains "what the model predicts from
  clinical data alone," a clean and well-defined quantity.

## Evaluation

`src/engine/evaluate.py` reports accuracy, macro/per-class precision,
recall (sensitivity), specificity, F1, and AUC-ROC (one-vs-rest), plus a
confusion matrix and ROC curve plot — everything in the architecture
diagram's "Performance Evaluation" panel.

## Known limitations (and how to upgrade them)

This is a research/course-project-grade pipeline, not a clinical one. The
seams to swap in production-grade tools are intentionally small:

| Simplification here | Production-grade replacement |
|---|---|
| Skull stripping = Otsu threshold + largest connected component | FSL BET, HD-BET, or ANTs brain extraction |
| "Registration" = isotropic resampling to a fixed grid | FSL FLIRT/FNIRT or ANTs (rigid/affine/deformable) to a real template (e.g. MNI152) |
| ViT explainability = last-layer `[CLS]` attention | Full multi-layer attention rollout, or Grad-CAM-style methods for transformers |
| Synthetic data | Real ADNI/OASIS-3/OpenNeuro scans — see `DATASETS.md` |

The functions to replace are `skull_strip` and `register_to_template` in
`src/data/preprocessing.py` — everything downstream (caching, the Dataset,
the model, training, evaluation, explainability) is agnostic to how those
two functions are implemented.

## Compute notes

Default `config.yaml` (600 subjects, 64³ volumes, 25 epochs) is sized for a
CUDA GPU. On CPU-only hardware, reduce `data.image_size` (e.g. `[48,48,48]`)
and/or `train.epochs` in `config.yaml` first — that's exactly what was done
for the 520-subject verification run described above, which took roughly
2 minutes to preprocess and ~1 minute/epoch to train on a single CPU core.
`train.device: "auto"` will use CUDA automatically if it's available.
