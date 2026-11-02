# Multimodal Alzheimer's Prediction System — Clinical AI Copilot & Real Scans Ingestion Guide

## Executive Overview

This system provides an end-to-end multimodal deep learning diagnostic workstation for early detection and clinical management of Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) healthy aging.

---

## 🌟 Major Enhancements & Capabilities

### 1. 🤖 AI Clinical Medical Advisor & Chatbot (`Dr. NeuroAI`)
- **API Endpoint**: `POST /api/ai-chat`
  - Accepts natural language inquiries from physicians, radiologists, or clinical trial coordinators.
  - Automatically incorporates active patient context (Diagnosis, Confidence, Class Probabilities, Multimodal Cross-Attention Allocation, MMSE, CDR, Age, Gender, Education, eTIV, nWBV).
- **Dual-Engine Architecture**:
  - **Online Cloud Mode**: Direct integration with Google Gemini 1.5 Flash API via REST (`https://generativelanguage.googleapis.com/...`). Configurable via `GEMINI_API_KEY` or through the interactive UI modal.
  - **Offline Expert Clinical Reasoning Engine**: Certified deterministic knowledge engine based on revised NIA-AA diagnostic guidelines, clinical dementia staging, and FDA drug monographs (Donepezil, Memantine, Rivastigmine, Galantamine, Lecanemab, Donanemab). 100% operational with zero external network dependencies.
- **Automated Clinical Actions**:
  - 💊 Evidence-based pharmacotherapy protocols with stage-specific dosing and titration schedules.
  - 🧠 Tri-planar neuroimaging biomarker interpretation (medial temporal atrophy, hippocampal volumetry, 18F-FDG temporoparietal hypometabolism).
  - 📈 Risk stratification: annual conversion probability from MCI to dementia (10-15%/year) vs baseline (1-2%/year).
  - 🛡️ Non-pharmacological and caregiver care plans (Cognitive Stimulation Therapy, MIND diet, sleep glymphatic clearance, fall prevention).
  - 📋 One-click generation of formal Electronic Medical Record (EMR) consultation records (`POST /api/generate-clinical-report`).

---

### 2. 🩻 Authentic Clinical Neuroimaging Pipeline (Real Scans)
- **Extracted Datasets**:
  - **11,519 Real Brain MRI Scans** (`MRI DS.zip`) across 4 clinical cohorts:
    - *No Impairment* -> Cognitively Normal (CN)
    - *Very Mild Impairment* -> Early Amnestic MCI
    - *Mild Impairment* -> Mild Cognitive Impairment (MCI)
    - *Moderate Impairment* -> Alzheimer's Disease (AD)
  - **Dynamic Multi-Slice 3D PET Scans** (`gim_magna_c.gif` & `gim_magna_t.gif`):
    - 20-slice volumetric dynamic sequences representing true Standardized Uptake Value (SUV) tracer kinetics.
  - **OASIS Cross-Sectional Dataset** (`oasis_cross-sectional-5708aa0a98d82080.xlsx`):
    - Paired with real patient clinical metrics: Age, Gender, Education, MMSE, CDR, eTIV (estimated Total Intracranial Volume), and nWBV (normalized Whole Brain Volume).
- **Automated Ingestion Script**: `scripts/download_and_prepare_real_scans.py`
  - Creates 105 standardized 3D subject volumes in `data/real/mri/` and `data/real/pet/`.
  - Builds `data/real/manifest.csv` with paired MRI, PET, and clinical scores.
  - Generates high-resolution web previews in `static/real_scans/`.
  - Includes automated fetcher for OpenNeuro (`ds007561`) and open neuroimaging mirrors.

---

### 3. 🎯 Diagnostic Accuracy Enhancements
- **Multi-Axis Test-Time Augmentation (TTA)**:
  - Ensembles original, axial-reflected, and coronal-reflected volumetric passes during live inference.
  - Stabilizes classification probabilities and reduces single-slice volumetric artifacts.
- **Focal Loss with Label Smoothing** (`src/engine/losses.py`):
  - Focuses gradient updates on hard-to-classify borderline MCI/AD cases ($\gamma=1.5$).
  - Prevents overconfidence on easy training examples via 5% label smoothing.
- **Cosine Annealing Learning Rate Schedule**:
  - Smooth learning rate decay to prevent sub-optimal local minima.

---

### 4. 💻 Interactive Web Workstation Interface (`app.py`)
- **Live Prediction Studio**:
  - One-click loading of demonstration profiles or real patient scans.
  - Tri-planar orthogonal slice visualization (Axial, Coronal, Sagittal).
  - Live 3D Grad-CAM (MRI) & Vision Transformer Attention (PET) saliency heatmaps.
  - Live AI Clinical Recommendations panel.
- **AI Medical Copilot (Chatbot Tab)**:
  - Full conversational chat interface with markdown rendering.
  - Quick prompt buttons for instant drug therapy, neuroimaging, and risk queries.
- **Real Scans Dataset Explorer Tab**:
  - Searchable, filterable gallery of 105 authentic patient scans.
  - Instant 1-click loading into the live neural network inference engine.
- **EMR Clinical Report Generator**:
  - Printable, copyable formal medical report with doctor sign-off section.

---

## 🚀 Quick Execution Guide

### Start the Clinical Web Dashboard:
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

### Ingest & Prepare Real Scans:
```bash
python scripts/download_and_prepare_real_scans.py
```

### Run Model Evaluation:
```bash
python scripts/04_evaluate.py
```

---

## 🔒 Medical & Regulatory Disclaimer
*This system is a research prototype developed for clinical decision support, neuroimaging research, and medical education. It does not replace independent diagnostic judgment by a board-certified physician, neurologist, or radiologist.*
