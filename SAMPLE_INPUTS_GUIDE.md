# Live Prediction & Sample Patient Test Inputs Guide

This directory contains pre-packaged multimodal test cases for immediate testing of the **Alzheimer's Multimodal Deep Learning Prediction Model**.

---

## 📁 Sample Test Packages

### 1. Cognitively Normal (CN) Patient
- **Directory**: `sample_inputs/CN_Cognitively_Normal/`
- **Files**:
  - `mri_scan.npy`: 3D Skull-stripped & normalized brain MRI volume ($64 \times 64 \times 64$ voxels).
  - `pet_scan.npy`: 3D SUV-normalized cerebral PET volume ($64 \times 64 \times 64$ voxels).
  - `patient_profile.json`:
    - `Age`: 64.0
    - `Gender`: Female
    - `Education`: 16.0 years
    - `MMSE`: 29.5 (Normal score)
    - `CDR`: 0.0 (No impairment)
  - `mri_slice_preview.jpg` & `pet_slice_preview.jpg`: Orthogonal slice image previews.

### 2. Mild Cognitive Impairment (MCI) Patient
- **Directory**: `sample_inputs/MCI_Mild_Cognitive_Impairment/`
- **Files**:
  - `mri_scan.npy`: 3D normalized brain MRI volume ($64 \times 64 \times 64$ voxels).
  - `pet_scan.npy`: 3D SUV-normalized cerebral PET volume ($64 \times 64 \times 64$ voxels).
  - `patient_profile.json`:
    - `Age`: 72.0
    - `Gender`: Male
    - `Education`: 14.0 years
    - `MMSE`: 23.5 (Mild impairment range)
    - `CDR`: 0.5 (Very mild impairment)

### 3. Alzheimer's Disease (AD) Patient
- **Directory**: `sample_inputs/AD_Alzheimers_Disease/`
- **Files**:
  - `mri_scan.npy`: 3D normalized brain MRI volume ($64 \times 64 \times 64$ voxels).
  - `pet_scan.npy`: 3D SUV-normalized cerebral PET volume ($64 \times 64 \times 64$ voxels).
  - `patient_profile.json`:
    - `Age`: 78.0
    - `Gender`: Female
    - `Education`: 12.0 years
    - `MMSE`: 15.0 (Severe impairment range)
    - `CDR`: 1.5 (Moderate-to-severe dementia)

---

## 🚀 How to Run Live Predictions

### Option A: Via the Web Dashboard (Recommended)
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Navigate to `http://localhost:5000` in your web browser.
3. In the **Live Prediction Studio** tab, click any of the 3 quick-load buttons:
   - **Load CN (Cognitively Normal)**
   - **Load MCI (Mild Impairment)**
   - **Load AD (Alzheimer's Disease)**
4. Or upload custom MRI/PET files (`.npy`, `.nii`, `.nii.gz`, or 2D `.jpg`/`.png` slices) and adjust the clinical sliders.
5. Click **"Execute Multimodal Diagnostic Inference"**.
6. The system displays:
   - Class probabilities ($P(\text{CN}), P(\text{MCI}), P(\text{AD})$)
   - Diagnostic confidence level
   - Multimodal cross-attention weights (MRI vs. PET vs. Clinical)
   - Live generated 3D Grad-CAM (MRI) & ViT Saliency (PET) across axial, coronal, and sagittal planes.

---

### Option B: Via Python Script
```python
import torch
import numpy as np
from pathlib import Path
from src.config import load_config
from src.models.network import MultimodalAlzheimerNet

# 1. Load configuration and model
config = load_config(Path("config.yaml"))
model = MultimodalAlzheimerNet(config)
checkpoint = torch.load("outputs/checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 2. Load sample data
mri_vol = np.load("sample_inputs/AD_Alzheimers_Disease/mri_scan.npy")
pet_vol = np.load("sample_inputs/AD_Alzheimers_Disease/pet_scan.npy")
clinical = torch.tensor([[78.0, 1.0, 12.0, 15.0, 1.5]], dtype=torch.float32)

mri_t = torch.from_numpy(mri_vol[None, None, ...]).float()
pet_t = torch.from_numpy(pet_vol[None, None, ...]).float()
mask = torch.ones(1, 3)  # MRI, PET, Clinical all present

# 3. Predict
with torch.no_grad():
    logits, extra = model(mri_t, pet_t, clinical, mask, return_intermediates=True)
    probs = torch.softmax(logits, dim=-1)[0]
    classes = ["CN", "MCI", "AD"]
    pred = classes[torch.argmax(probs)]
    print(f"Predicted Diagnosis: {pred} (Confidence: {probs.max().item()*100:.1f}%)")
    print(f"Attention Weights: {extra['fusion_weights'][0].numpy()}")
```
