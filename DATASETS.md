# Where to get real MRI + PET Alzheimer's data

Every option below requires you to register and agree to a data-use
agreement directly with the data provider — none of it can be downloaded
automatically from a script (this is a genuine restriction of the source
institutions, meant to protect patient privacy, not a limitation of this
codebase). Once you have the files, format them as described at the bottom
and everything else in this project stays the same.

## 1. ADNI (Alzheimer's Disease Neuroimaging Initiative) — best fit

The standard dataset for exactly this task: T1 MRI **and** FDG/amyloid PET
**and** clinical/cognitive scores (MMSE, CDR, demographics) for CN/MCI/AD
subjects, thousands of scans.

- Apply for access: https://adni.loni.usc.edu/data-samples/access-data/
- Approval typically takes a few days to a couple of weeks (academic/
  research use is usually approved).
- Download via the LONI Image and Data Archive (IDA) once approved.

## 2. OASIS-3

Longitudinal MRI + PET (PIB/AV45 amyloid, FDG) + clinical data, including
many cognitively normal and impaired subjects.

- https://www.oasis-brains.org/ — register for access (free, fast approval).

## 3. OpenNeuro `ds007561` (the dataset already in your `final_pro.zip`)

This is a real, public, BIDS-formatted dataset — but your uploaded copy only
contains **20 subjects**, and every scan is a `git-annex` placeholder (a few
hundred bytes of pointer text, not actual voxel data). To fetch the real
files on your own machine (this sandbox can't reach OpenNeuro's storage):

```bash
pip install datalad
datalad clone https://github.com/OpenNeuroDatasets/ds007561.git
cd ds007561
datalad get .          # downloads the actual NIfTI files, can take a while
```

Note this dataset alone has only 20 subjects — not enough for the "500+"
target on its own. Combine it with ADNI/OASIS-3, or use it for a quick
real-data smoke test of the pipeline before scaling up.

## 4. Kaggle "Alzheimer's MRI" datasets — MRI only, no PET

Useful only if you decide to do MRI-only ablations; several Kaggle datasets
have thousands of 2D MRI slices but **no paired PET**, so they don't fit this
project's multimodal design as-is.

---

## Formatting real data for this project

1. Put MRI files in `data/raw/MRI/` and PET files in `data/raw/PET/` — any
   filename that **starts with the subject ID** is matched automatically
   (e.g. `sub-001_T1w.nii.gz`, `sub-001_pet.nii.gz`).
2. Create `data/raw/labels.csv`:
   ```csv
   subject_id,diagnosis
   sub-001,CN
   sub-002,AD
   sub-003,MCI
   ```
3. Create `data/raw/clinical.csv` with the columns listed in
   `config.yaml -> data.clinical_features` (default: `Age, Gender, Education,
   MMSE, CDR`):
   ```csv
   subject_id,Age,Gender,Education,MMSE,CDR
   sub-001,72,Female,16,28,0.0
   sub-002,79,Male,12,18,1.0
   ```
   Missing values are fine — `src/data/clinical.py` imputes them.
4. Run:
   ```bash
   python scripts/02_preprocess.py --real-data
   python scripts/03_train.py
   python scripts/04_evaluate.py
   python scripts/05_explain.py
   ```
   If some subjects only have one modality, that's handled automatically —
   just leave the other file out; the missing-modality mask takes care of it.
