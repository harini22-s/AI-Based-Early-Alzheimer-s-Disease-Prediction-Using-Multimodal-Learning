Put real data here (see ../../DATASETS.md):
  MRI/           <- .nii / .nii.gz MRI volumes
  PET/           <- .nii / .nii.gz PET volumes
  labels.csv     <- subject_id, diagnosis
  clinical.csv   <- subject_id, Age, Gender, Education, MMSE, CDR

Or just run `python scripts/01_generate_synthetic_data.py` from the project
root to populate this folder with a synthetic dataset instead.
