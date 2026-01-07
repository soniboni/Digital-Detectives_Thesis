# Digital Detectives - NTFS Timestomping Detection

**Machine Learning-Based Detection of Timestamp Manipulation in NTFS File Systems**

This repository contains a thesis project developing machine learning models to detect timestamp manipulation (timestomping) in NTFS filesystems using $LogFile and $UsnJrnl artifacts, based on Oh et al. (2024) methodology.

---

## Project Overview

### Research Objective

Develop an ML-based system to automatically detect timestamp manipulation in NTFS filesystems by analyzing cross-artifact patterns in $LogFile and $UsnJrnl transaction logs.

### Base Methodology

**Oh, Lee, and Hwang (2024)** - "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"  
Published in IEEE Access ([DOI: 10.1109/ACCESS.2024.10517044](https://ieeexplore.ieee.org/document/10517044))

### Detection Approach

- **Core Logic**: Compare Redo vs Undo timestamps in $LogFile and cross-check MAC timestamps in $UsnJrnl. Flag suspicious files.
- **$FN Checker**: Detect $FN timestamp manipulation:
  1. Timestamp change occurs → file moved in same volume → timestamp change again (NtSetInformationFile API)
  2. Only MAC timestamps changed via SetFileTime() or PowerShell Get-Item. No re-manipulation.
- **Combined Logic**: Core logic first, then $FN checker for additional confirmation. Determines Malicious vs Suspicious.

---

## Current Status: Model Training Complete

### Completed Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Raw Data Parsing | Complete | Parsed $MFT, $LogFile, $UsnJrnl from 27 datasets |
| Phase 2: Data Preprocessing | Complete | Event grouping per file, timestamp normalization |
| Phase 3: Feature Engineering | Complete | 29 features extracted per file |
| Phase 4: Model Training | Complete | 4 models trained, LightGBM selected as best |
| Phase 5: Hyperparameter Tuning | Complete | Baseline retained (tuning did not improve recall) |
| Phase 6: Autopsy Integration | In Progress | Standalone scripts ready |

---

## Final Model Performance

### Selected Model: LightGBM (Baseline Configuration)

| Metric | Test Set | Validation (Aggregate) |
|--------|----------|------------------------|
| **Recall** | **1.0 (100%)** | **1.0 (100%)** |
| CRR (Candidate Reduction Rate) | 99.85% | 99.47% - 99.98% |
| NNI (Number Needed to Investigate) | 29.7 | 4.0 - 27.6 |
| FPR (False Positive Rate) | 0.145% | 0.02% - 1.8% |
| Threshold | 0.02 | 0.02 |

**Key Result**: All 22 known timestomped files detected across validation datasets with zero false negatives.

### Forensic Metrics Framework

Traditional metrics (F1, Precision) are misleading under extreme class imbalance (19,064:1). We adopt forensic-appropriate metrics:

**Primary Metrics (Model Selection):**
- **Recall**: Hard constraint - must detect ALL timestomped files
- **CRR**: Percentage of files analyst does NOT need to review
- **NNI**: Files reviewed to find 1 true positive

**Secondary Metrics:**
- **FPR**: False positive rate (more honest than Precision)
- **Recall@K**: Were all positives in top K% of results?

**Supplementary (Completeness Only):**
- F2 Score, AUCPR
- F1 Score (NOT used for selection)

---

## Dataset Structure

### Training Datasets (23 total)

**PE Cases (12)**: 01-PE through 12-PE  
**APT Cases (11)**: 01-APT17, 02-APT19, 03-APT21, 04-APT28, 05-APT29, 06-APT30, 07-APT37, 08-APT38, 10-DarkHotel663, 11-DarkHotelbbd, 14-Winnti53b

### Validation Datasets (4 total)

| Dataset | Files | Known Timestomped | Recall Achieved |
|---------|-------|-------------------|-----------------|
| 09-APT40 | 29,857 | 6 | 100% |
| 12-Kimsuky | 14,869 | 3 | 100% |
| 13-Winnti731 | 16,701 | 1 | 100% |
| LoneWolf | 17,897 | 12 | 100% |

### Ground Truth

- **Total known timestomped files**: 64 (from Oh et al. LogTracker tool)
- **Training set**: 52 files
- **Validation set**: 22 files
- **Source**: `data/Suspicious Files (v1).csv`

---

## Feature Engineering (29 Features)

### Timestamp Change Features
- `num_timestamp_changes` - Count of timestamp modification events
- `num_backward_jumps` - Timestamps going backwards (strong indicator)
- `num_forward_jumps` - Normal timestamp progression
- `num_creation_changes` - Creation time modifications
- `max_backward_jump_seconds` - Maximum backward time delta
- `mean_jump_seconds` - Average time delta between changes
- `timestamp_change_density` - Changes per event ratio

### Structural Pattern Features
- `num_zero_nanosecond_events` - Round timestamps (timestomping indicator)
- `only_SI_modified` - $SI changed but $FN unchanged
- `num_update_resident_value` - UpdateResidentValue operations
- `repeated_update_resident_value` - Rapid repeated modifications
- `consecutive_timestamp_changes` - Sequential changes without other events

### Cross-Artifact Consistency Features
- `num_logfile_events` - LogFile event count
- `has_logfile_ts_change` - LogFile shows timestamp change
- `has_usn_basic_info` - USN BASIC_INFO_CHANGE present
- `has_usn_close` - USN CLOSE present
- `has_usn_file_create` - USN FILE_CREATE present
- `num_usn_basic_info`, `num_usn_close`, `num_usn_file_create` - Event counts
- `logfile_usn_mismatch` - Inconsistency between artifacts
- `has_usn_basic_pattern` - BASIC_INFO + CLOSE without FILE_CREATE

### Temporal Behavior Features
- `num_usnjrnl_events` - Total USN events
- `min_inter_event_delta`, `max_inter_event_delta`, `mean_inter_event_delta`
- `burstiness_score` - Event clustering measure
- `event_time_span_seconds` - Total time span
- `total_events` - Combined event count

---

## Project Structure
Digital-Detectives_Thesis/
├── data/
│   ├── Phase 1 - Raw Data Parsing/              # Parsed CSV files per dataset
│   ├── Phase 2 - Data Preprocessing/            # Grouped events per file
│   ├── Phase 3 - Feature Engineering/           # File features and detection flags
│   ├── Phase 4 - Model Training/
│   │   └── final_version/
│   │       ├── model_lightgbm.joblib             # Best model (~1 MB)
│   │       ├── model_xgboost.joblib
│   │       ├── model_random_forest.joblib
│   │       ├── model_logistic_regression.joblib
│   │       ├── feature_scaler.joblib
│   │       ├── training_config.json              # Feature columns, thresholds
│   │       ├── model_comparison_forensic.csv
│   │       ├── validation_results_forensic.csv
│   │       └── feature_importance.csv
│   ├── Phase 5 - Hyperparameter Tuning/          # Tuning results (baseline retained)
│   └── Suspicious Files (v1).csv                 # Ground truth labels
│
├── notebooks/
│   ├── Phase 1 - Raw Data Parsing/               # Parsing notebooks
│   ├── Phase 2 - Data Preprocessing/             # Batch processing notebooks
│   ├── Phase 3 - Feature Engineering/            # Feature extraction notebooks
│   ├── Phase 4 - Model Training/                 # Model training with forensic metrics
│   └── Phase 5 - Hyperparameter Tuning/          # LightGBM tuning (Optuna)
│
├── scripts/
│   └── apply_model.py                            # Standalone detection script
│
├── Autopsy File Ingest Module/
│   └── ntfs_timestomping_detector.py             # Autopsy integration (in progress)
│
└── README.md



---

## Model Artifacts for Deployment

Located in `data/Phase 4: Model Training/final version/`:

| File | Description |
|------|-------------|
| `model_lightgbm.joblib` | Trained LightGBM model (recommended) |
| `training_config.json` | Feature columns, threshold (0.02) |
| `feature_scaler.joblib` | StandardScaler (for Logistic Regression only) |
| `feature_importance.csv` | Top features ranked by importance |

---

## Standalone Detection Script

The `scripts/apply_model.py` script applies the trained model to new data:

```bash
python apply_model.py --input file_features.csv --output ./results 
```

## Outputs

- **detected_files.csv**  
  Flagged files with detection explanations

- **files_with_features.csv**  
  All files with confidence scores

- **summary.txt**  
  Human-readable detection report

---

## Next Steps: Autopsy Integration (Phase 6)

### Planned Workflow

1. Autopsy calls parsing scripts on the disk image  
2. Preprocessing groups events per file  
3. Feature extraction computes 29 features  
4. Model application generates detection results  
5. Results displayed in Autopsy with explanations

### Output Files for Autopsy

- **detected_files.csv**  
  Files flagged as timestomped with confidence and reasons

- **files_with_features.csv**  
  All files with confidence scores for manual review

- **summary.txt**  
  Report with statistics and top suspicious files

---

## Hyperparameter Tuning Results (Phase 5)

Bayesian optimization (Optuna, 100 trials) was attempted to improve LightGBM:

| Metric | Baseline | Tuned | Result |
|------|----------|-------|--------|
| Recall | 1.000 | 0.955 | Worse |
| CRR | 0.9985 | 0.9986 | Marginal improvement |

**Decision:** Baseline model retained. The tuned model failed to maintain 100% recall, missing one file during validation. For forensic applications, perfect recall is mandatory.

---

## References

- Oh, S., Lee, S., & Hwang, D. (2024).  
  *Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation.*  
  IEEE Access.

- NTFS `$LogFile` and `$UsnJrnl` forensic analysis methodologies

---

## Requirements

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
optuna>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Important Notes

- Core logic + `$FN` checker is sufficient to detect a significant portion of timestomped files
- All timestamps are preserved at **nanosecond precision**
- Model achieves **100% recall** — no known timestomped files missed
- Detection threshold of **0.02** balances perfect recall with a manageable false-positive rate

---

This updated README reflects all work completed in model training, including the forensic metrics framework, final model performance, and preparation for Autopsy integration.

--- 
Setup Instructions for Phase 6
```bash 
# 1. Clone/pull the branch
git pull origin model-training-soni-v1

# 2. Create their own fresh virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# 4. Install all dependencies from requirements.txt
pip install -r requirements.txt

# 5. Verify installation
python -c "import pandas, lightgbm, sklearn; print('All packages installed successfully')"

```