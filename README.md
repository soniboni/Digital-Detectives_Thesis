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

1. **Event-level data merge**: Preserve LSN/USN identifiers for exact ground truth matching
2. **Forensic pattern extraction**: Zero nanoseconds, time reversal events, cross-artifact validation
3. **ML classification**: Train models on forensic features to detect timestomped files
4. **Production deployment**: Autopsy integration for operational forensic investigations

---

## Dataset Structure

### Training Datasets (19 total)

**PE Cases (12)**: 01-PE through 12-PE  
**APT Cases (7)**: 01-APT17, 02-APT19, 03-APT21, 04-APT28, 05-APT29, 06-APT30, 07-APT37

Each dataset contains:
- **LogFile CSV**: NTFS transaction log with timestamp change events
- **UsnJrnl CSV**: NTFS change journal with file modification events  
- **Suspicious CSV**: Ground truth labels from Oh et al.'s NTFS Artifact Analysis Tool

### Validation Datasets (5 total)

**Lone Wolf**: 12 timestomped files (Autopsy integration validation)  
**09-APT40**: 1 file (zero nanoseconds + file move)  
**12-Kimusky**: 3 files (zero nanoseconds)  
**13-Winnti731**: 1 file (zero nanoseconds + file move)  
**02-APT19**: 1 file (zero nanoseconds + file move)

**Total**: 18 validation files, all with "Zero in 100-nanoseconds" pattern

**Held-out datasets**: 08-APT38, 10-DarkHotel663, 11-DarkHotelbbd, 14-Winnti43b (moved to training for increased dataset size)

---

## Implementation Pipeline

### Phase 1: Event-Level Data Merge

**Goal**: Merge LogFile, UsnJrnl, and Suspicious CSVs while preserving event-level granularity

**Process**:
1. Load all 19 training datasets
2. Filter LogFile: Keep Time Reversal + Update events
3. Filter UsnJrnl: Keep Basic_Info_Change events  
4. Merge LogFile + Suspicious by exact LSN match
5. Merge UsnJrnl + Suspicious by exact USN match
6. Concatenate all events (preserve LSN/USN for ground truth)
7. Process validation datasets separately (same pipeline, no ground truth labels)

**Output**:
- `data/processed/Phase 1/training_events.csv`
- `data/processed/Phase 1/validation/[dataset]_events.csv` (5 files)

### Phase 2: Feature Engineering

**Goal**: Extract production-ready forensic features from raw LogFile/UsnJrnl fields

**Features** (~30 total):
- **Forensic patterns**: zero_in_nanoseconds, time_reversal_event, basic_info_changed, using_another_timestamp
- **Cross-artifact validation**: has_logfile_evidence, has_usnjrnl_evidence, cross_artifact_validation_score
- **Temporal patterns**: event_count_per_file, events_in_1min_window, events_in_5min_window
- **File characteristics**: is_executable, is_document, is_archive, is_system_file, path_depth, filename_length

**Output**:
- `data/processed/Phase 2/training_features.csv`
- `data/processed/Phase 2/validation/[dataset]_features.csv` (5 files)

### Phase 3: Model Training

**Goal**: Train ML models with proper train/test split and class imbalance handling

**Models**:
1. Random Forest (ensemble baseline)
2. XGBoost (gradient boosting)
3. LightGBM (fast gradient boosting)
4. Logistic Regression (linear baseline)

**Strategy**:
- Case-based split: 15 datasets training, 4 datasets testing
- SMOTE for class imbalance (minority oversampling)
- Class weights for cost-sensitive learning
- Cross-validation for hyperparameter tuning

**Output**:
- `models/best_model.pkl`
- `data/processed/Phase 3/model_comparison.csv`
- `data/processed/Phase 3/feature_importance.csv`

### Phase 4: Validation Testing

**Goal**: Evaluate model on 5 completely unseen validation datasets

**Process**:
1. Load validation features (from Phase 2)
2. Run model predictions (no ground truth used)
3. Compare predictions with Suspicious CSV ground truth
4. Calculate metrics: Recall, Precision, F1-score
5. Analyze false positives and misses

**Success Criteria**:
- Recall: 16+/18 files detected (89%+)
- Precision: 75%+
- F1-score: 80%+

**Output**:
- `data/processed/Phase 4/validation_results.csv`
- `data/processed/Phase 4/flagged_files.csv`
- `data/processed/Phase 4/performance_report.txt`

### Phase 5: Autopsy Integration

**Goal**: Deploy as Autopsy plugin for operational use

**Components**:
- Ingest module for automatic detection during acquisition
- Custom artifact type for timestomped files
- Reporting module with LSN/USN references

---

---

## Key Findings

### Ground Truth Handling

**Critical**: Raw UsnJrnl CSVs have NO detail field. Ground truth labels and "Zero in 100-nanoseconds" patterns come from Suspicious CSVs (Oh et al.'s tool output).

**Suspicious CSV Structure**:
```csv
source,lsn/usn,category,detail
usnjrnl,239046272,Timestamp Manipulation,"[Suspicious] Zero in CreationTime's 100-nanoseconds"
logfile,8730038250,Timestamp Manipulation,"[Suspicious] $SI timestamp changed to past" 
```

---

## Key Findings
**File-level detection:** Model detects timestomped FILES (not specific events). A file with multiple timestomping events is correctly detected if ANY event is flagged. Example: DeathToll.jpg has 6 BASIC_INFO_CHANGE events. Oh et al. flagged USN 239046272. Our model flagged USN 249181568. Both are correct (same file, different events).

---
## Current Status
**Phase:** Starting fresh implementation
**Next:** Phase 1 - Event-level data merge

--- 
## Research Contribution
1. Event-level granularity: Preserves LSN/USN for exact ground truth matching
2. Cross-artifact validation: Combines LogFile + UsnJrnl for higher confidence
3. Production-ready features: All features extractable from raw forensic artifacts
4. Operational deployment: Autopsy integration for real-world forensic investigations

This is a thesis research project focused on developing ML-based timestomping detection for NTFS filesystems.


---

