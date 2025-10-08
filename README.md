**🕵️ Digital Detectives – Autopsy Plugin Prototype**

This repository is part of our thesis project, which aims to develop a prototype Autopsy plugin capable of detecting timestomped files through in-depth analysis of $LogFile and $UsnJrnl artifacts. The system integrates machine learning models to enhance detection accuracy and reduce false positives.

**🚧 Work in Progress – The plugin is currently under active development.**

---

## 📋 Project Overview

### Problem Statement
Timestomping (timestamp manipulation) is a common anti-forensic technique used by adversaries to hide malicious activity by altering file MAC (Modified, Accessed, Created) timestamps. Traditional detection methods, while effective at identifying patterns, suffer from high false positive rates that hinder forensic investigations.

### Solution
We propose an **ML-enhanced Autopsy plugin** that:
- Parses NTFS artifacts ($LogFile & $UsnJrnl)
- Applies machine learning classification (Random Forest / XGBoost)
- Provides **confidence-based reporting** to prioritize suspicious files
- Reduces false positives through ensemble learning and feature engineering

### Research Foundation
This work builds upon:
- **Oh et al. (2024)** - "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"
  - Provided base dataset (12 forensic cases)
  - NTFS Log Tracker tool for artifact parsing
  - Pattern-based detection algorithm (⚠️ high false positive rate)
- **Machine Learning Extensions:**
  - Random Forest for NTFS artifact analysis achieving >99% accuracy (Computers & Security, 2020)

---

## 📂 Dataset Description

### Source
**Oh et al.'s Timestomping Detection Dataset**
- 12 forensic case scenarios (01-PE through 12-PE)
- Real-world timestomping attack simulations
- Parsed using NTFS Log Tracker tool

### Data Structure

#### LogFile Artifacts (`data/raw/XX-PE-LogFile.csv`)
- **Records:** ~40,000 per case
- **Key Columns:**
  - `LSN`: Log Sequence Number (unique event ID)
  - `EventTime(UTC+8)`: When the event occurred
  - `Event`: Operation type (Create, Delete, Update, etc.)
  - `Full Path`: Complete file path
  - `CreationTime`, `ModifiedTime`, `MFTModifiedTime`, `AccessedTime`: MAC timestamps

#### UsnJrnl Artifacts (`data/raw/XX-PE-UsnJrnl.csv`)
- **Records:** ~300,000+ per case
- **Key Columns:**
  - `USN`: Update Sequence Number
  - `TimeStamp(UTC+8)`: Event timestamp
  - `FullPath`: File path
  - `EventInfo`: Change type (File_Created, File_Closed, etc.)
  - `FileAttribute`: Attribute flags

#### Suspicious Labels (`data/raw/suspicious/XX-PE-Suspicious.csv`)
- **Total Records:** 504 labeled suspicious events across all cases
- **Label Distribution:** Highly imbalanced (1-161 events per case)
- **Categories:**
  - `Execution of Suspicious Programs`: Timestomping tools detected
  - `Timestamp Manipulation`: Files with manipulated timestamps
- **Mapping Keys:**
  - `source`: logfile or usnjrnl
  - `lsn/usn`: Links to original artifact record

---

## 🔬 Methodology

### Phase 1: Data Collection & Preprocessing

#### **A. Data Labelling**
**Objective:** Annotate forensic artifacts with ground truth labels

**Process:**
1. Load 12 LogFile datasets (~40K records each)
2. Load 12 UsnJrnl datasets (~300K records each)
3. Load suspicious behavior indicators from NTFS Log Tracker
4. Match labels using LSN/USN identifiers
5. Create binary target variables:
   - `is_timestomped`: File timestamps were manipulated
   - `is_suspicious_execution`: Timestomping tool execution detected
   - `is_suspicious`: Overall binary target (either condition true)
   - `label_source`: Provenance tracking (logfile/usnjrnl/both)

**Output:** `data/processed/XX-PE-LogFile-Labelled.csv` & `XX-PE-UsnJrnl-Labelled.csv`

#### **B. Data Case Merging**
**Objective:** Create unified timeline per case combining LogFile + UsnJrnl

**Challenge:** Direct timestamp merge loses 90%+ records due to sparse exact matches

**Strategy - Feature Union Approach (Recommended):**
1. Standardize column names across both artifact types:
   ```
   [timestamp, filepath, event_type, source_artifact,
    creation_time, modified_time, accessed_time,
    is_timestomped, is_suspicious_execution]
   ```
2. Add `source_artifact` column: 'logfile' or 'usnjrnl'
3. Concatenate vertically (union) preserving all events
4. Sort by timestamp chronologically
5. This preserves complete event timeline (~340K records per case)

**Alternative - Temporal Join (if needed):**
- Merge on filepath + timestamp window (±1 second tolerance)
- Outer join to preserve unmatched records
- Fill missing features with appropriate defaults

**Output:** `data/processed/XX-PE-Merged.csv` (01-PE through 12-PE)

#### **C. Master Timeline Creation**
**Objective:** Aggregate all 12 cases into single training dataset

**Process:**
1. Add `case_id` column (01-12) to each merged case
2. Concatenate all 12 merged datasets vertically
3. Stratified shuffle maintaining case distribution
4. Final dataset: ~4 million events with 504 positive labels

**Output:** `data/processed/master_timeline.csv`

---

### Phase 2: Feature Engineering

**Objective:** Extract ML-relevant features to capture timestomping patterns

#### **Temporal Features**
```python
time_delta_seconds          # Time since previous event (same file)
event_frequency_1min        # Events per file in 1-min window
event_frequency_1hour       # Events per file in 1-hour window
hour_of_day                 # 0-23 (detect off-hours activity)
day_of_week                 # 0-6 (detect weekend anomalies)
inter_event_variance        # Timing inconsistency metric
time_since_case_start       # Relative position in case timeline
```

#### **Timestamp Anomaly Features** (Critical for Timestomping)
```python
nanosec_is_zero             # Classic timestomping indicator
timestamp_goes_backward     # Creation > Modification (impossible)
creation_after_modification # C > M timestamp (suspicious)
accessed_before_creation    # A < C timestamp (impossible)
timestamp_year_delta        # Years between event time and recorded timestamp
timestamp_future            # Timestamp > event time
mac_timestamps_identical    # All MAC times exactly match (suspicious)
```

#### **File Path Features**
```python
path_depth                  # Directory nesting level
is_system_path              # In Windows/System32/Program Files
is_temp_path                # In Temp/AppData directories
file_extension              # Categorical encoding
filename_length             # Abnormally long names (evasion)
path_entropy                # Randomness score (obfuscation detection)
```

#### **Event Pattern Features**
```python
event_type_encoded          # One-hot or label encoding
consecutive_same_events     # Repetition counter
rare_event_type             # Statistical frequency score
event_type_count_per_file   # Unique operations per file
```

#### **Cross-Artifact Features** (if using merged data)
```python
appears_in_both_artifacts   # Binary: present in LogFile AND UsnJrnl
timestamp_mismatch_seconds  # Delta between LogFile & UsnJrnl times
```

**Output:** `data/processed/master_timeline_features.csv`

---

### Phase 3: Model Training & Evaluation

#### **A. Data Splitting Strategy**
**Case-Based Stratification (Recommended):**
- **Train:** Cases 01-08 (66%)
- **Validation:** Cases 09-10 (17%)
- **Test:** Cases 11-12 (17%)

*Rationale:* Prevents data leakage from same case appearing in train/test

**Class Imbalance Handling:**
- SMOTE oversampling for minority class
- Class weight adjustment (`class_weight='balanced'`)
- Stratified sampling preserving positive label ratio

#### **B. Model Architectures**

**Random Forest Classifier**
```python
- n_estimators: 100-500 (tuned)
- max_depth: 10-30 (prevent overfitting)
- min_samples_split: 2-10
- class_weight: 'balanced_subsample'
- Strengths: Feature importance, interpretability, robust to outliers
```

**XGBoost Classifier**
```python
- learning_rate: 0.01-0.1
- max_depth: 5-10
- scale_pos_weight: Ratio of negative to positive samples
- Strengths: Higher accuracy, better handling of imbalance
```

**Ensemble Voting Classifier (Optional)**
- Combines RF + XGBoost predictions
- Soft voting with probability averaging

#### **C. Evaluation Metrics**
Given extreme class imbalance (504 positives / ~4M total):

**Primary Metrics:**
- **Precision:** Minimize false positives (critical for forensic workflow)
- **Recall:** Catch actual timestomped files
- **F1-Score:** Harmonic mean balance
- **AUC-ROC:** Overall discriminative ability
- **Precision-Recall Curve:** Better for imbalanced data

**Forensic-Specific Metrics:**
- **False Positive Rate @ 95% Recall:** Acceptable false alarm rate
- **Top-K Accuracy:** Precision in top 100/500 predictions

#### **D. Model Interpretability**
```python
- Feature importance ranking (Random Forest built-in)
- SHAP values for individual predictions
- Confusion matrix analysis
- Example case studies of TP/FP/FN
```

#### **E. Confidence Score Calibration**
Map probability outputs to risk levels:
- **High Risk:** P(timestomped) > 0.8
- **Medium Risk:** 0.5 < P < 0.8
- **Low Risk:** 0.3 < P < 0.5
- **Benign:** P < 0.3

**Output:**
- Trained models: `models/random_forest_model.pkl`, `models/xgboost_model.pkl`
- Evaluation report: `outputs/model_evaluation.md`
- Feature importance: `outputs/feature_importance.csv`

---

## 🔌 Autopsy Plugin Integration (Future Work)

### Architecture
```
Autopsy Case → Plugin Ingest Module
    ↓
Extract $MFT, $LogFile, $UsnJrnl
    ↓
Parse to CSV format
    ↓
Feature Engineering Pipeline
    ↓
Load Trained ML Model
    ↓
Predict + Generate Confidence Scores
    ↓
Autopsy Report with Risk-Ranked Files
```

### Key Features
- Real-time artifact parsing (MFTECmd, LogFileParser integration)
- Confidence-based filtering (reduce analyst workload)
- Detailed report with SHAP explanations
- Comparison with Oh et al.'s algorithm results

---

## 📊 Expected Outcomes

1. **Improved Detection Accuracy:** >95% precision while maintaining >90% recall
2. **Reduced False Positives:** 50-70% reduction vs. rule-based methods
3. **Confidence Scoring:** Risk-stratified output for investigative prioritization
4. **Interpretable Results:** Feature importance + SHAP for court admissibility
5. **Scalable Plugin:** Automated workflow integrated into Autopsy forensic platform

---

## 📚 References

1. Oh, J. et al. (2024). "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation." *IEEE Access*.
2. "Detection of Timestamps Tampering in NTFS using Machine Learning." *Procedia Computer Science*, 2019.
3. "De-Wipimization: Detection of data wiping traces for investigating NTFS file system." *Computers & Security*, 2020.
4. MITRE ATT&CK: T1070.006 - Indicator Removal: Timestomp

---

## 🛠️ Project Structure

```
Digital-Detectives_Thesis/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── XX-PE-LogFile.csv        # 12 LogFile artifacts
│   │   ├── XX-PE-UsnJrnl.csv        # 12 UsnJrnl artifacts
│   │   └── suspicious/               # Ground truth labels
│   │       └── XX-PE-Suspicious.csv
│   └── processed/                    # Cleaned & engineered data
│       ├── XX-PE-Merged.csv         # Per-case merged timelines
│       └── master_timeline_features.csv
├── notebooks/
│   ├── Phase 1 - Data Labelling.ipynb
│   ├── Phase 2 - Feature Engineering.ipynb
│   └── Phase 3 - Model Training.ipynb
├── models/                           # Trained ML models
├── outputs/                          # Reports & visualizations
└── plugin/                           # Autopsy plugin code (TBD)
