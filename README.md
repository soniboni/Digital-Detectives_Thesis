**🕵️ Digital Detectives – Autopsy Plugin Prototype**

This repository is part of our thesis project, which aims to develop a prototype Autopsy plugin capable of detecting timestomped files through in-depth analysis of $LogFile and $UsnJrnl artifacts. The system integrates machine learning models to enhance detection accuracy and reduce false positives.

**🚧 Work in Progress – The plugin is currently under active development.**

---

## 📊 Project Progress

| Phase | Task | Status | Output |
|-------|------|--------|--------|
| **Phase 1A** | Data Labelling | ✅ **Completed** | 24 labelled CSV files (12 LogFile + 12 UsnJrnl) |
| **Phase 1B** | Data Case Merging | 🔄 **In Progress** | - |
| **Phase 1C** | Master Timeline Creation | ⏳ **Pending** | - |
| **Phase 2** | Feature Engineering | ⏳ **Pending** | - |
| **Phase 3** | Model Training & Evaluation | ⏳ **Pending** | - |

### Latest Accomplishments

#### ✅ Phase 1A: Data Labelling (Completed)
**Date Completed:** October 10, 2025

**What We Did:**
- Successfully labelled all 12 forensic cases (01-PE through 12-PE)
- Matched suspicious behavior indicators from NTFS Log Tracker to forensic artifacts
- **Critical Design Decision:** Separated tool execution from actual timestomping
  - Tool execution (e.g., NewFileTime.exe) → **FEATURE** (not a label)
  - Actual timestamp manipulation → **LABEL** (what we're detecting)
  - This prevents model confusion and ensures clean training targets

**Key Findings:**
- **Total Records Processed:** 3,372,330 events (243,884 LogFile + 3,128,446 UsnJrnl)
- **Suspicious Indicators (Raw):** 504 indicator records from NTFS Log Tracker
- **Actual Timestomped Events:** 252 unique events (14 LogFile + 238 UsnJrnl)
- **Tool Executions Detected:** 16 events (tracked as features, not labels)
- **Class Imbalance Ratio:** 1:13,382 (extreme imbalance requiring SMOTE/class weighting)

**Understanding the 504 → 252 Reduction:**
The reduction from 504 suspicious indicators to 252 actual labels occurred due to:
1. **Duplicate Indicators (178 duplicates):** Same LSN/USN flagged by multiple detection rules
2. **Missing USN Values (~58 missing):** USNs in indicators not found in parsed UsnJrnl CSVs (likely carving-recovered records)
3. **Tool Execution Separated (16 events):** Now properly categorized as features instead of labels

**Output:**
- ✅ 12 labelled LogFile datasets: `data/processed/Phase 1 - Data Collection & Preprocessing/A. Data Labelled/XX-PE-LogFile-Labelled.csv`
- ✅ 12 labelled UsnJrnl datasets: `data/processed/Phase 1 - Data Collection & Preprocessing/A. Data Labelled/XX-PE-UsnJrnl-Labelled.csv`

**New Column Structure:**
- **Label (prediction target):**
  - `is_timestomped`: Binary flag for actual timestamp manipulation (1 = manipulated, 0 = benign)
  - `label_source`: Provenance tracking ('logfile' or 'usnjrnl')

- **Features (help predict):**
  - `timestomp_tool_executed`: Binary flag (1 = timestomping tool detected)
  - `suspicious_tool_name`: Tool name (e.g., "NewFileTime.exe", NaN if none)

**Why This Matters:**
- Model learns **timestamp patterns** (actual behavior), not **file name patterns** (tool signatures)
- Better generalization to detect timestomping with unknown tools
- Cleaner labels = better model performance
- Single clear target variable (`is_timestomped`) prevents confusion

**Notebook:** [Phase 1 - Data Collection & Preprocessing/A. Data Labelling.ipynb](notebooks/Phase%201%20-%20Data%20Collection%20%26%20Preprocessing/A.%20Data%20Labelling.ipynb)

---

### 🎯 Current Work: Phase 1B - Data Case Merging

**Objective:** Merge LogFile and UsnJrnl artifacts per case using smart aggregation to eliminate duplicates while preserving correlation

**Updated Strategy: Smart Aggregation + Join**
After initial exploration, we discovered that simple concatenation creates ~4M rows with many NaN values. The new approach:

1. **Aggregate UsnJrnl Events:**
   - Combine multiple UsnJrnl events at same timestamp+filepath into one representative row
   - Prevents duplicate LogFile records in the merge
   - Reduces UsnJrnl from 3.1M → ~631K rows

2. **Intelligent Join:**
   - Outer join on timestamp + filepath + filename
   - Preserves events from both artifacts (matched/logfile_only/usnjrnl_only)
   - Creates rich cross-artifact features for the model

3. **Handle New Columns from Phase 1A:**
   - Aggregate `timestomp_tool_executed` (max across grouped events)
   - Aggregate `suspicious_tool_name` (preserve tool names)
   - `is_timestomped` tracks ONLY actual timestomping (not tool execution)

**Expected Results:**
- Dataset reduction: ~3.4M → ~860K rows (74.5% smaller)
- Each LogFile LSN appears exactly once (no duplicates)
- 252 timestomped events preserved across 12 cases
- Statistics will make mathematical sense!

**Next Steps:**
1. ✅ Update Phase 1B aggregation to handle new feature columns
2. ✅ Simplified labeling: removed redundant `is_suspicious` column
3. Re-run merging with corrected Phase 1A labels
4. Verify: 252 timestomped events, 16 tool executions as features

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
3. Load suspicious behavior indicators from NTFS Log Tracker (504 indicator records)
4. Match labels using LSN/USN identifiers
5. **Critical Design Decision:** Separate labels from features
   - **Label (what we predict):**
     - `is_timestomped`: Actual timestamp manipulation detected (PRIMARY TARGET)
     - `label_source`: Provenance tracking (logfile/usnjrnl)
   - **Features (help predict):**
     - `timestomp_tool_executed`: Binary flag for tool execution
     - `suspicious_tool_name`: Tool name (e.g., "NewFileTime.exe")

**Key Insight:**
Tool execution (e.g., running NewFileTime.exe) is a **feature**, not a **label**. Only actual timestamp manipulation is labeled. This prevents model confusion and ensures it learns timestamp patterns, not file name patterns.

**Design Simplification:**
Originally had both `is_timestomped` and `is_suspicious` columns, but they were identical. Removed redundancy to keep only `is_timestomped` as the single, clear target variable.

**Output:**
- `data/processed/XX-PE-LogFile-Labelled.csv` (252 timestomped events total)
- `data/processed/XX-PE-UsnJrnl-Labelled.csv` (16 tool executions as features)

#### **B. Data Case Merging**
**Objective:** Create unified timeline per case combining LogFile + UsnJrnl

**Challenge:** Simple concatenation creates ~4M rows with many NaN values. Direct join creates duplicate LogFile records.

**Solution - Smart Aggregation + Join:**
1. **Normalize timestamps** to fix format mismatches (e.g., '0:21:57' vs '00:21:57')
2. **Aggregate UsnJrnl events** at same timestamp+filepath+filename:
   - Multiple UsnJrnl events (File_Created → Data_Added → File_Closed) → single row
   - Use max() for timestomped flags to preserve detections
   - Combine event info to show complete event sequence
   - Reduces 3.1M UsnJrnl rows → ~631K aggregated rows
3. **Prepare LogFile** with 'lf_' column prefix
4. **Outer join** on timestamp + filepath + filename:
   - Creates matched, logfile_only, and usnjrnl_only records
   - Each LogFile LSN appears exactly once (no duplicates!)
   - Preserves all 252 timestomped events
5. **Merge labels** using np.maximum() from both artifacts
6. **Handle new feature columns:**
   - `timestomp_tool_executed`: max across aggregated events
   - `suspicious_tool_name`: preserve tool names from both artifacts

**Results:**
- Dataset reduction: 3.4M → ~860K rows (74.5% smaller)
- No duplicate LogFile records
- Rich cross-artifact features for model training
- Clean separation: 252 timestomped labels + 16 tool execution features

**Output:** `data/processed/XX-PE-Merged.csv` (01-PE through 12-PE)

#### **C. Master Timeline Creation**
**Objective:** Aggregate all 12 cases into single training dataset

**Process:**
1. Add `case_id` column (01-12) to each merged case
2. Concatenate all 12 merged datasets vertically
3. Stratified shuffle maintaining case distribution
4. Final dataset: ~860K events with 252 timestomped labels

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
Timestamp Anomaly Features (Critical for Timestomping)
nanosec_is_zero             # Classic timestomping indicator
timestamp_goes_backward     # Creation > Modification (impossible)
creation_after_modification # C > M timestamp (suspicious)
accessed_before_creation    # A < C timestamp (impossible)
timestamp_year_delta        # Years between event time and recorded timestamp
timestamp_future            # Timestamp > event time
mac_timestamps_identical    # All MAC times exactly match (suspicious)
File Path Features
path_depth                  # Directory nesting level
is_system_path              # In Windows/System32/Program Files
is_temp_path                # In Temp/AppData directories
file_extension              # Categorical encoding
filename_length             # Abnormally long names (evasion)
path_entropy                # Randomness score (obfuscation detection)
Event Pattern Features
event_type_encoded          # One-hot or label encoding
consecutive_same_events     # Repetition counter
rare_event_type             # Statistical frequency score
event_type_count_per_file   # Unique operations per file
Cross-Artifact Features (if using merged data)
appears_in_both_artifacts   # Binary: present in LogFile AND UsnJrnl
timestamp_mismatch_seconds  # Delta between LogFile & UsnJrnl times
Output: data/processed/master_timeline_features.csv
Phase 3: Model Training & Evaluation
A. Data Splitting Strategy
Case-Based Stratification (Recommended):
Train: Cases 01-08 (66%)
Validation: Cases 09-10 (17%)
Test: Cases 11-12 (17%)
Rationale: Prevents data leakage from same case appearing in train/test Class Imbalance Handling:
SMOTE oversampling for minority class
Class weight adjustment (class_weight='balanced')
Stratified sampling preserving positive label ratio
B. Model Architectures
Random Forest Classifier
- n_estimators: 100-500 (tuned)
- max_depth: 10-30 (prevent overfitting)
- min_samples_split: 2-10
- class_weight: 'balanced_subsample'
- Strengths: Feature importance, interpretability, robust to outliers
XGBoost Classifier
- learning_rate: 0.01-0.1
- max_depth: 5-10
- scale_pos_weight: Ratio of negative to positive samples
- Strengths: Higher accuracy, better handling of imbalance
Ensemble Voting Classifier (Optional)
Combines RF + XGBoost predictions
Soft voting with probability averaging
C. Evaluation Metrics
Given extreme class imbalance (252 positives / ~860K total): Primary Metrics:
Precision: Minimize false positives (critical for forensic workflow)
Recall: Catch actual timestomped files
F1-Score: Harmonic mean balance
AUC-ROC: Overall discriminative ability
Precision-Recall Curve: Better for imbalanced data
Forensic-Specific Metrics:
False Positive Rate @ 95% Recall: Acceptable false alarm rate
Top-K Accuracy: Precision in top 100/500 predictions
D. Model Interpretability
- Feature importance ranking (Random Forest built-in)
- SHAP values for individual predictions
- Confusion matrix analysis
- Example case studies of TP/FP/FN
E. Confidence Score Calibration
Map probability outputs to risk levels:
High Risk: P(timestomped) > 0.8
Medium Risk: 0.5 < P < 0.8
Low Risk: 0.3 < P < 0.5
Benign: P < 0.3
Output:
Trained models: models/random_forest_model.pkl, models/xgboost_model.pkl
Evaluation report: outputs/model_evaluation.md
Feature importance: outputs/feature_importance.csv
🔌 Autopsy Plugin Integration (Future Work)
Architecture
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
Key Features
Real-time artifact parsing (MFTECmd, LogFileParser integration)
Confidence-based filtering (reduce analyst workload)
Detailed report with SHAP explanations
Comparison with Oh et al.'s algorithm results
📊 Expected Outcomes
Improved Detection Accuracy: >95% precision while maintaining >90% recall
Reduced False Positives: 50-70% reduction vs. rule-based methods
Confidence Scoring: Risk-stratified output for investigative prioritization
Interpretable Results: Feature importance + SHAP for court admissibility
Scalable Plugin: Automated workflow integrated into Autopsy forensic platform
📚 References
Oh, J. et al. (2024). "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation." IEEE Access.
"Detection of Timestamps Tampering in NTFS using Machine Learning." Procedia Computer Science, 2019.
"De-Wipimization: Detection of data wiping traces for investigating NTFS file system." Computers & Security, 2020.
MITRE ATT&CK: T1070.006 - Indicator Removal: Timestomp
🛠️ Project Structure
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