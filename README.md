**🕵️ Digital Detectives – Autopsy Plugin Prototype**

This repository is part of our thesis project, which aims to develop a prototype Autopsy plugin capable of detecting timestomped files through in-depth analysis of $LogFile and $UsnJrnl artifacts. The system integrates machine learning models to enhance detection accuracy.

**🚧 Work in Progress – The plugin is currently under active development.**

---

## 📊 Current Project Status

**Current Phase:** Phase 3 - Feature Engineering (Complete)

**Branch:** `dataset-soni`

**Last Updated:** 2025-10-08

---

## 🗂️ Dataset Overview

The project analyzes **12 forensic case datasets** (01-PE through 12-PE), each containing:
- **$LogFile** artifacts: LSN-based NTFS file system event logs
- **$UsnJrnl** artifacts: Update Sequence Number journal entries
- **Suspicious indicators:** Manually labeled timestomping and malicious execution events
- **ML Algorithm:** Isolation Forest (unsupervised anomaly detection)

**Initial Records:** ~3.19 million timeline entries
**After Cleaning:** 2.26 million high-quality forensic events

---

## 🔄 Data Processing Pipeline

### ✅ Phase 1: Data Labeling
**Status:** Complete

**Objective:** Label suspicious activities in raw forensic data

**Process:**
- Loaded 12 LogFile datasets (39K–40K records each)
- Loaded 12 UsnJrnl datasets (300K+ records each)
- Matched suspicious file indicators from manual analysis
- Added binary labels:
  - `is_timestomped`: Files with manipulated MAC timestamps
  - `is_suspicious_execution`: Execution of potentially malicious programs

**Output:**
- 24 labeled CSV files (12 LogFile + 12 UsnJrnl)
- Location: `data/processed/Phase 1 - Data Labeling/`

**Notebook:** `notebooks/presentation notebooks/Phase 1 - Data Labeling.ipynb`

---

### ✅ Phase 2: Data Cleaning
**Status:** Complete

**Objective:** Clean and standardize LogFile and UsnJrnl datasets separately

#### Part A: LogFile Cleaning

**Process:**
- Dropped rows with null `event` AND `detail` (46.19% reduction)
- Dropped irrelevant columns (`targetvcn`, `clusterindex`)
- Conditional imputation for `eventtime(utc+8)` based on event type
- Extracted timestamps from `detail` column (recovered 12,675 timestamps)
- Preserved all 14 timestomped + 8 suspicious labeled rows
- Converted timestamps to datetime format (UTC)
- Handled missing `fullpath` and `file/directory name`

**Key Metrics:**
- Initial: 243,884 records
- Final: **83,458 records**
- Reduction: 65.77%
- All 22 labeled rows preserved ✅

**Output:**
- `data/processed/Phase 2 - Data Cleaning/Master_LogFile_Cleaned.csv`

#### Part B: UsnJrnl Cleaning

**Process:**
- Dropped irrelevant columns (`sourceinfo`, `carving flag`)
- **Filtered low-value events** (30.28% reduction):
  - Kept: HIGH_VALUE events (`Basic_Info_Changed`, `File_Created`, `Data_Overwritten`, etc.)
  - Dropped: LOW_VALUE events (`File_Closed/Deleted`, `Access_Right_Changed`, etc.)
- Handled missing `fullpath` (9.93% missing, acceptable)
- Converted `timestamp(utc+8)` to datetime format
- Renamed `Case_ID` to `case_id` and moved to first column
- Preserved all 238 timestomped + 8 suspicious labeled rows

**Key Metrics:**
- Initial: 3,128,446 records
- Final: **2,181,063 records**
- Reduction: 30.28%
- All 246 labeled rows preserved ✅
- Timestamp completeness: 100% ✅

**Output:**
- `data/processed/Phase 2 - Data Cleaning/Master_UsnJrnl_Cleaned.csv`

**Notebooks:**
- `notebooks/presentation notebooks/Phase 2 - Data Cleaning.ipynb`

---

### ✅ Phase 2.1: Data Merging
**Status:** Complete

**Objective:** Merge cleaned LogFile and UsnJrnl into unified temporal timeline

**Process:**
- Loaded both cleaned master datasets
- Added `source` identifier column (LogFile/UsnJrnl)
- Standardized column names (`fullpath`, `timestamp_primary`)
- Merged datasets vertically (concatenation)
- Sorted by `case_id` and `timestamp_primary`
- Reordered columns for readability

**Key Metrics:**
- LogFile: 83,458 records (3.7%)
- UsnJrnl: 2,181,063 records (96.3%)
- **Master Timeline: 2,264,521 records**
- Labeled rows: 252 timestomped + 16 suspicious = **268 total**
- Temporal coverage: ~24 years

**Output:**
- `data/processed/Phase 2.1 - Data Merging/Master_Timeline.csv`

**Notebook:**
- `notebooks/presentation notebooks/Phase 2.1 - Data Merging.ipynb`

---

### ✅ Phase 3: Feature Engineering
**Status:** Complete

**Objective:** Calculate temporal and contextual features for timestomping detection

**Process:**

**Time Delta Features (6 features - CRITICAL for timestomping):**
- `Delta_MFTM_vs_M`: MFT Modified vs Modified timestamp delta
- `Delta_M_vs_C`: Modified vs Creation timestamp delta
- `Delta_C_vs_A`: Creation vs Accessed timestamp delta
- `Delta_A_vs_MFTM`: Accessed vs MFT Modified timestamp delta
- `Delta_Event_vs_C`: Event time vs Creation timestamp delta
- `Delta_Event_vs_M`: Event time vs Modified timestamp delta

**Temporal Features (8 features):**
- `hour`, `day_of_week`, `day_of_month`, `month`, `year`
- `is_weekend`: Binary flag for weekend activity
- `is_business_hours`: Binary flag for 9am-5pm weekday activity
- `days_since_epoch`: Days since Unix epoch (temporal distance)

**File Path Features (3 features):**
- `file_extension`: Extracted file extension for categorization
- `path_depth`: Directory depth (indicator of nested/hidden files)
- `is_system_file`: Binary flag for Windows system paths

**Categorical Encoding (2 features):**
- `source_encoded`: LogFile (0) vs UsnJrnl (1)
- `event_type_encoded`: Numeric encoding of event types

**Key Metrics:**
- Initial: 2,264,521 records (25 features)
- Final: **2,264,521 records (44 features)**
- Features added: 19 engineered features
- All 268 labeled rows preserved ✅
- Delta feature completeness: 2.5-3.1% (expected - LogFile only)
- Temporal feature completeness: 99.99% ✅

**Critical Observations:**
- Time delta ranges: ±756M seconds (±24 years) - **outliers identified for Phase 4 clipping**
- 74.6% records are code/executable files (forensically relevant)
- 9.0% business hours activity, 13.7% weekend activity

**Output:**
- `data/processed/Phase 3 - Feature Engineering/Master_Timeline_Features.csv` (866 MB)

**Notebook:**
- `notebooks/presentation notebooks/Phase 3 - Feature Engineering.ipynb`

---

### 📋 Upcoming Phases

#### Phase 4: Feature Preprocessing (Next)
**Objective:** Prepare features for machine learning model training

**Planned Tasks:**
- **Outlier handling:** Clip time delta features to ±10 years (±315,360,000 seconds)
- **Categorical encoding:**
  - One-hot or target encoding for `event_type`
  - Group rare file extensions, then encode
- **Feature scaling:** StandardScaler for numerical features (deltas, depth, temporal)
- **Train/Val/Test split:** 70/15/15 stratified by `case_id` (prevent data leakage)
  - **Important:** Stratify by case_id, NOT random split
  - Ensures model tested on completely unseen forensic cases
- **Class imbalance:** Not a concern for Isolation Forest (unsupervised)
  - Labels only used for evaluation, not training

**Target Output:**
- `data/processed/Phase 4 - Preprocessing/X_train.csv`, `X_val.csv`, `X_test.csv`
- `data/processed/Phase 4 - Preprocessing/y_train.csv`, `y_val.csv`, `y_test.csv`

---

#### Phase 5: Model Training (Planned)
- **Algorithm:** Isolation Forest (unsupervised anomaly detection)
- Hyperparameter tuning (contamination, n_estimators, max_features)
- Model evaluation:
  - Precision, Recall, F1-Score
  - ROC-AUC curve
  - Confusion matrix
- Cross-validation across 12 case datasets
- Feature importance analysis

---

#### Phase 6: Plugin Development (Planned)
- Integration with Autopsy framework
- Real-time forensic artifact parsing
- Visualization dashboard for detected anomalies
- Export capabilities (CSV, JSON reports)
- User documentation

---

## 🛠️ Tech Stack

**Languages & Libraries:**
- Python 3.x
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- matplotlib, seaborn (visualization)
- Jupyter Notebook (interactive analysis)

**Development Environment:**
- Virtual environment: `.venv/`
- Git version control

---

## 📁 Project Structure

Digital-Detectives_Thesis/ ├── data/ │ ├── raw/ # Original forensic artifacts │ │ ├── 01-PE-LogFile.csv # Case 1 LogFile │ │ ├── 01-PE-UsnJrnl.csv # Case 1 UsnJrnl │ │ ├── suspicious/ # Manual suspicious file indicators │ │ └── ... (12 cases total) │ └── processed/ │ ├── Phase 1 - Data Labeling/ # Labeled datasets │ ├── Phase 2 - Data Cleaning/ # Cleaned LogFile & UsnJrnl │ ├── Phase 2.1 - Data Merging/ # Merged Master Timeline │ ├── Phase 3 - Feature Engineering/ # Feature-rich datasets │ └── Phase 4 - Preprocessing/ # Train/Val/Test splits (next) ├── notebooks/ │ ├── presentation notebooks/ # Clean, documented notebooks │ │ ├── Phase 1 - Data Labeling.ipynb │ │ ├── Phase 2 - Data Cleaning.ipynb │ │ ├── Phase 2.1 - Data Merging.ipynb │ │ └── Phase 3 - Feature Engineering.ipynb │ ├── display notebooks/ # Analysis notebooks │ └── specific dataset notebooks/ # Per-case notebooks ├── models/ # Saved ML models (future) ├── outputs/ # Analysis outputs ├── src/ # Source code (future plugin) ├── requirements.txt # Python dependencies └── README.md # This file

---

## 🚀 Getting Started

### Prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Running Notebooks
jupyter lab
# Navigate to notebooks/presentation notebooks/
## 🎯 Current Work Session

**Focus:** Completed Phase 3 - Feature Engineering

**Next Steps:**
- Begin Phase 4: Feature Preprocessing
- Clip extreme outliers in time delta features (±10 years)
- Encode categorical variables (event_type, file_extension)
- Apply StandardScaler to numerical features
- Stratified train/val/test split by case_id (70/15/15)
## 📝 Notes for Resuming Work

**Completed:**
- ✅ Phase 1: Data Labeling (24 labeled datasets)
- ✅ Phase 2: Data Cleaning (LogFile + UsnJrnl separately cleaned)
- ✅ Phase 2.1: Data Merging (Unified Master Timeline created)
- ✅ Phase 3: Feature Engineering (44 features engineered)

**Current Datasets:**
- `Master_LogFile_Cleaned.csv`: 83,458 records
- `Master_UsnJrnl_Cleaned.csv`: 2,181,063 records
- `Master_Timeline.csv`: 2,264,521 records (merged)
- `Master_Timeline_Features.csv`: 2,264,521 records, 44 features (866 MB)

**Key Files:**
- Latest dataset: `data/processed/Phase 3 - Feature Engineering/Master_Timeline_Features.csv`
- Working notebook: `notebooks/presentation notebooks/Phase 3 - Feature Engineering.ipynb`

**Labeled Data Summary:**
- Total timestomped events: 252 (across all sources)
- Total suspicious execution events: 16
- Total labeled for evaluation: **268 events** (0.01% of dataset)

**Next Phase:**
- Create Phase 4 notebook for Feature Preprocessing
- Handle extreme outliers (time deltas ±24 years → clip to ±10 years)
- Encode categorical variables and scale numerical features
- **CRITICAL:** Stratified split by `case_id` to prevent data leakage
## 📊 Data Pipeline Summary

| Phase | Input | Output | Records | Features | Status |
|-------|-------|--------|---------|----------|--------|
| Phase 1 | Raw CSVs (12 cases) | 24 labeled CSVs | 3.37M | - | ✅ Complete |
| Phase 2 | 24 labeled CSVs | 2 cleaned masters | 2.26M | 25 | ✅ Complete |
| Phase 2.1 | 2 cleaned masters | Unified timeline | 2.26M | 25 | ✅ Complete |
| Phase 3 | Unified timeline | Feature matrix | 2.26M | 44 | ✅ Complete |
| Phase 4 | Feature matrix | Train/Val/Test splits | TBD | TBD | 🔜 Next |
| Phase 5 | Preprocessed data | Trained model | N/A | N/A | ⏳ Planned |
👥 Team
Project: Digital Detectives Thesis Institution: [Your University] Supervisor: [Supervisor Name]
📄 License
[Add your license information here]
 
---

**Key Changes:**
1. ✅ Updated status to Phase 2.1 Complete
2. ✅ Documented Phase 2 (LogFile + UsnJrnl cleaning separately)
3. ✅ Added Phase 2.1 (Data Merging) with full details
4. ✅ Updated metrics (2.26M final records, 268 labeled events)
5. ✅ Clarified next steps: Phase 3 Feature Engineering
6. ✅ Added data pipeline summary table
7. ✅ Updated project structure and file paths