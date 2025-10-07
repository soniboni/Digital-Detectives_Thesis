**🕵️ Digital Detectives – Autopsy Plugin Prototype**

This repository is part of our thesis project, which aims to develop a prototype Autopsy plugin capable of detecting timestomped files through in-depth analysis of $LogFile and $UsnJrnl artifacts. The system integrates machine learning models to enhance detection accuracy.

**🚧 Work in Progress – The plugin is currently under active development.**

---

## 📊 Current Project Status

**Current Phase:** Phase 2.1 - Data Merging (Complete)

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

### 📋 Upcoming Phases

#### Phase 3: Feature Engineering (Next)
**Objective:** Calculate temporal and contextual features for timestomping detection

**Planned Tasks:**
- Calculate time delta features:
  - `Delta_MFTM_vs_M` (MFT Modified vs Modified) - **CRITICAL**
  - `Delta_M_vs_C` (Modified vs Creation)
  - `Delta_C_vs_A` (Creation vs Accessed)
  - Event timestamp deltas
- Aggregate statistics per file (event count, unique timestamps)
- File extension categorization
- Directory depth features
- Temporal pattern features (hour-of-day, day-of-week)

**Target Output:**
- `data/processed/Phase 3 - Feature Engineering/Master_Timeline_Features.csv`

---

#### Phase 4: Feature Preprocessing (Planned)
- Handle outliers in time delta features
- Feature scaling/normalization (StandardScaler)
- Encode categorical variables (event types, file attributes)
- Create train/test split (stratified by case_id)
- Handle class imbalance (SMOTE or class weights)

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

Digital-Detectives_Thesis/ ├── data/ │ ├── raw/ # Original forensic artifacts │ │ ├── 01-PE-LogFile.csv # Case 1 LogFile │ │ ├── 01-PE-UsnJrnl.csv # Case 1 UsnJrnl │ │ ├── suspicious/ # Manual suspicious file indicators │ │ └── ... (12 cases total) │ └── processed/ │ ├── Phase 1 - Data Labeling/ # Labeled datasets │ ├── Phase 2 - Data Cleaning/ # Cleaned LogFile & UsnJrnl │ ├── Phase 2.1 - Data Merging/ # Merged Master Timeline │ └── Phase 3 - Feature Engineering/ # Feature-rich datasets (next) ├── notebooks/ │ ├── presentation notebooks/ # Clean, documented notebooks │ │ ├── Phase 1 - Data Labeling.ipynb │ │ ├── Phase 2 - Data Cleaning.ipynb │ │ └── Phase 2.1 - Data Merging.ipynb │ ├── display notebooks/ # Analysis notebooks │ └── specific dataset notebooks/ # Per-case notebooks ├── models/ # Saved ML models (future) ├── outputs/ # Analysis outputs ├── src/ # Source code (future plugin) ├── requirements.txt # Python dependencies └── README.md # This file

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
Current Work Session
Focus: Completed Phase 2.1 - Data Merging Next Steps:
Begin Phase 3: Feature Engineering
Calculate time delta features from Master Timeline
Extract temporal patterns and contextual features
Prepare feature matrix for ML model training
📝 Notes for Resuming Work
Completed:
✅ Phase 1: Data Labeling (24 labeled datasets)
✅ Phase 2: Data Cleaning (LogFile + UsnJrnl separately cleaned)
✅ Phase 2.1: Data Merging (Unified Master Timeline created)
Current Datasets:
Master_LogFile_Cleaned.csv: 83,458 records
Master_UsnJrnl_Cleaned.csv: 2,181,063 records
Master_Timeline.csv: 2,264,521 records (merged)
Key Files:
Latest dataset: data/processed/Phase 2.1 - Data Merging/Master_Timeline.csv
Working notebook: notebooks/presentation notebooks/Phase 2.1 - Data Merging.ipynb
Labeled Data Summary:
Total timestomped events: 252 (across all sources)
Total suspicious execution events: 16
Total labeled for training: 268 events
Next Phase:
Create Phase 3 notebook for Feature Engineering
Focus on time delta calculations (critical for timestomping detection)
Prepare features for Isolation Forest model
📊 Data Pipeline Summary
Phase	Input	Output	Records	Status
Phase 1	Raw CSVs (12 cases)	24 labeled CSVs	3.37M	✅ Complete
Phase 2	24 labeled CSVs	2 cleaned masters	2.26M	✅ Complete
Phase 2.1	2 cleaned masters	Unified timeline	2.26M	✅ Complete
Phase 3	Unified timeline	Feature matrix	TBD	🔜 Next
Phase 4	Feature matrix	Preprocessed data	TBD	⏳ Planned
Phase 5	Preprocessed data	Trained model	N/A	⏳ Planned
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