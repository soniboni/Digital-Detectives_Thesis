**🕵️ Digital Detectives – Autopsy Plugin Prototype**

This repository is part of our thesis project, which aims to develop a prototype Autopsy plugin capable of detecting timestomped files through in-depth analysis of $LogFile and $UsnJrnl artifacts. The system integrates machine learning models to enhance detection accuracy.

**🚧 Work in Progress – The plugin is currently under active development.**

---

## 📊 Current Project Status

**Current Phase:** Phase 2.2 - Feature Engineering (In Progress)

**Branch:** `dataset-soni`

**Last Updated:** 2025-10-08

---

## 🗂️ Dataset Overview

The project analyzes **12 forensic case datasets** (01-PE through 12-PE), each containing:
- **$LogFile** artifacts: LSN-based NTFS file system event logs
- **$UsnJrnl** artifacts: Update Sequence Number journal entries
- **Suspicious indicators**: Manually labeled timestomping and malicious execution events

**Total Records:** ~3.19 million timeline entries (before filtering)

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

### ✅ Phase 2.1: Data Cleaning & Merging
**Status:** Complete

**Objective:** Clean, standardize, and merge LogFile + UsnJrnl into unified timeline

**Process:**
- Standardized column names to lowercase
- Converted all timestamps to UTC datetime objects
- Merged LogFile and UsnJrnl on `fullpath` + time proximity
- Created master timeline with unified schema
- Added `Case_ID` column (1-12) for multi-case analysis

**Key Metrics:**
- LogFile records: ~62K (across all 12 cases)
- UsnJrnl records: ~3.1M (across all 12 cases)
- Combined master timeline: **3,190,725 rows**

**Output:**
- `data/processed/Phase 2.1 - Data Cleaning/Master_LogFile_Cleaned.csv`
- `data/processed/Phase 2.1 - Data Cleaning/Master_UsnJrnl_Cleaned.csv`
- `data/processed/phase 2.1 - data merged (all sub-folders)/MASTER_TIMELINE_ALL_CASES.csv`

**Notebooks:**
- `notebooks/display notebooks/Phase 2.1 - Merging All Sub-Folder.ipynb`
- `notebooks/presentation notebooks/Phase 2 - Data Cleaning.ipynb`

---

### 🔄 Phase 2.2: Feature Engineering
**Status:** In Progress

**Objective:** Reduce noise and calculate forensic features for ML model training

#### Step 1: UsnJrnl Event Filtering (Noise Reduction)
**Goal:** Filter out low-value system noise events

**Strategy:** Keep only high-forensic-value UsnJrnl events that indicate file manipulation:
- `Basic_Info_Changed` – Direct MAC timestamp modification (CRITICAL)
- `File_Created` – File creation events
- `File_Renamed` – File renaming
- `Data_Overwritten` / `Data_Added` / `Data_Truncated` – Content modifications

**Events Excluded:**
- Permission changes (`Access_Right_Changed`, `SECURITY_CHANGE`)
- Internal system operations (`Transacted_Changed`, `Reparse_Point_Changed`)
- Non-contextual `File_Closed` events

**Results:**
- **Rows dropped:** 951,307 (29.81% reduction)
- **Filtered dataset:** 2,239,418 rows
- All LogFile records retained (62K rows)
- UsnJrnl reduced to 2.18M high-value records

**Output:** `data/processed/phase 3 - feature engineered/MASTER_TIMELINE_FILTERED.csv`

#### Step 2: Time Delta Feature Calculation
**Goal:** Calculate temporal anomaly features for timestomping detection

**Features Calculated (6 total):**

| Feature | Formula | Forensic Significance |
|---------|---------|----------------------|
| `Delta_MFTM_vs_M` | MFT Modified - Modified | **Most critical:** Detects direct timestamp manipulation |
| `Delta_M_vs_C` | Modified - Creation | File age since creation |
| `Delta_C_vs_A` | Creation - Accessed | Time before first access |
| `Delta_Event_vs_M` | Event Timestamp - Modified | Event-to-modification lag |
| `Delta_Event_vs_MFTM` | Event Timestamp - MFT Modified | Event-to-MFT lag |
| `Delta_Event_vs_C` | Event Timestamp - Creation | Event-to-creation lag |

**Output:** `data/processed/Phase 2.2 - Feature Engineering/MASTER-TIMELINE-FEATURES-CALCULATED.csv`

**Notebook:** `notebooks/presentation notebooks/Phase 2 - Data Cleaning and Feature Engineering.ipynb`

---

### 📋 Upcoming Phases

#### Phase 3: Additional Feature Engineering (Planned)
- Aggregate statistics per file (event count, unique timestamps)
- File extension categorization
- Directory depth features
- File size change tracking

#### Phase 4: Feature Preprocessing (Planned)
- Outlier handling
- Feature scaling/normalization
- Encoding categorical variables
- Train/test split preparation

#### Phase 5: Model Training (Planned)
- **Algorithm:** Isolation Forest (unsupervised anomaly detection)
- Hyperparameter tuning
- Model evaluation metrics
- Cross-validation across 12 case datasets

#### Phase 6: Plugin Development (Planned)
- Integration with Autopsy framework
- Real-time forensic artifact parsing
- Visualization dashboard
- Export capabilities

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

```
Digital-Detectives_Thesis/
├── data/
│   ├── raw/                          # Original forensic artifacts
│   │   ├── 01-PE-LogFile.csv         # Case 1 LogFile
│   │   ├── 01-PE-UsnJrnl.csv         # Case 1 UsnJrnl
│   │   ├── suspicious/               # Manual suspicious file indicators
│   │   └── ... (12 cases total)
│   └── processed/
│       ├── Phase 1 - Data Labeling/  # Labeled datasets
│       ├── Phase 2.1 - Data Cleaning/# Cleaned & merged data
│       └── Phase 2.2 - Feature Engineering/ # Feature-rich datasets
├── notebooks/
│   ├── presentation notebooks/       # Clean, documented notebooks
│   ├── display notebooks/            # Analysis notebooks
│   └── specific dataset notebooks/   # Per-case notebooks
├── models/                           # Saved ML models (future)
├── outputs/                          # Analysis outputs
├── src/                              # Source code (future plugin)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Getting Started

### Prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running Notebooks
```bash
jupyter lab
# Navigate to notebooks/presentation notebooks/
```

### Current Work Session
**Focus:** Phase 2.2 - Feature Engineering
**Next Steps:**
1. Validate calculated time delta features
2. Explore feature distributions for anomalies
3. Begin Phase 3: Additional contextual features

---

## 📝 Notes for Resuming Work

**Git Status:**
- Modified: `data/processed/Phase 1 - Data Labeling/01-PE-LogFile_labeled.csv`
- New files in Phase 2.1 and 2.2 directories (not yet committed)

**Key Files to Check:**
- Latest feature dataset: `data/processed/Phase 2.2 - Feature Engineering/MASTER-TIMELINE-FEATURES-CALCULATED.csv`
- Working notebook: `notebooks/presentation notebooks/Phase 2 - Data Cleaning and Feature Engineering.ipynb`

**Known Issues/TODOs:**
- [ ] Commit Phase 2.2 feature engineering outputs
- [ ] Validate time delta calculations on known timestomped files
- [ ] Document data dictionary for all 27 columns in feature dataset

---

## 👥 Team

**Project:** Digital Detectives Thesis
**Institution:** [Your University]
**Supervisor:** [Supervisor Name]

---

## 📄 License

[Add your license information here]
