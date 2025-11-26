# 🕵️ Digital Detectives – Timestamp Manipulation Detection for NTFS

**A Machine Learning-Based Approach for Detecting Timestomped Files Using $LogFile and $UsnJrnl Cross-Artifact Analysis**

This repository contains the complete thesis project focused on developing and evaluating machine learning models capable of detecting timestamp manipulation (timestomping) in NTFS file systems through intelligent correlation of $LogFile and $UsnJrnl artifacts.

---

## 🎯 Project Objectives

This thesis project aims to:

1. **Develop a cross-artifact correlation methodology** for merging $LogFile and $UsnJrnl data that preserves temporal relationships and detection patterns from both artifacts

2. **Engineer forensically-relevant features** that capture timestamp manipulation signatures, including:
   - Temporal anomaly patterns (impossible timestamp sequences)
   - Cross-artifact consistency checks
   - Event frequency and behavioral patterns
   - Tool-specific signatures

3. **Evaluate 5 machine learning algorithms** for timestomping detection:
   - **Random Forest** (baseline model)
   - **Binary Logistic Regression**
   - **XGBoost**
   - **LightGBM**
   - **Neural Network with Focal Loss** (for handling extreme class imbalance)

4. **Produce confidence-based predictions** with three-tier risk categorization:
   - **HIGH**: Strong evidence of timestamp manipulation (multiple indicators)
   - **MEDIUM**: Suspicious patterns requiring investigation
   - **LOW**: Minimal indicators, likely benign

5. **Develop an Autopsy Ingest Module** that integrates the best-performing model for operational use, providing:
   - Automatic detection and parsing of $LogFile and $UsnJrnl artifacts
   - Real-time timestamp manipulation detection during forensic acquisition
   - Structured output files for investigator review

---

## 🔬 Current Progress

### ✅ Phase 1: Data Cleaning & Preparation - **COMPLETED**

**Phase 1A: Smart Union Merging**
- ✅ Implemented research-backed correlation methodology (Oh et al., 2024)
- ✅ Cross-artifact pattern detection (BASIC_INFO_CHANGE + CLOSE in UsnJrnl, Time Reversal in LogFile)
- ✅ File system tunneling identification to reduce false positives
- ✅ Smart union strategy preserving both matched and single-source events
- ✅ **Output**: 154,550 records from 3.2M raw entries (95.4% reduction), 252 timestomped events captured (100%)

**Phase 1B: Column Analysis & Cleanup**
- ✅ Parsed `lf_detail` text field to extract structured timestamp manipulation data
- ✅ Created 18 new ML-ready features (before/after timestamps, delta calculations, manipulation indicators)
- ✅ Removed 7 useless columns (empty or no variance)
- ✅ Organized dataset: `case_id` first, sorted by case (1→12)
- ✅ **Output**: 38 columns, all 252 timestomped events preserved

**Key Achievements**:
- Zero data loss (100% of ground truth labels preserved)
- Extracted structured timestamp manipulation patterns from unstructured text
- Identified and removed columns with zero ML information value
- Dataset ready for feature engineering

### 🔄 Currently Working On:
- **Phase 2: Feature Engineering** (see detailed plan below)

### 📋 Next Steps:
- Phase 3: Baseline model training (Random Forest)
- Phase 4: Comparative evaluation of 5 algorithms
- Phase 5: Hyperparameter optimization and model selection
- Phase 6: Autopsy module integration

---

## 📊 Expected Model Outputs

The final detection system will generate three output files:

### 1. `predictions.csv`
Complete predictions for all analyzed files with:
- File path and name
- Prediction label (0=benign, 1=timestomped)
- Confidence score (0.0 to 1.0)
- Risk level (HIGH/MEDIUM/LOW)
- Contributing features and detection factors

### 2. `flagged_files.csv`
Filtered list containing only timestomped predictions (confidence > threshold) for investigator triage, prioritized by risk level

### 3. `summary_report.txt`
Human-readable summary including:
- Total files analyzed
- Number of flagged files by risk level
- Detection statistics (precision, recall, F1-score if ground truth available)
- Top detection patterns identified
- Recommended investigation priorities

---

## 🛠️ Methodology Overview

| Phase | Status | Objective | Key Deliverable |
|:------|:------:|:----------|:----------------|
| **Phase 1A: Smart Merging** | ✅ Complete | Cross-artifact correlation of $LogFile and $UsnJrnl | 154,550 merged records, 252 timestomped events |
| **Phase 1B: Column Cleanup** | ✅ Complete | Parse text fields, remove useless columns | 38 ML-ready columns with structured features |
| **Phase 2: Feature Engineering** | 🔄 In Progress | Extract temporal, anomaly, behavioral, and file-level features | ~60-80 ML-ready features |
| **Phase 3: Baseline Model** | 📋 Planned | Train Random Forest as baseline | Performance benchmarks (precision, recall, F1, AUC) |
| **Phase 4: Algorithm Comparison** | 📋 Planned | Evaluate 5 algorithms on same dataset | Comparative performance metrics, best model selection |
| **Phase 5: Model Optimization** | 📋 Planned | Hyperparameter tuning, ensemble methods | Optimized production model |
| **Phase 6: Autopsy Integration** | 📋 Planned | Develop Ingest Module with best model | Deployable Autopsy plugin (LogFile + UsnJrnl only) |

---

## 📂 Project Structure

```
Digital-Detectives_Thesis/
├── data/
│   ├── raw/
│   │   ├── logfile/                         # Raw $LogFile CSVs (12 cases)
│   │   ├── usnjrnl/                         # Raw $UsnJrnl CSVs (12 cases)
│   │   └── suspicious/                      # Ground truth labels
│   └── processed/
│       ├── Phase 1 - Data Cleaning/         # ✅ Complete
│       │   ├── all_cases_combined.csv       # 154,550 records, 27 columns
│       │   ├── smart_union_summary.csv      # Phase 1A statistics
│       │   └── ground_truth_*.csv           # Labeled datasets
│       └── Phase 1B - Column Cleanup/       # ✅ Complete
│           └── all_cases_combined_clean.csv # 154,550 records, 38 columns
│
├── notebooks/
│   ├── Phase 1 - Data Cleaning/             # ✅ Complete
│   │   ├── 01_Smart_Union_Merging.ipynb
│   │   ├── 01B_Column_Analysis_and_Cleanup.ipynb
│   │   └── Forensic_Detection_of_Timestamp_Manipulation_for_D.pdf
│   └── Phase 2 - Feature Engineering/       # 🔄 Next
│
├── Autopsy File Ingest Module/              # 📋 Future work
│   └── (to be developed - LogFile + UsnJrnl only, MFT excluded)
│
└── README.md                                 # This file
```

---

## 🔑 Key Research Foundation

This project builds upon the methodology proposed by **Oh, Lee, and Hwang (2024)** in their paper ["Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"](https://ieeexplore.ieee.org/document/10517044) published in IEEE Access.

**Key Findings from Research Applied**:
- NTFS journal-based detection (LogFile + UsnJrnl) is the most effective method for detecting timestamp manipulation
- Direct detection of timestamp change events provides "Which" (file) and "When" (time) information critical for timeline analysis
- File system tunneling causes false positives in naive detection approaches - must be identified and filtered
- Cross-artifact correlation increases detection confidence (HIGH when both artifacts agree)
- Additional indicators (zero nanoseconds, copied timestamps, $FN manipulation) improve detection accuracy

**Autopsy Module Scope** (Phase 6):
- ✅ **$LogFile** - Direct detection of Time Reversal events
- ✅ **$UsnJrnl** - Detection via BASIC_INFO_CHANGE + CLOSE pattern
- ❌ **$MFT excluded** - Indirect detection only, lower reliability, not suitable for automated module
