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

## 🔬 Current Focus: Data Handling & Model Development

Currently in the **data preparation and model experimentation phase**, focusing on:

### ✅ Completed:
- Initial data collection (12 forensic cases)
- Preliminary data labeling and merging strategies

### 🔄 In Progress:
- **Phase 1: Data Cleaning & Smart Merging**
  - Implementing research-backed correlation methodology (based on "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation" by Oh et al., 2024)
  - Cross-artifact pattern detection (BASIC_INFO_CHANGE + CLOSE in UsnJrnl, UpdateResidentValue in LogFile)
  - File system tunneling identification to reduce false positives
  - Smart union strategy to preserve both matched and single-source events

### 📋 Next Steps:
- Feature engineering (temporal, anomaly, cross-artifact features)
- Baseline model training (Random Forest)
- Comparative evaluation of 5 algorithms
- Hyperparameter optimization and model selection
- Autopsy module integration

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
| **Phase 1: Data Cleaning** | 🔄 In Progress | Smart merging of $LogFile and $UsnJrnl with cross-artifact correlation | Clean, merged datasets per case with source indicators |
| **Phase 2: Feature Engineering** | 📋 Planned | Extract temporal, anomaly, and behavioral features | ML-ready feature vectors (~80-100 features) |
| **Phase 3: Baseline Model** | 📋 Planned | Train Random Forest as baseline | Performance benchmarks (precision, recall, AUC) |
| **Phase 4: Algorithm Comparison** | 📋 Planned | Evaluate 5 algorithms on same dataset | Comparative performance metrics, best model selection |
| **Phase 5: Model Optimization** | 📋 Planned | Hyperparameter tuning, ensemble methods | Optimized production model |
| **Phase 6: Autopsy Integration** | 📋 Planned | Develop Ingest Module with best model | Deployable Autopsy plugin |

---

## 📂 Project Structure

```
Digital-Detectives_Thesis/
├── data/
│   ├── raw/
│   │   ├── logfile/                    # Raw $LogFile CSVs (12 cases)
│   │   ├── usnjrnl/                    # Raw $UsnJrnl CSVs (12 cases)
│   │   └── suspicious/                 # Ground truth labels
│   └── processed/
│       └── Phase 1 - Data Cleaning/    # 🔄 Current work
│           ├── case_1_merged.csv
│           ├── case_2_merged.csv
│           └── ...
│
├── notebooks/
│   └── Phase 1 - Data Cleaning/        # 🔄 Current work
│       ├── 01_Smart_Union_Merging.ipynb
│       └── Forensic_Detection_of_Timestamp_Manipulation_for_D.pdf
│
├── Autopsy File Ingest Module/         # 📋 Future work
│   └── (to be developed)
│
└── README.md                            # This file
```

---

## 🔑 Key Research Foundation

This project builds upon the methodology proposed by **Oh, Lee, and Hwang (2024)** in their paper ["Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"](https://ieeexplore.ieee.org/document/10517044) published in IEEE Access.
