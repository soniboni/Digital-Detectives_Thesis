# 🕵️ Digital Detectives – Timestamp Manipulation Detection for NTFS

**A Machine Learning-Based Approach for Detecting Timestomped Files Using $LogFile and $UsnJrnl Cross-Artifact Analysis**

This repository contains the complete thesis project focused on developing and evaluating machine learning models capable of detecting timestamp manipulation (timestomping) in NTFS file systems through intelligent correlation of $LogFile and $UsnJrnl artifacts.

---

## 🏆 Key Accomplishments

This thesis aims to develop a **complete end-to-end machine learning pipeline** for detecting timestamp manipulation in NTFS filesystems:

### **✅ Methodology Achievements**

1. **Research-Backed Data Processing**:
   - Implemented smart union merging of $LogFile and $UsnJrnl based on Oh et al. (2024) methodology
   - 95.4% data reduction (3.2M → 154K records) while preserving 100% of timestomped events
   - File system tunneling identification to reduce false positives

2. **Comprehensive Feature Engineering**:
   - Created 25 features across 5 categories (location, file type, temporal, cross-artifact, pattern)
   - Rigorous feature selection: 16 final features representing 97.92% of predictive power
   - Parsed unstructured `lf_detail` text field to extract structured timestamp manipulation data

3. **Robust Model Training & Optimization**:
   - Evaluated 5 ML algorithms (XGBoost, Random Forest, Logistic Regression, LightGBM, Neural Network)
   - XGBoost achieved **98.57% recall, 84.15% precision, F1=0.9079** on training data
   - Hyperparameter optimization and threshold optimization (19 thresholds tested)
   - 4-tier risk classification system (CRITICAL/HIGH/MEDIUM/LOW)

4. **Production-Ready Detection Tools**:
   - `detect_processed.py`: Fast validation tool for processed data
   - `detect_complete.py`: Complete pipeline for raw LogFile + UsnJrnl CSV files
   - Comprehensive output: predictions.csv, flagged_files.csv, summary_report.txt

5. **Rigorous External Validation**:
   - Tested on Case 11-APT (external dataset, not in training)
   - **Identified critical overfitting issue** - model does not generalize to external data
   - Documented limitations with root cause analysis and proposed solutions

### **✅ Research Contributions**

- **Complete ML Pipeline**: Phases 1-5 fully implemented and documented
- **Feature Engineering**: Novel forensic features for timestamp manipulation detection
- **Critical Finding**: Demonstrated importance of diverse training data in forensic ML
- **Honest Evaluation**: External validation revealed overfitting - valuable research insight
- **Reproducible Methodology**: All code, notebooks, and documentation available

### **✅ Model Performance Analysis (v1.0)**

**Corrected Analysis Results** (See: `analyze_all_pe_detections_CORRECTED.py`):

The initial evaluation metrics were misleading due to ground truth quality issues. A corrected comprehensive analysis across all 12 PE cases reveals:

**TRUE Malicious Timestomping Detection** (LogFile evidence OR [Malicious] label):
- **19 total events** across 12 cases (NOT 252 as initially thought)
- **73.7% HIGH confidence detection rate** (14/19 detected at ≥70% probability)
- **84.2% overall recall** (16/19 detected at ≥30% probability)
- **3 missed events** (15.8% false negatives)

**File System Tunneling** (UsnJrnl only, [Suspicious] label):
- **233 total events** (mostly WindowsUpdate.etl files)
- **54.9% correctly given LOW confidence** (<30% probability)
- **39.9% incorrectly flagged as HIGH confidence** (93/233 events)

**Key Insight**: The ground truth labels include 233 WindowsUpdate.etl files marked as "Timestamp Manipulation" that are actually file system tunneling (Windows OS behavior, not attacks). The model's 98.57% recall metric was inflated because it treated both categories equally.

**What This Means**:
- ✅ **Model detects TRUE malicious timestomping at 73.7% HIGH confidence rate**
- ⚠️ **Model over-detects file system tunneling** (39.9% false HIGH confidence on benign OS behavior)
- ⚠️ **Ground truth quality issue**: Labels don't distinguish malicious attacks from suspicious OS behavior
- 📊 **Realistic performance**: 84.2% recall on actual attacks (not the misleading 98.57%)

### **⚠️ Known Limitations**

- **Tunneling Over-Detection**: Model flags 40% of file system tunneling events as HIGH confidence (should be LOW)
- **Location-based features**: Model relies heavily on `in_temp_dir` (30% importance), causing both overfitting AND tunneling false positives
- **Ground truth limitations**: Training data includes 233 suspicious (but non-malicious) events that inflate metrics
- **External dataset generalization**: Needs validation on diverse APT datasets
- **NOT production-ready**: Requires retraining with cleaner ground truth and diverse datasets

**See**:
- [CRITICAL_FINDINGS.md](CRITICAL_FINDINGS.md) for full overfitting analysis
- [detection_analysis_CORRECTED.csv](detection_analysis_CORRECTED.csv) for detailed per-case results
- Run `python analyze_all_pe_detections_CORRECTED.py` to reproduce corrected analysis

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

### ✅ Phase 2: Feature Engineering - **COMPLETED**

**Phase 2A: File-Level & Behavioral Features**
- ✅ Created 18 features (location, file type, temporal behavioral)
- ✅ 100% coverage features (work on all timestomped events)
- ✅ Key features: `in_windows_dir`, `filename_length`, temporal clustering patterns
- ✅ **Output**: 56 columns (38 + 18 new)

**Phase 2B: Cross-Artifact & Pattern Features**
- ✅ Created 7 features (cross-artifact correlation, UsnJrnl patterns, event-time comparison)
- ✅ Research-backed pattern detection: BASIC_INFO_CHANGE + CLOSE signature
- ✅ Cross-artifact confidence scoring (both artifacts = HIGH confidence)
- ✅ **Output**: 63 columns (38 + 18 + 7 new)

**Phase 2C: Feature Quality Analysis & Selection**
- ✅ Fixed path_depth bug (regex issue from Phase 2A)
- ✅ Analyzed all 25 new features (distribution, correlation, importance)
- ✅ Trained preliminary Random Forest for feature importance ranking
- ✅ Selected top 16 features (97.92% cumulative importance)
- ✅ Removed 9 low-importance/redundant features (<0.5% importance or r>0.95 correlation)
- ✅ **Output**: Final ML-ready dataset with 16 curated features

**Feature Selection Results:**
- **Top 3 Features**: `in_temp_dir` (30.02%), `event_frequency_per_file` (20.53%), `events_in_5min_window` (11.30%)
- **Feature Groups**: 4 location, 3 file type, 6 temporal, 1 cross-artifact, 1 pattern, 0 event-time
- **Total Features**: 16 (from original 25) representing 97.92% of predictive power

**Key Achievements**:
- Discovered `in_temp_dir` as dominant signal (30% importance)
- All 6 temporal features retained (100% - critical for detection)
- Successfully fixed path_depth bug (now 5.44% importance, ranked #6)
- Identified and removed 3 highly correlated feature pairs
- Zero data loss (252 timestomped events preserved throughout all phases)

### ✅ Phase 3: Baseline Model Training - **COMPLETED**

**Models Trained**: 5 algorithms evaluated on 16-feature dataset
- ✅ Random Forest: 98% recall, 57% precision (37 FP, 1 FN)
- ✅ Logistic Regression: 98% recall, 14% precision (310 FP, 1 FN)
- ✅ **XGBoost: 98% recall, 83% precision (10 FP, 1 FN) - BEST MODEL** ⭐
- ✅ LightGBM: 92% recall, 5% precision (846 FP, 4 FN)
- ❌ Neural Network (Focal Loss): 0% recall (complete failure)

**Key Achievements**:
- **XGBoost selected as primary model** (F1=0.8991, best balance)
- Only 1 false negative (missed 1/50 timestomped files) for top 3 models
- XGBoost achieved 83% precision with only 10 false positives
- ROC-AUC of 0.9998 (near-perfect discrimination) for XGBoost and Random Forest
- Successfully handled 1:612 class imbalance

**Dataset Split**:
- Train: 123,640 samples (202 timestomped, 123,438 benign)
- Test: 30,910 samples (50 timestomped, 30,860 benign)
- Stratified 80/20 split preserving class distribution

**Output Files**:
- 5 trained models saved (pkl/h5 format)
- Model comparison metrics and visualizations
- Comprehensive performance report

### ✅ Phase 4: Hyperparameter Optimization - **COMPLETED**

**Optimization Method**: RandomizedSearchCV (50 iterations × 5-fold CV per model)
- ✅ XGBoost: CV F1=0.8722, Test F1=0.8972 (slight decrease from baseline)
- ✅ Random Forest: CV F1=0.8508, **Test F1=0.8522 (huge +13.16pp improvement!)**

**XGBoost Optimization Results**:
- Baseline: 98% recall, 83.05% precision, F1=0.8991 (10 FP, 1 FN)
- Optimized: 96% recall, 84.21% precision, F1=0.8972 (9 FP, 2 FN)
- **Verdict**: Baseline is better (more FN is unacceptable for forensics)

**Random Forest Optimization Results**:
- Baseline: 98% recall, 56.98% precision, F1=0.7206 (37 FP, 1 FN)
- Optimized: 98% recall, **75.38% precision**, **F1=0.8522** (16 FP, 1 FN)
- **Verdict**: Massive improvement! Reduced FP by 57% while maintaining recall

**Key Achievements**:
- Proved baseline XGBoost was already near-optimal (Phase 3 config validated)
- Dramatically improved Random Forest (now viable backup model)
- Optimized RF can detect files XGBoost misses (100% recall at threshold 0.25)
- **Final selection: Baseline XGBoost (Phase 3)** remains production model

**Output Files**:
- Optimized models saved (xgboost_optimized.pkl, random_forest_optimized.pkl)
- Best hyperparameters (JSON)
- Performance comparison visualizations

### ✅ Phase 5: Threshold Optimization & Risk Tiers - **COMPLETED**

**Thresholds Tested**: 19 thresholds (0.05 to 0.95) on both XGBoost and Random Forest

**Critical Finding**: XGBoost cannot achieve 100% recall at any threshold
- **Maximum recall**: 98% (thresholds 0.15-0.60 all identical performance)
- **1 timestomped file** is a statistical outlier XGBoost cannot detect
- **Random Forest CAN achieve 100% recall** at threshold 0.25 (71.43% precision, 20 FP)

**Best Threshold for XGBoost**:
- **Optimal range**: 0.15-0.60 (all give identical results - model is well-calibrated!)
- **Selected**: 0.5 (default) - Precision 83.05%, Recall 98%, F1=0.8991
- **Performance**: 59 files flagged (49 timestomped, 10 false positives)

**Three-Tier Risk System**:
- **HIGH** (prob ≥ 0.7): 59 files (49 timestomped = 98% of all detections)
- **MEDIUM** (prob 0.5-0.7): 0 files
- **LOW** (prob 0.3-0.5): 0 files
- **NONE** (prob <0.3): 30,851 files (1 timestomped missed)

**Key Insight**: Model gives very confident predictions (HIGH or NONE, no "maybes")
- This is **excellent** - investigators get clear priorities
- **Binary classification**: Flag as HIGH RISK or mark as CLEAN

**Key Achievements**:
- Proved XGBoost threshold 0.5 is optimal (no benefit from changing)
- Identified that 1 file is a true outlier (fundamentally undetectable by XGBoost)
- Designed simple, effective risk tier system for Autopsy module
- **Final production config**: XGBoost at threshold 0.5, binary HIGH/NONE classification

**Output Files**:
- Threshold performance curves (precision, recall, F1 vs threshold)
- Final production configuration (JSON)
- Comprehensive threshold analysis report

### 🎉 **MODEL TRAINING COMPLETE**

**Final Production Model**:
- **Algorithm**: Baseline XGBoost (Phase 3)
- **Features**: 16 selected features (97.92% cumulative importance)
- **Threshold**: 0.5 (default)
- **Performance on Training Data (Cases 01-PE to 12-PE)**:
  - **Initial Metrics** (treating all labels equally): 98.57% recall, 84.15% precision, F1=0.9079
  - **CORRECTED Metrics** (TRUE malicious events only): 73.7% HIGH confidence detection, 84.2% overall recall
  - **Ground Truth Breakdown**: 19 TRUE malicious events, 233 file system tunneling events
  - **Key Issue**: Model over-detects file system tunneling (39.9% false HIGH confidence)
- **Risk Classification**: 4-tier system (CRITICAL ≥0.7, HIGH ≥0.5, MEDIUM ≥0.3, LOW <0.3)

**Important**: See "Model Performance Analysis (v1.0)" section above for detailed corrected analysis

### ⚠️ **CRITICAL LIMITATIONS - MUST READ**

**External Validation Results (Case 11-APT)**:
- ❌ **Model FAILED on external dataset** (0% recall, 0/3 timestomped files detected)
- ❌ **Model is severely overfitted to Cases 01-PE to 12-PE training data**
- ❌ **Does NOT generalize to real-world forensic cases**

**Root Cause**: Model learned location-based patterns (e.g., "files in `\Windows\Temp\` are suspicious") instead of general timestamp manipulation patterns. Case 11-APT timestomped files are in `\Windows\SysWOW64\`, causing model to miss them entirely.

**What This Means**:
- ✅ **Model works on Cases 01-PE to 12-PE** (same dataset distribution)
- ❌ **Model does NOT work on external datasets** (different file locations, patterns, tools)
- ⚠️ **NOT production-ready** - requires retraining with diverse data


### 📋 Next Steps:
- **Option 1**: Retrain with diverse external datasets (Cases 11-APT + additional APT cases)
- **Option 2**: Re-engineer features to be location-agnostic (remove `in_temp_dir`, add timestamp anomaly features)
- **Option 3**: Document as limitation in thesis (academically rigorous approach)
- Phase 6: Autopsy module integration (ONLY after addressing overfitting)

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
| **Phase 2A: File-Level Features** | ✅ Complete | Extract location, file type, and temporal behavioral features | 18 new features (100% coverage) |
| **Phase 2B: Cross-Artifact Features** | ✅ Complete | Create cross-artifact correlation and UsnJrnl pattern features | 7 new features (research-backed patterns) |
| **Phase 2C: Feature Selection** | ✅ Complete | Analyze, select, and optimize feature set | 16 curated features (97.92% importance) |
| **Phase 3: Baseline Model Training** | ✅ Complete | Train and evaluate 5 ML algorithms | **XGBoost winner** (98% recall, 83% precision, F1=0.90) |
| **Phase 4: Hyperparameter Optimization** | ✅ Complete | Tune XGBoost and Random Forest | Baseline XGBoost validated as optimal, RF improved +13.16pp F1 |
| **Phase 5: Threshold Optimization** | ✅ Complete | Test 19 thresholds, design risk tier system | Threshold 0.5 optimal, binary HIGH/CLEAN classification |
| **Phase 6: Autopsy Integration** | 📋 Optional | Develop Ingest Module with best model | Deployable Autopsy plugin (LogFile + UsnJrnl only) |

---

## 🎯 Final Feature Set (16 Features)

After rigorous analysis and selection from 25 engineered features, the final ML-ready dataset contains:

### Top 5 Features (73.7% of predictive power):
1. **in_temp_dir** (30.02%) - Files in temporary directories
2. **event_frequency_per_file** (20.53%) - Number of modification events per file
3. **events_in_5min_window** (11.30%) - Event clustering in 5-minute window
4. **events_in_1min_window** (6.20%) - Event clustering in 1-minute window
5. **filename_length** (5.65%) - Length of filename (longer = suspicious)

### Complete Feature List by Category:

**Location Features (4):**
- `in_temp_dir`, `path_depth`, `in_program_files`, `in_windows_dir`

**File Type Features (3):**
- `filename_length`, `is_archive`, `is_executable`

**Temporal Behavioral Features (6):**
- `event_frequency_per_file`, `events_in_5min_window`, `events_in_1min_window`
- `event_frequency_per_case`, `time_until_next_event_seconds`, `time_since_previous_event_seconds`

**Cross-Artifact Features (1):**
- `has_logfile_evidence`

**Pattern Features (1):**
- `usn_complete_manipulation_pattern`

**Additional Location Features (1):**
- `in_users_dir`

**Key Insights:**
- Temporal features are critical (6/6 retained, 100% retention rate)
- `in_temp_dir` alone provides 30% of detection power
- All features have ≥0.5% importance, with top 10 representing 87.68% of total importance
- No highly correlated redundant features (all r<0.95 for kept features)

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
│       ├── Phase 1B - Column Cleanup/       # ✅ Complete
│       │   └── all_cases_combined_clean.csv # 154,550 records, 38 columns
│       ├── Phase 2A - File Level and Behavioral Features/  # ✅ Complete
│       │   └── all_cases_combined_with_phase2a_features.csv  # 56 columns
│       ├── Phase 2B - Cross Artifact and Pattern Features/  # ✅ Complete
│       │   └── all_cases_combined_with_phase2b_features.csv  # 63 columns
│       ├── Phase 2C - Feature Quality/      # ✅ Complete
│       │   ├── all_cases_combined_final_features.csv  # 26 columns (16 features)
│       │   ├── feature_importance_rankings.csv
│       │   ├── FEATURE_QUALITY_REPORT.md
│       │   └── visualizations/              # Distribution, correlation, importance plots
│       ├── Phase 3 - Model Training/        # ✅ Complete
│       │   ├── model_comparison.csv         # Performance metrics for 5 models
│       │   ├── PHASE_3_MODEL_TRAINING_REPORT.md
│       │   ├── roc_curves_comparison.png
│       │   ├── precision_recall_curves_comparison.png
│       │   ├── metrics_comparison_bars.png
│       │   └── confusion_matrices_all_models.png
│       ├── Phase 4 - Hyperparameter Optimization/  # ✅ Complete
│       │   ├── hyperparameter_optimization_results.csv
│       │   ├── best_hyperparameters.json
│       │   ├── baseline_vs_optimized_comparison.png
│       │   ├── false_positives_negatives_comparison.png
│       │   └── PHASE_4_HYPERPARAMETER_OPTIMIZATION_REPORT.md
│       └── Phase 5 - Threshold Optimization/   # ✅ Complete
│           ├── xgboost_threshold_analysis.csv
│           ├── random_forest_threshold_analysis.csv
│           ├── final_production_config.json  # ⭐ Production configuration
│           ├── xgboost_threshold_performance.png
│           ├── xgboost_fp_fn_vs_threshold.png
│           └── PHASE_5_THRESHOLD_OPTIMIZATION_REPORT.md
│
├── models/                                  # ✅ Trained Models
│   ├── random_forest_model.pkl
│   ├── random_forest_optimized.pkl
│   ├── logistic_regression_model.pkl
│   ├── xgboost_model.pkl                    # ⭐ PRODUCTION MODEL
│   ├── xgboost_optimized.pkl
│   ├── lightgbm_model.pkl
│   ├── neural_network_model.h5
│   └── feature_scaler.pkl
│
├── notebooks/
│   ├── Phase 1 - Data Cleaning/             # ✅ Complete
│   │   ├── 01_Smart_Union_Merging.ipynb
│   │   ├── 01B_Column_Analysis_and_Cleanup.ipynb
│   │   └── Forensic_Detection_of_Timestamp_Manipulation_for_D.pdf
│   ├── Phase 2 - Feature Engineering/       # ✅ Complete
│   │   ├── 02A_File_Level_and_Behavioral_Features.ipynb
│   │   ├── 02B_Cross_Artifact_and_Pattern_Features.ipynb
│   │   ├── 02C_Feature_Quality_Analysis.ipynb
│   │   ├── PHASE_2_PLAN_REVISED.md
│   │   ├── PHASE_2A_RESULTS_ANALYSIS.md
│   │   ├── PHASE_2B_RESULTS_ANALYSIS.md
│   │   └── PHASE_2_STATE_SUMMARY.md
│   ├── Phase 3 - Model Training/            # ✅ Complete
│   │   └── 03_Baseline_Model_Training.ipynb
│   ├── Phase 4 - Hyperparameter Optimization/  # ✅ Complete
│   │   └── 04_Hyperparameter_Optimization.ipynb
│   └── Phase 5 - Threshold Optimization/    # ✅ Complete
│       └── 05_Threshold_Optimization.ipynb
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

---

## 📋 Project Status Summary

### **Current Version: v1.0 - Training Data Validation Complete** ✅

This version represents the **completed Phases 1-5** of the Digital Detectives thesis project:

**✅ What Works**:
- Complete end-to-end ML pipeline (data cleaning → feature engineering → model training → optimization)
- XGBoost model achieves **98.57% recall, 84.15% precision** on Cases 01-PE to 12-PE
- Production-ready detection tools (`detect_processed.py`, `detect_complete.py`)
- Comprehensive documentation and reproducible methodology

**⚠️ Known Issues**:
- **Model overfitting**: Fails on external datasets (0% recall on Case 11-APT)
- **Feature engineering**: Location-based features cause overfitting
- **NOT production-ready**: Requires retraining before real-world deployment

**📝 Recommended Use Cases for This Version**:
- ✅ **Thesis documentation**: Demonstrate complete ML pipeline development
- ✅ **Methodology validation**: Show rigorous evaluation (including external validation failure)
- ✅ **Research contribution**: Document importance of diverse training data in forensic ML
- ✅ **Academic rigor**: Honest evaluation of model limitations

**🔄 Future Work (Next Version)**:
- **Option 1**: Retrain with diverse datasets (Case 11-APT + additional APT cases)
- **Option 2**: Re-engineer features to be location-agnostic (remove `in_temp_dir`, add robust timestamp anomaly features)
- **Option 3**: Combine both approaches for production-ready system

### **Recommended Next Steps for v2.0**:

Based on the corrected analysis, v2.0 should address:

1. **Ground Truth Re-Labeling**:
   - Separate TRUE malicious events from file system tunneling in ground truth
   - Create stratified evaluation metrics (malicious vs tunneling)
   - Add label category field: "malicious_timestomping" vs "filesystem_tunneling"

2. **Feature Engineering Improvements**:
   - Reduce reliance on `in_temp_dir` (30% importance → location overfitting)
   - Add timestamp anomaly features (e.g., zero nanoseconds, impossible sequences)
   - Incorporate cross-artifact validation features (Time Reversal + BASIC_INFO_CHANGE)
   - Add file system tunneling detection features (15-second window, same filename patterns)

3. **Training Data Diversification**:
   - Add Case 11-APT to training set (APT attack patterns)
   - Collect additional APT datasets with different timestomping tools
   - Include diverse file locations (not just `\Windows\Temp\`)

4. **Model Retraining Strategy**:
   - Train on cleaner ground truth (19 malicious + diverse APT data)
   - Use file system tunneling events as negative class (not positive class)
   - Implement stratified evaluation: HIGH confidence on malicious, LOW confidence on tunneling
   - Target: >90% HIGH confidence on TRUE malicious, <10% HIGH confidence on tunneling

### **Files to Review Before Branching**:
- [detection_analysis_CORRECTED.csv](detection_analysis_CORRECTED.csv) - Per-case corrected performance metrics
- [analyze_all_pe_detections_CORRECTED.py](analyze_all_pe_detections_CORRECTED.py) - Corrected analysis script
- [CRITICAL_FINDINGS.md](CRITICAL_FINDINGS.md) - Comprehensive overfitting analysis
- [EXTERNAL_VALIDATION_REPORT.md](EXTERNAL_VALIDATION_REPORT.md) - Case 11-APT test results
- [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md) - Case 12 performance (baseline)
- [COMPLETE_PIPELINE_GUIDE.md](COMPLETE_PIPELINE_GUIDE.md) - Detection tool usage guide

### **Citation**

If using this work, please cite:

```
Digital Detectives: Machine Learning-Based Timestamp Manipulation Detection for NTFS Filesystems
Thesis Project, 2024-2025
Based on methodology by Oh, Lee, and Hwang (2024) - "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"
IEEE Access, DOI: 10.1109/ACCESS.2024.10517044
```

---

## 🔄 v2.0 Retraining Strategy - Addressing Location-Based Overfitting

### **Overview**

Based on comprehensive external validation across 14 APT datasets, v1.0 exhibits severe location-based overfitting, achieving **0% detection on external data**. The v2.0 retraining strategy addresses this critical issue through:

1. **Training Data Diversification**: Add 6 APT datasets (30+ timestomped events) to training set
2. **Location-Agnostic Feature Engineering**: Remove location features, add cross-artifact validation and manipulation pattern features
3. **Stratified Evaluation**: Separate metrics for TRUE malicious vs file system tunneling
4. **External Testing**: Validate on 8 held-out APT datasets to prove generalization

### **Root Cause Analysis**

**v1.0 Overfitting Issue**:
- Model learned "files in `\Windows\Temp\` are suspicious" instead of forensic patterns
- `in_temp_dir` feature has 30% importance (location-based shortcut)
- PE training data: 40-50% of timestomped files in temp_dir
- APT external data: Only 3.1% in temp_dir
- **Result**: 0% detection on all 14 external APT datasets

**Evidence**:
- Detailed analysis: [EXTERNAL_DATASET_ANALYSIS_SUMMARY.md](EXTERNAL_DATASET_ANALYSIS_SUMMARY.md)
- Per-dataset results: `external_dataset_analysis.csv`
- Dataset organization: [V2_DATASET_ORGANIZATION.md](V2_DATASET_ORGANIZATION.md)

### **v2.0 Success Criteria**

| Metric | v1.0 (Baseline) | v2.0 (Target) |
|--------|-----------------|---------------|
| **PE Test Set (TRUE Malicious)** | 73.7% HIGH | **>90% HIGH** |
| **PE Test Set (Tunneling)** | 39.9% HIGH (wrong!) | **<10% HIGH** |
| **APT Training Set (External)** | 0% | **>70% HIGH** |
| **APT Testing Set (Held-Out)** | 0% | **>70% HIGH** ⭐ |
| **Location Feature Importance** | 41.5% total | **<10% total** |

**Critical Metric**: APT Testing Set detection proves model generalizes to unseen attack groups and patterns.

---

### **v2.0 Training Pipeline**

#### **Phase 0: v2.0 Preparation & Planning**

**Objective**: Document strategy changes and prepare combined training data

**Key Tasks**:
1. **Document v2.0 Strategy**:
   - Comprehensive analysis of v1.0 overfitting patterns
   - Feature engineering changes (location → forensic patterns)
   - Training data composition (PE + 6 APT datasets)
   - Success criteria and evaluation strategy

2. **Prepare Combined Training Data**:
   - PE Cases (01-12): Keep all 252 rows (19 TRUE malicious + 233 tunneling)
   - APT Training (6 cases): ~30 timestomped events
   - **Total v2.0 Training**: ~273 timestomped events
   - **Class Balance**: 1:565 (manageable, down from 1:612 in v1.0)

3. **Optional: Ground Truth Enhancement**:
   - Add `label_category` field: "malicious_timestomping" vs "filesystem_tunneling"
   - Enable stratified evaluation (HIGH confidence on malicious, LOW on tunneling)
   - Document labeling methodology

**Deliverables**:
- `V2_DATASET_ORGANIZATION.md` - Dataset split strategy
- `EXTERNAL_DATASET_ANALYSIS_SUMMARY.md` - Overfitting analysis
- `count_unique_timestomped_files.py` - File counting methodology

---

#### **Phase 1: Data Cleaning & Smart Union Merging (v2.0)**

**Objective**: Process 6 APT training datasets and combine with existing PE data

**Key Tasks**:
1. **Process APT Training Datasets**:
   - 01-APT17, 02-APT19, 04-APT28, 05-APT29 (2 events each)
   - 10-DarkHotel663, 11-DarkHotelbbd (4-6 events each)
   - Apply same smart union merging as v1.0 Phase 1A
   - Parse `lf_detail` text field (same as v1.0 Phase 1B)

2. **Combine with Existing PE Data**:
   - Merge APT Phase 1 output with existing PE Phase 1 data
   - Validate data structure consistency (column alignment)
   - Verify ground truth labels are preserved (100% retention)

3. **Data Quality Validation**:
   - Check for missing values, data type consistency
   - Verify timestomped event counts match expectations
   - Confirm cross-artifact patterns are preserved

**Input**:
- APT datasets: `data/added datasets/training/logfile/`, `data/added datasets/training/usnjrnl/`
- APT ground truth: `data/added datasets/training/suspicious/`
- Existing PE Phase 1: `data/processed/Phase 1B - Column Cleanup/all_cases_combined_clean.csv`

**Output**:
- `data/processed/Phase 1 - V2 Data Cleaning/all_cases_combined_v2.csv`
- **~157,000 records** (154,550 PE + ~2,500 APT)
- **~273 timestomped events** (252 PE + ~21 APT)
- **38 columns** (same structure as v1.0)

---

#### **Phase 2A: Location-Agnostic Feature Engineering**

**Objective**: Remove location features and add forensically robust features

**Features to REMOVE** (Location-Based Overfitting):
- ❌ `in_temp_dir` (30% importance in v1.0 - PRIMARY overfitting cause)
- ❌ `in_program_files` (11.5% importance - contributes to overfitting)
- ❌ `in_users_dir` (minimal importance but location-based)
- **Target**: Location features should be <10% total importance in v2.0

**Features to ADD** (Forensically Robust):

1. **Cross-Artifact Validation Score** (`cross_artifact_validation_score`):
   - Based on Oh et al. (2024) Algorithm 6
   - Scoring system:
     - **3 points**: Time Reversal Event (LogFile) + BASIC_INFO_CHANGE (UsnJrnl) on same file
     - **2 points**: Time Reversal Event only (LogFile)
     - **1 point**: BASIC_INFO_CHANGE + CLOSE pattern only (UsnJrnl)
     - **0 points**: No clear evidence
   - **Rationale**: Cross-artifact agreement = highest confidence (location-agnostic)

2. **Timestamp Manipulation Pattern Score** (`timestamp_manipulation_pattern_score`):
   - Detect manipulation patterns from Oh et al. (2024):
     - Copied timestamps (multiple files with identical timestamps)
     - $FN manipulation patterns (BASIC_INFO_CHANGE on $FN attribute)
     - Rapid sequential manipulation (multiple files timestomped in <1 minute)
   - Scoring: 0-3 (number of patterns detected)
   - **Rationale**: Attackers often timestomp multiple files with similar patterns

3. **File System Tunneling Detection** (`file_system_tunneling_detected`):
   - Based on Oh et al. (2024) Algorithms 4, 7
   - Detection criteria:
     - UsnJrnl BASIC_INFO_CHANGE event only (no LogFile Time Reversal)
     - Same filename deleted and recreated within 15-second window
     - Windows OS behavior, NOT attack
   - Binary: 1 (tunneling detected) or 0 (not tunneling)
   - **Rationale**: Reduces false positives on benign OS behavior (addresses 39.9% tunneling over-detection)

**Features to KEEP** (From v1.0):
- ✅ All 6 temporal features (critical for detection)
- ✅ File type features (filename_length, is_archive, is_executable)
- ✅ Cross-artifact features (has_logfile_evidence)
- ✅ Pattern features (usn_complete_manipulation_pattern)
- ✅ `path_depth` (general file structure, not location-specific)
- ✅ `in_windows_dir` (keep but reduce importance)

**Implementation Notes**:
- Use exact algorithms from Oh et al. (2024) paper
- Only implement features feasible with LogFile + UsnJrnl CSVs (no $MFT required)
- Zero nanoseconds feature NOT implemented (requires raw $MFT with 100-nanosecond precision)

**Output**:
- `data/processed/Phase 2A - V2 Location Agnostic Features/all_cases_combined_v2_phase2a.csv`
- **~40-45 columns** (38 base + 3 new - 3 removed = ~38 columns)
- All timestomped events preserved

---

#### **Phase 2B: Feature Selection & Quality (v2.0)**

**Objective**: Analyze v2.0 feature distributions and select optimal feature set

**Key Tasks**:
1. **Feature Distribution Analysis**:
   - Compare PE vs APT feature distributions
   - Validate location features have reduced importance
   - Verify cross_artifact_validation_score has high importance

2. **Correlation Analysis**:
   - Identify highly correlated features (r > 0.95)
   - Remove redundant features
   - Ensure feature diversity

3. **Preliminary Feature Importance**:
   - Train Random Forest on v2.0 data
   - Rank features by importance
   - Target: 15-20 features with >95% cumulative importance

4. **Quality Validation**:
   - Ensure location features <10% total importance
   - Verify cross-artifact features are top 5
   - Confirm temporal features are retained

**Output**:
- `data/processed/Phase 2B - V2 Feature Quality/all_cases_combined_v2_final_features.csv`
- **15-20 curated features** (final ML-ready dataset)
- Feature importance rankings and visualizations
- `PHASE_2B_V2_FEATURE_QUALITY_REPORT.md`

**Expected Top Features** (v2.0):
1. `cross_artifact_validation_score` (estimated 25-30% importance)
2. `event_frequency_per_file` (high importance retained from v1.0)
3. `events_in_5min_window` (temporal clustering)
4. `timestamp_manipulation_pattern_score` (new forensic feature)
5. `has_logfile_evidence` (cross-artifact indicator)

---

#### **Phase 3: Model Training (v2.0 - 5 Algorithms)**

**Objective**: Train 5 ML algorithms on v2.0 dataset for comparison

**Algorithms** (Same as v1.0 for Comparison):
1. **Logistic Regression** (baseline)
2. **Random Forest** (ensemble baseline)
3. **XGBoost Baseline** (primary model)
4. **XGBoost Tuned** (hyperparameter optimization)
5. **Neural Network** (deep learning approach)

**Training Configuration**:
- **Dataset Split**: 80/20 stratified split (same as v1.0)
- **Class Balance**: Handle 1:565 imbalance with class weights
- **Cross-Validation**: 5-fold stratified CV during training
- **Evaluation Metrics**: Precision, Recall, F1, ROC-AUC

**Key Differences from v1.0**:
- Training data includes 6 APT datasets (diverse attack patterns)
- Location features removed (prevents overfitting)
- Cross-artifact features prioritized (forensically robust)

**Output**:
- `models/v2_xgboost_model.pkl` (primary production model)
- `models/v2_random_forest_model.pkl`, `models/v2_logistic_regression_model.pkl`
- `models/v2_neural_network_model.h5`
- `data/processed/Phase 3 - V2 Model Training/model_comparison_v2.csv`
- `PHASE_3_V2_MODEL_TRAINING_REPORT.md`

**Expected Performance** (v2.0 Targets):
- **Precision**: >80% (similar to v1.0 XGBoost)
- **Recall**: >95% on training/test split
- **F1 Score**: >0.85
- **Feature Importance**: Location features <10% total

---

#### **Phase 4: Model Evaluation - Stratified (v2.0)**

**Objective**: Rigorous evaluation across PE, APT training, and APT testing sets

**Evaluation Strategy** (3-Tier):

**Tier 1: PE Test Set (Internal Validation)**
- **TRUE Malicious Events** (19 events with LogFile evidence):
  - Target: **>90% HIGH confidence detection** (≥70% probability)
  - v1.0 baseline: 73.7% HIGH confidence
  - **Critical**: Must improve over v1.0

- **File System Tunneling** (233 events, UsnJrnl only):
  - Target: **<10% HIGH confidence detection** (<70% probability)
  - v1.0 baseline: 39.9% HIGH confidence (wrong!)
  - **Critical**: Must differentiate tunneling from malicious

**Tier 2: APT Training Set (External Validation - Training Data)**
- **6 APT datasets** used in training (01-APT17, 02-APT19, 04-APT28, 05-APT29, 10-DarkHotel663, 11-DarkHotelbbd)
- **~30 timestomped events** total
- Target: **>70% HIGH confidence detection**
- v1.0 baseline: 0% (complete failure)
- **Purpose**: Validate model learned from APT data

**Tier 3: APT Testing Set (External Validation - HELD-OUT)** ⭐
- **8 held-out APT datasets** NOT in training (03-APT21, 06-APT30, 07-APT37, 08-APT38, 09-APT40, 12-Kimsuky, 13-Winnti731, 14-Winnti43b)
- **~12-14 timestomped events** total
- Target: **>70% HIGH confidence detection**
- v1.0 baseline: 0% (complete failure)
- **CRITICAL METRIC**: Proves model generalizes to unseen attack groups

**Analysis Tasks**:
1. **Per-Dataset Analysis**: Detection rate for each of 14 APT datasets
2. **Feature Importance Validation**: Confirm location features <10% total
3. **False Positive Analysis**: Identify common false positives (like OneDrive files in v1.0)
4. **Confidence Distribution**: Verify model gives HIGH/LOW predictions (not uncertain MEDIUM)

**Output**:
- `test/v2_evaluation/pe_test_set_stratified_results.csv`
- `test/v2_evaluation/apt_training_set_results.csv`
- `test/v2_evaluation/apt_testing_set_results.csv` (CRITICAL)
- `PHASE_4_V2_STRATIFIED_EVALUATION_REPORT.md`
- Comparison table: v1.0 vs v2.0 performance

---

#### **Phase 5: Terminal-Based Detection Tool (v2.0)**

**Objective**: Update detection tools with v2.0 model and features

**Tools to Update**:

1. **`detect_accurate.py`** (v2.0 Production Tool):
   - Load v2.0 XGBoost model and features
   - Implement v2.0 feature engineering (cross-artifact validation, pattern detection, tunneling detection)
   - Generate comprehensive reports with stratified confidence levels
   - Input: Raw LogFile + UsnJrnl CSVs
   - Output: predictions.csv, flagged_files.csv, summary_report.txt

2. **Feature Engineering Module**:
   - Implement `cross_artifact_validation_score` calculation
   - Implement `timestamp_manipulation_pattern_score` detection
   - Implement `file_system_tunneling_detected` algorithm
   - Ensure compatibility with production Autopsy module

3. **Confidence Categorization** (4-Tier System):
   - **CRITICAL** (prob ≥ 0.85): Time Reversal Event + cross-artifact validation
   - **HIGH** (prob ≥ 0.70): Strong forensic indicators
   - **MEDIUM** (prob ≥ 0.50): Suspicious patterns, investigate
   - **LOW** (prob < 0.50): Likely benign or file system tunneling

**Validation Tests**:
- Run on all 14 APT datasets (comprehensive external validation)
- Compare v1.0 vs v2.0 detection results side-by-side
- Verify >70% HIGH confidence on held-out APT test set

**Output**:
- `detect_accurate.py` (updated for v2.0)
- `test/v2_detection_tool/all_apt_datasets_results/` (14 folders with predictions)
- `V2_DETECTION_TOOL_VALIDATION_REPORT.md`

---

#### **Phase 6: Comprehensive Evaluation & Documentation**

**Objective**: Document v2.0 improvements and prepare for Autopsy integration

**Key Deliverables**:

1. **v1.0 vs v2.0 Comparison Report**:
   - Side-by-side performance metrics table
   - Feature importance comparison (location features before/after)
   - External validation results (0% → >70% detection rate)
   - Root cause analysis and solution summary

2. **Complete v2.0 Documentation**:
   - Update README.md with v2.0 results
   - Document v2.0 training pipeline (this section)
   - Feature engineering methodology (Oh et al. algorithms implemented)
   - Model selection rationale and performance

3. **Production Readiness Assessment**:
   - ✅ If APT test set >70% HIGH: Model is production-ready
   - ⚠️ If APT test set 50-70% HIGH: Model needs refinement
   - ❌ If APT test set <50% HIGH: Model requires additional training data

4. **Autopsy Integration Preparation** (Phase 7 - Future Work):
   - Finalize v2.0 feature engineering code
   - Package model and dependencies
   - Create Autopsy Ingest Module specification
   - User guide for LogFile + UsnJrnl input workflow

**Output Files**:
- `V2_FINAL_EVALUATION_REPORT.md` (comprehensive analysis)
- `V1_VS_V2_COMPARISON.md` (side-by-side metrics)
- `PRODUCTION_READINESS_ASSESSMENT.md`
- Updated README.md (this file)

---

### **v2.0 Training Data Composition**

**Total Training Set** (v2.0):

| Source | Cases | Timestomped Events | Key Characteristics |
|--------|-------|-------------------|---------------------|
| **PE Cases (01-12)** | 12 | 252 (19 TRUE malicious + 233 tunneling) | Tools: SetMACE, nTimestomp, PowerShell |
| **APT Training** | 6 | ~30 | Attack groups: APT17, APT19, APT28, APT29, DarkHotel |
| **TOTAL v2.0 Training** | 18 | **~273** | Diverse locations and attack patterns |

**Reserved for Testing** (Held-Out):

| Source | Cases | Timestomped Events | Purpose |
|--------|-------|-------------------|---------|
| **APT Testing** | 8 | ~12-14 | Prove generalization to Winnti, Kimsuky, APT21/30/37/38/40 |

**Class Balance**:
- v1.0: 252 timestomped / 154,298 benign = 1:612
- v2.0: ~273 timestomped / ~154,000 benign = 1:565 (slightly better)

---

### **Key Differences: v1.0 vs v2.0**

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| **Training Data** | PE Cases 01-12 only (252 events) | PE Cases + 6 APT datasets (~273 events) |
| **Location Features** | `in_temp_dir`, `in_program_files`, `in_users_dir` (41.5% total importance) | Removed (target <10% total importance) |
| **Top Feature** | `in_temp_dir` (30% importance) | `cross_artifact_validation_score` (est. 25-30% importance) |
| **Cross-Artifact Features** | 1 feature (`has_logfile_evidence`) | 3 features (validation score, pattern score, tunneling detection) |
| **PE TRUE Malicious Detection** | 73.7% HIGH confidence | Target: **>90% HIGH** |
| **PE Tunneling Over-Detection** | 39.9% HIGH (wrong!) | Target: **<10% HIGH** |
| **APT External Detection** | 0% (complete failure) | Target: **>70% HIGH** |
| **Location Overfitting** | Severe (learns temp_dir = suspicious) | Minimal (forensic patterns only) |
| **Production Readiness** | ❌ NOT ready (overfitted) | ✅ READY (if targets met) |

---

### **Expected v2.0 Outcomes**

**If v2.0 Achieves Targets**:
- ✅ **Model generalizes to external APT datasets** (>70% detection on held-out test set)
- ✅ **Reduces file system tunneling false positives** (<10% HIGH confidence on benign OS behavior)
- ✅ **Improves TRUE malicious detection** (>90% HIGH confidence on actual attacks)
- ✅ **Production-ready for Autopsy integration** (Phase 7)

**If v2.0 Falls Short**:
- Additional training data required (more APT datasets)
- Feature engineering refinement (add more forensic indicators)
- Consider ensemble approach (combine multiple models)

**Research Contribution** (Regardless of Outcome):
- Demonstrates importance of diverse training data in forensic ML
- Documents methodology for addressing location-based overfitting
- Provides reproducible pipeline for timestamp manipulation detection
- Shows rigorous evaluation including external validation

---

### **Files Added for v2.0**

**Analysis Scripts**:
- `count_unique_timestomped_files.py` - Count unique timestomped files (not rows)
- `analyze_external_datasets.py` - Comprehensive overfitting analysis

**Documentation**:
- `V2_DATASET_ORGANIZATION.md` - Dataset split strategy (6 training, 8 testing)
- `EXTERNAL_DATASET_ANALYSIS_SUMMARY.md` - Root cause analysis of v1.0 failure
- `UNIQUE_FILES_COUNT_CORRECTED.md` - Corrected file counts across all datasets

**Data Directories**:
- `data/added datasets/training/` - 6 APT training datasets
- `data/added datasets/testing/` - 8 held-out APT testing datasets

---

**This is a thesis research project. Model v1.0 is NOT production-ready without retraining on diverse data. v2.0 retraining is currently in progress to address identified limitations.**
