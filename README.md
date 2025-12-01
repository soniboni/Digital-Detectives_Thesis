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
**This is a thesis research project. Model is NOT production-ready without retraining on diverse data.**
