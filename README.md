**🕵️ Digital Detectives – Autopsy Plugin Prototype**

This repository is part of our thesis project, which aims to develop a prototype Autopsy plugin capable of detecting timestomped files through in-depth analysis of $LogFile and $UsnJrnl artifacts. The system integrates machine learning models to enhance detection accuracy and reduce false positives.

**🚧 Work in Progress – The plugin is currently under active development.**

---

## 📊 Project Progress & Status Update

| Phase | Task | Status | Output |
| :--- | :--- | :--- | :--- |
| **Phase 1** | A. Data Labelling | ✅ **Completed** | 24 labelled CSV files (12 LogFile + 12 UsnJrnl) |
| | B. Data Case Merging | ✅ **Completed** | 12 merged case files (01-PE through 12-PE) |
| | C. Master Timeline Creation | ✅ **Completed** | Single unified dataset (824,605 events) |
| **Phase 2** | Feature Engineering | ✅ **Completed** | ML-ready dataset (778,692 events, 87 features) |
| **Phase 3** | Model Training & Evaluation | ✅ **Completed** | Final Random Forest model (v3), evaluation metrics, visualizations |
| **Phase 4** | Demo & Deployment | ✅ **Completed** | Terminal-based inference scripts for production use |

***

## ✅ Latest Accomplishments

### Phase 2: Feature Engineering (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Transformed raw forensic timeline into **ML-ready feature vectors** with comprehensive temporal, behavioral, and anomaly detection features.
* Successfully handled extreme class imbalance (1:3,151 ratio) and missing data while preserving all 247 timestomped events.

**Key Achievements:**

1. **Intelligent Data Cleanup:**
   - Recovered **102,070 timestamps** from `lf_creation_time` (69% recovery rate).
   - Dropped **45,913 benign records** without valid timestamps.
   - Preserved **all 247 timestomped events** (100% label preservation).
   - Final dataset: **778,692 events** (5.6% reduction from master timeline).

2. **Comprehensive Feature Engineering (87 features total):**
   - **Temporal Features (12):** hour, day, month, year, weekend, off-hours, time deltas, event frequency metrics.
   - **Timestamp Anomaly Features (8):** Impossible sequences (creation after modification, accessed before creation), MAC timestamp patterns, future timestamps, year deltas.
   - **File Path Features (8):** Path depth, system/temp/user indicators, filename length, executable detection, entropy analysis.
   - **Event Pattern Features (6):** Label-encoded event types, rare event detection, consecutive event patterns.
   - **Cross-Artifact Features (46):** Merge type encoding, artifact presence flags, label source encoding, USN file attribute one-hot encoding (36 categories).
   - **Additional Flags (7):** Missing timestamp flag, high activity detection, timestamp tool execution indicators.

3. **Advanced Temporal Analysis:**
   - **Event Frequency Metrics:** Vectorized calculations for events-per-file and events-per-minute (activity rate).
   - **Time Delta Analysis:** Calculated time since previous event on same file with categorical binning (immediate/seconds/minutes/hours/days).
   - **Optimized Performance:** Replaced O(n²) rolling window approach with O(n) vectorized group aggregations for 825K records.

4. **Anomaly Detection Indicators:**
   - **24,703 cases** of creation after modification (impossible sequence).
   - **29 cases** of accessed before creation (impossible sequence).
   - **14,950 cases** of all MAC timestamps identical (suspicious pattern).
   - **102,576 cases** of future timestamps (MAC > eventtime).

5. **Data Quality Metrics:**
   - **All 247 timestomped events preserved** across all transformations.
   - **Class imbalance:** 1:3,151 (247 positive / 778,445 negative).
   - **No null values** in target variable.
   - **No object columns** remaining (all categorical data properly encoded).
   - **Clean data types:** 43 bool, 25 int64, 18 float64, 1 datetime64[ns].

**Technical Implementation:**
* **Smart Timestamp Recovery:** Used `lf_creation_time` as fallback for missing `eventtime`, recovering 69% of invalid timestamps.
* **Categorical Encoding Strategy:**
  - Label encoding for event types (preserves ordinality).
  - One-hot encoding for file attributes (35 unique categories → 36 binary features).
  - Feature flags for merge type and label source.
* **Performance Optimization:** Implemented vectorized pandas operations for event frequency calculations to avoid 10+ minute computation times.
* **Case ID Retention:** Preserved `case_id` for case-based stratified splitting in Phase 3 (prevent data leakage).

**Feature Categorization Summary:**
Labels: 1 → ['is_timestomped'] ID/Case: 1 → ['case_id'] Flags: 16 → Timestamp tool execution, rare events, activity patterns Temporal: 12 → Time components, deltas, frequency metrics Anomalies: 7 → Impossible sequences, suspicious patterns, year deltas Path: 8 → Depth, system/temp/user paths, entropy, executables Events: 6 → Encoded event types, rare events, consecutive patterns Cross-Artifact: 46 → Merge types, artifact presence, USN attributes (36 one-hot)

**Output:**
* ✅ Engineered dataset: `data/processed/Phase 2 - Feature Engineering/features_engineered.csv`
* ✅ Dataset size: **323.72 MB** (778,692 rows × 87 columns).
* ✅ Memory usage: **293.33 MB** (optimized data types).
* **Notebook:** `notebooks/Phase 2 - Feature Engineering/Feature Engineering.ipynb`

***

### Phase 1C: Master Timeline Creation (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Consolidated all 12 merged case files into a single unified master timeline.
* Validated data integrity and label preservation across the complete dataset.

**Key Achievements:**

1. **Unified Dataset Creation:**
   - Successfully merged **12 case files** into single timeline.
   - **824,605 total events** across all cases.
   - Chronologically sorted by `case_id` → `eventtime`.

2. **Data Integrity Validation:**
   - **Zero duplicates** found within each case (LSN/USN uniqueness verified).
   - **247 timestomped events** preserved (100% from Phase 1B).
   - All 12 cases maintain temporal integrity.

3. **Temporal Overlap Analysis:**
   - All 12 cases have **overlapping timeframes** (2022-12-16 to 2024-01-01).
   - `case_id` is critical for distinguishing events from different cases.
   - Enables **case-based stratified splitting** in Phase 3.

4. **Missing Timestamp Handling:**
   - **147,991 records** (17.9%) had invalid/missing `eventtime`.
   - Identified recovery strategy using `lf_creation_time` for Phase 2.
   - **8 timestomped events** flagged with missing timestamps (preserved for Phase 2).

**Technical Implementation:**
* **Vertical Concatenation:** Loaded and concatenated all 12 case files maintaining `case_id` integrity.
* **Duplicate Detection:** Checked LSN/USN uniqueness **within each case** (not globally, as LSN/USN are case-specific).
* **Temporal Analysis:** Analyzed eventtime ranges per case to identify overlapping timeframes.
* **Quality Assurance:** Verified 247 timestomped events preserved from Phase 1B output.

**Output:**
* ✅ Master timeline: `data/processed/Phase 1 - Data Collection & Preprocessing/C. Master Timeline/master_timeline.csv`
* ✅ Dataset size: **824,605 events** across 12 cases.
* ✅ Column structure: 34 columns (case_id, eventtime, filename, filepath + artifact columns with `lf_*` and `usn_*` prefixes).
* **Notebook:** `notebooks/Phase 1 - Data Collection & Preprocessing/C. Master Timeline Creation.ipynb`

***

### Phase 1B: Data Case Merging (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Successfully merged **LogFile** and **UsnJrnl** artifacts for all 12 forensic cases.
* Eliminated duplicate events while preserving cross-artifact correlation.
* Achieved **98.0% label preservation rate** (247/252 timestomped events).

**Key Achievements:**

1. **Massive Dataset Reduction:** 3,372,330 → 824,605 events (**75.5% reduction**).
2. **Zero Duplication:** Each LSN and USN appears exactly once.
3. **Preserved Events:** 247/252 timestomped events (98.0%):
   - 9 LogFile timestomped events (actual manipulation).
   - 238 UsnJrnl timestomped events.
   - **230 events detected by BOTH artifacts** (high confidence!).
4. **UsnJrnl Aggregation:** 3.1M → 631K events (removed 2.5M duplicates).

**Technical Implementation:**
* **Smart UsnJrnl Aggregation:** Grouped multiple events at the same timestamp+filepath+filename and combined event sequences (`File_Created` → `Data_Added` → `File_Closed`). Preserved detection flags using `max()` aggregation.
* **Intelligent Join Strategy:** Outer join on `eventtime` + `filepath` + `filename`. Prioritized "**Time Reversal Event**" over "**File Creation**" for duplicate LogFile entries. Preserved all LogFile-only and UsnJrnl-only events.
* **Label & Feature Preservation:** Separated `is_timestomped` (Primary label) from `timestomp_tool_executed` (Feature).

**Understanding the 5 "Missing" LogFile Events:**
* The 5 "missing" events are **NOT actual timestomping events**—they are tool execution indicators (e.g., Prefetch files like `NTIMESTOMP_V1.2_X64.EXE-6EA682C3.pf`).
* They have `is_timestomped=0` but `timestomp_tool_executed=1`. They indicate a tool was executed, but not the timestamp manipulation itself.
* They are **FEATURES, not LABELS** (a core Phase 1A design principle) and are fully preserved via the `timestomp_tool_executed=1` field.
* **Impact:** All 247 actual timestomped events were preserved with zero data loss!

**Output:**
* ✅ 12 merged datasets: `data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/XX-PE-Merged.csv`
* ✅ Column structure: `case_id`, `eventtime`, `filename`, `filepath` (leftmost), with artifact prefixes `lf_*` and `usn_*`.
* **Notebook:** `notebooks/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging.ipynb`

***

### Phase 1A: Data Labelling (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Successfully labelled all 12 forensic cases by matching suspicious behavior indicators from NTFS Log Tracker to forensic artifacts.
* **Critical Design Decision:** Separated tool execution (→ **FEATURE**) from actual timestamp manipulation (→ **LABEL**).

**Key Findings:**
* **Total Records Processed:** 3,372,330 events.
* **Actual Timestomped Events:** 252 unique events (14 LogFile + 238 UsnJrnl).
* **Tool Executions Detected:** 16 events (tracked as features).
* **Class Imbalance Ratio:** **1:13,382** (extreme imbalance).

**Understanding the 504 → 252 Reduction:**
The reduction from 504 suspicious indicators to 252 actual labels was due to:
* Duplicate Indicators (178 duplicates).
* Missing USN Values (~58 missing).
* Tool Execution Separated (16 events), correctly categorized as **features**.

**New Column Structure:**
* **Label (Prediction Target):** `is_timestomped` (Binary flag for actual manipulation).
* **Features (Help Predict):** `timestomp_tool_executed` (Binary flag for tool detection) and `suspicious_tool_name`.

**Why This Matters:**
Model learns timestamp patterns (**actual behavior**), ensuring better generalization to detect timestomping with unknown tools.

**Output:**
* ✅ 24 labelled datasets: `data/processed/Phase 1 - Data Collection & Preprocessing/A. Data Labelled/...`
* **Notebook:** `notebooks/Phase 1 - Data Collection & Preprocessing/A. Data Labelling.ipynb`

***

### Phase 3: Model Training & Evaluation (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Trained and optimized a **Random Forest classifier** using minimal SMOTE strategy to detect timestomped files.
* Conducted experimental testing of 3 imbalance-handling strategies to find optimal precision-recall balance.
* Evaluated model on held-out test set with comprehensive forensic metrics.

**Key Achievements:**

1. **Experimental Strategy Testing (Model v2):**
   - **Strategy A (No SMOTE):** 97.1% recall, 11.4% precision → Too many false positives
   - **Strategy B (Minimal SMOTE 1:1000):** 65.7% recall, 42.7% precision, **F1=0.517** ✓ **BEST**
   - **Strategy C (Conservative SMOTE 1:500):** 64.7% recall, 42.9% precision, F1=0.516
   - Selected Strategy B for final model due to superior balance

2. **Final Model Configuration (v3):**
   - **Algorithm:** Random Forest (50 trees, max_depth=4, regularized)
   - **SMOTE Strategy:** 1:1000 ratio (created only 338 synthetic samples vs. 2,415 in aggressive approach)
   - **Class Weights:** 5:1 (minority:majority) for moderate recall emphasis
   - **Regularization:** min_samples_leaf=150, max_leaf_nodes=25 to prevent overfitting

3. **Test Set Performance (295,539 samples, 102 timestomped):**
   - **Precision:** 42.7% (67 true positives / 157 flagged files)
   - **Recall:** 65.7% (67 detected / 102 actual timestomped files)
   - **F1-Score:** 0.517 (optimal balance for forensic triage)
   - **AUC-ROC:** 0.999 (near-perfect ranking ability)
   - **AUC-PR:** 0.348 (10× better than baseline 0.0345%)
   - **False Positive Rate:** 0.030% (only 90 benign files incorrectly flagged)
   - **False Negative Rate:** 34.3% (35 timestomped files missed)

4. **Forensic Triage Effectiveness:**
   - **Files requiring review:** 157 out of 295,539 (0.053%)
   - **Investigation reduction:** 99.95%
   - **Detection rate:** 65.7% (67 of 102 timestomped files surfaced)
   - **Practical value:** Investigators review 1/1,881 of original workload

5. **Feature Importance Analysis:**
   - **Top 5 predictive features:**
     1. `usn_event_encoded` (10.72%) - $UsnJrnl event types
     2. `lf_event_encoded` (9.99%) - $LogFile event types
     3. `is_temp_path` (9.59%) - Temporary directory indicator
     4. `events_per_minute` (8.80%) - Event frequency patterns
     5. `has_both_artifacts` (7.40%) - Cross-artifact correlation
   - **Category importance:** Cross-Artifact features contributed 29.05% (most important)
   - **Surprising finding:** Timestamp anomalies only 0.65% importance (event patterns more reliable)

6. **Model Stability:**
   - **OOB Score:** 0.999 (matches test AUC-ROC, confirming generalization)
   - **Case-based stratification:** Model tested on 3 completely unseen cases
   - **Training→Test gap:** Recall 87.3%→65.7%, Precision 65.5%→42.7% (expected generalization)

**Interpretation of AUC-ROC = 0.999:**
- Reflects **distinct forensic signatures** of timestomping tools (NTimeStomp, SetMACE) in journal files
- Validated by consistent OOB score and strong test performance
- Indicates **technical feasibility** rather than production-ready robustness
- Establishes **upper-bound benchmark** for journal-based ML detection
- Limitation: Tool-specific training (novel techniques may evade detection)

**Technical Implementation:**
* **Train/Test Split:** Case-based stratification (75/25) to prevent data leakage
  - Training: 483,153 samples (145 timestomped)
  - Test: 295,539 samples (102 timestomped)
* **SMOTE Application:** Applied only to training set (not test) with k_neighbors=3
* **Regularization:** Constrained tree depth and leaf sizes to prevent memorization
* **Evaluation:** Precision-Recall focus (more informative than accuracy for 1:3,151 imbalance)

**Output:**
* ✅ Trained models:
  - `data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib` (final model)
  - `data/processed/Phase 3 - Model Training/v2_experiments/` (strategy comparison)
* ✅ Evaluation metrics: `evaluation_metrics.csv`
* ✅ Feature importance: `feature_importance.csv` (75 features ranked)
* ✅ Test predictions: `test_predictions.csv` (295,539 predictions with probabilities)
* ✅ Visualizations:
  - Confusion matrix heatmap
  - ROC curve (AUC=0.999)
  - Precision-Recall curve (AUC=0.348)
  - Feature importance plots (top 20 + by category)
* **Notebooks:**
  - `notebooks/Phase 3 - Model Training/Model Training.ipynb` (original)
  - `notebooks/Phase 3 - Model Training/Model Training v2.ipynb` (strategy testing)
  - `notebooks/Phase 3 - Model Training/Model Training v3.ipynb` (final model)

***

### Phase 4: Demo & Deployment Scripts (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Created **production-ready inference scripts** for terminal-based timestomping detection.
* Developed two deployment options: quick demo (pre-engineered features) and full pipeline (raw artifacts).
* Implemented feature alignment to handle different CSV formats from various forensic cases.

**Key Achievements:**

1. **Option 1: Quick Demo (`predict_timestomping.py`):**
   - **Input:** Pre-engineered features CSV (from Phase 2 or similar processing)
   - **Process:** Load model → Prepare features → Generate predictions → Save results
   - **Output:** 3 files (predictions.csv, flagged_files.csv, summary_report.txt)
   - **Use case:** Fast inference on already-processed data (~30 seconds for 300K events)

2. **Option 2: Full Pipeline (`full_pipeline_demo.py`):**
   - **Input:** Raw $LogFile CSV + $UsnJrnl CSV (unlabeled, direct from parsers)
   - **Process:**
     - Stage 1: Load and standardize column names (handles different parser formats)
     - Stage 2: Merge artifacts into master timeline
     - Stage 3: Engineer 75+ ML features (replicates Phase 2)
     - Stage 4: Load model and generate predictions with feature alignment
     - Stage 5: Save results and generate forensic triage report
   - **Output:** 5 files (predictions, flagged files, summary, optional intermediates)
   - **Use case:** End-to-end detection on new forensic cases (~1-2 minutes for 300K events)

3. **Feature Alignment System:**
   - Automatically handles mismatched one-hot encoded features (e.g., different file attributes)
   - Adds missing features with zeros (e.g., `usn_attr_Sparse` not in training data)
   - Removes extra features (e.g., `usn_attr_NewAttribute` not in model)
   - Reorders columns to match training data (preserves model compatibility)
   - **Tested on:** Case 06-APT (358,926 events) with successful alignment

4. **User-Friendly Interface:**
   - **Colored terminal output** (headers, success/error messages, progress indicators)
   - **Verbose mode** for detailed progress tracking
   - **Custom thresholds** for risk categorization (LOW/MEDIUM/HIGH confidence)
   - **Summary statistics** in terminal + detailed text report
   - **Help documentation** with usage examples and expected I/O

5. **Setup & Installation:**
   - **Virtual environment** created with all dependencies (pandas, sklearn, joblib, imbalanced-learn)
   - **Automated setup script** (`setup.sh`) for one-command installation
   - **Requirements file** (`requirements.txt`) for manual setup
   - **Quick start guide** (`QUICKSTART.md`) with 5-minute setup instructions

6. **Validation Testing:**
   - Tested on Case 06-APT (64,347 events)
   - Result: 0 timestomped files flagged (confirmed benign case with test data)
   - Feature alignment handled 82 features → 75 features (added 7 missing, removed 14 extra)
   - Pipeline completed successfully in ~1 minute

**Practical Deployment:**
```bash
# One-time setup
cd demo
./setup.sh

# Activate environment
source venv/bin/activate

# Option 1: Quick demo
python predict_timestomping.py ../data/processed/Phase\ 2\ -\ Feature\ Engineering/features_engineered.csv

# Option 2: Full pipeline on new case
python full_pipeline_demo.py "test csv/06-APT-LogFile.csv" "test csv/06-APT-UsnJrnl.csv" \
  --output-dir ./results --case-id 6 --verbose
```

**Output:**
* ✅ Demo scripts:
  - `demo/predict_timestomping.py` (4,800 lines with comprehensive comments)
  - `demo/full_pipeline_demo.py` (6,200 lines with full pipeline)
* ✅ Setup files:
  - `demo/setup.sh` (automated installation)
  - `demo/requirements.txt` (Python dependencies)
  - `demo/README.md` (complete documentation)
  - `demo/QUICKSTART.md` (5-minute guide)
* ✅ Test data: `demo/test csv/` (Case 06-APT LogFile + UsnJrnl)
* ✅ Virtual environment: `demo/venv/` (isolated Python environment)

**Key Innovation:**
These scripts bridge the gap between trained models and operational forensic workflows, enabling investigators to run timestomping detection directly from the terminal on new cases without Python programming knowledge.

***

## 🔬 Methodology Overview

| Sub-Phase | Objective | Key Process / Design Insight | Results |
| :--- | :--- | :--- | :--- |
| **A. Data Labelling** ✅ | Annotate artifacts with ground truth labels. | **Critical Design Decision:** Separate `is_timestomped` (LABEL) from `timestomp_tool_executed` (FEATURE). Only actual manipulation is the target variable. | 252 timestomped events. 16 tool executions as features. |
| **B. Data Case Merging** ✅ | Create a unified timeline per case (LogFile + UsnJrnl). | **Solution:** Smart UsnJrnl Aggregation (3.1M → 631K) + Prioritized Outer Join. Preserved 98.0% of labels with zero duplication. | Dataset reduction: 3.4M → 825K rows (75.5% reduction). 12 merged datasets. |
| **C. Master Timeline Creation** ✅ | Aggregate all 12 cases into a single training dataset. | Vertical concatenation and chronological sorting. LSN/USN uniqueness validated within each case. | Single unified dataset (824,605 events, 247 labels). |
| **D. Feature Engineering** ✅ | Transform raw timeline into ML-ready features. | **Solution:** Timestamp recovery (69% from lf_creation_time), vectorized event frequency calculations, comprehensive anomaly detection (8 features), categorical encoding (46 cross-artifact features). | 778,692 events, 87 features, 100% label preservation. Class imbalance: 1:3,151. |
| **E. Model Training** 🔄 | Train ML models for timestomping detection. | Case-based stratified split + SMOTE oversampling + class weights. Random Forest & XGBoost with hyperparameter tuning. | **In Progress** |

***

## 🛠️ Project Structure (Updated)

Digital-Detectives_Thesis/ ├── data/ │ ├── raw/ │ │ └── suspicious/ # Ground truth labels │ └── processed/ # Cleaned & engineered data │ ├── Phase 1 - Data Collection & Preprocessing/ │ │ ├── A. Data Labelled/ # ✅ Phase 1A output │ │ │ ├── XX-PE-LogFile-Labelled.csv │ │ │ └── XX-PE-UsnJrnl-Labelled.csv │ │ ├── B. Data Case Merging/ # ✅ Phase 1B output │ │ │ └── XX-PE-Merged.csv # 12 merged case files │ │ └── C. Master Timeline/ # ✅ Phase 1C output │ │ └── master_timeline.csv # Unified dataset (824,605 events) │ └── Phase 2 - Feature Engineering/ # ✅ Phase 2 output │ └── features_engineered.csv # ML-ready dataset (778,692 events, 87 features) ├── notebooks/ │ ├── Phase 1 - Data Collection & Preprocessing/ │ │ ├── A. Data Labelling.ipynb # ✅ Completed │ │ ├── B. Data Case Merging.ipynb # ✅ Completed │ │ └── C. Master Timeline Creation.ipynb # ✅ Completed │ ├── Phase 2 - Feature Engineering/ │ │ └── Feature Engineering.ipynb # ✅ Completed │ └── Phase 3 - Model Training & Evaluation/ # 🔄 In Progress ├── models/ # Trained model artifacts └── outputs/ # Evaluation reports & visualizations

***

## 📊 Dataset Evolution Summary

| Phase | Records | Features | Timestomped Events | Key Transformation |
| :--- | ---: | ---: | ---: | :--- |
| **Raw Data** | 3,372,330 | 34 | 252 | Initial forensic artifacts |
| **Phase 1B: Merged** | 824,605 | 34 | 247 | 75.5% reduction, deduplicated |
| **Phase 1C: Master Timeline** | 824,605 | 34 | 247 | Unified 12 cases |
| **Phase 2: Engineered** | 778,692 | 87 | 247 | Feature engineering, timestamp recovery |

**Overall Pipeline Efficiency:**
* **Data Reduction:** 3.4M → 779K events (77% reduction).
* **Feature Expansion:** 34 → 87 features (156% increase).
* **Label Preservation:** 252 → 247 (98% preservation rate, 100% in final dataset).
* **Class Imbalance:** 1:13,382 → 1:3,151 (improved through intelligent cleanup).