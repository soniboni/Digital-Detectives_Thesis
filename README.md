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
| **Phase 3** | Model Training & Evaluation | 🔄 **In Progress** | - |

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

## 🎯 Current Work: Phase 3 - Model Training & Evaluation

**Objective:** Train machine learning models to detect timestomped files using the engineered feature set from Phase 2.

**Strategy:**
1. **Data Splitting:** Case-based stratified split (prevent data leakage, maintain case integrity).
2. **Handle Class Imbalance:** SMOTE oversampling + class weights (1:3,151 imbalance ratio).
3. **Model Training:** Random Forest & XGBoost with hyperparameter tuning.
4. **Evaluation Metrics:** Precision-Recall focus (not accuracy due to extreme imbalance).
5. **Interpretability:** Feature importance analysis + SHAP values.

**Target Metrics:**
* **Precision > 90%** (minimize false positives).
* **Recall > 85%** (catch actual timestomping).
* **F1-Score balance** between precision and recall.
* **AUC-ROC & PR curves** for threshold optimization.

**Expected Challenges:**
* Extreme class imbalance (247 positive / 778,445 negative).
* Preventing overfitting to specific cases.
* Balancing false positives vs. false negatives in forensic context.

**Output (Expected):**
* Trained models: Random Forest, XGBoost.
* Evaluation reports: Classification reports, confusion matrices, ROC/PR curves.
* Feature importance rankings: Top predictive features for timestomping detection.

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