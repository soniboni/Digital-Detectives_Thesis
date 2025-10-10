**🕵️ Digital Detectives – Autopsy Plugin Prototype**

This repository is part of our thesis project, which aims to develop a prototype Autopsy plugin capable of detecting timestomped files through in-depth analysis of $LogFile and $UsnJrnl artifacts. The system integrates machine learning models to enhance detection accuracy and reduce false positives.

**🚧 Work in Progress – The plugin is currently under active development.**

---

## 📊 Project Progress & Status Update

| Phase | Task | Status | Output |
| :--- | :--- | :--- | :--- |
| **Phase 1** | A. Data Labelling | ✅ **Completed** | 24 labelled CSV files (12 LogFile + 12 UsnJrnl) |
| | B. Data Case Merging | ✅ **Completed** | 12 merged case files (01-PE through 12-PE) |
| | C. Master Timeline Creation | 🔄 **In Progress** | - |
| **Phase 2** | Feature Engineering | ⏳ **Pending** | - |
| **Phase 3** | Model Training & Evaluation | ⏳ **Pending** | - |

***

## ✅ Latest Accomplishments (Phase 1)

### Phase 1B: Data Case Merging (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Successfully merged **LogFile** and **UsnJrnl** artifacts for all 12 forensic cases.
* Eliminated duplicate events while preserving cross-artifact correlation.
* Achieved **98.0% label preservation rate** (247/252 timestomped events).

**Key Achievements:**

1.  **Massive Dataset Reduction**: 3,372,330 $\rightarrow$ 824,605 events (**75.5% reduction**).
2.  **Zero Duplication**: Each LSN and USN appears exactly once.
3.  **Preserved Events**: 247/252 timestomped events (98.0%):
    * 9 LogFile timestomped events (actual manipulation).
    * 238 UsnJrnl timestomped events.
    * **230 events detected by BOTH artifacts** (high confidence!).
4.  **UsnJrnl Aggregation**: 3.1M $\rightarrow$ 631K events (removed 2.5M duplicates).

**Technical Implementation:**
* **Smart UsnJrnl Aggregation**: Grouped multiple events at the same timestamp+filepath+filename and combined event sequences (`File_Created` $\rightarrow$ `Data_Added` $\rightarrow$ `File_Closed`). Preserved detection flags using `max()` aggregation.
* **Intelligent Join Strategy**: Outer join on `eventtime` + `filepath` + `filename`. Prioritized "**Time Reversal Event**" over "**File Creation**" for duplicate LogFile entries. Preserved all LogFile-only and UsnJrnl-only events.
* **Label & Feature Preservation**: Separated `is_timestomped` (Primary label) from `timestomp_tool_executed` (Feature).

**Understanding the 5 "Missing" LogFile Events:**

* The 5 "missing" events are **NOT actual timestomping events**—they are tool execution indicators (e.g., Prefetch files like `NTIMESTOMP_V1.2_X64.EXE-6EA682C3.pf`).
* They have `is_timestomped=0` but `timestomp_tool_executed=1`. They indicate a tool was executed, but not the timestamp manipulation itself.
* They are **FEATURES, not LABELS** (a core Phase 1A design principle) and are fully preserved via the `timestomp_tool_executed=1` field.
* **Impact**: All 247 actual timestomped events were preserved with zero data loss!

**Output:**
* ✅ 12 merged datasets: `data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/XX-PE-Merged.csv`
* ✅ Column structure: `case_id`, `eventtime`, `filename`, `filepath` (leftmost), with artifact prefixes `lf_*` and `usn_*`.
* **Notebook**: `Phase 1 - Data Collection & Preprocessing/B. Data Case Merging.ipynb`

***

### Phase 1A: Data Labelling (Completed)

**Date Completed:** October 10, 2025

**What We Did:**
* Successfully labelled all 12 forensic cases by matching suspicious behavior indicators from NTFS Log Tracker to forensic artifacts.
* **Critical Design Decision**: Separated tool execution ($\rightarrow$ **FEATURE**) from actual timestamp manipulation ($\rightarrow$ **LABEL**).

**Key Findings:**
* **Total Records Processed**: 3,372,330 events.
* **Actual Timestomped Events**: 252 unique events (14 LogFile + 238 UsnJrnl).
* **Tool Executions Detected**: 16 events (tracked as features).
* **Class Imbalance Ratio**: **1:13,382** (extreme imbalance).

**Understanding the $504 \rightarrow 252$ Reduction:**
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
* **Notebook**: `Phase 1 - Data Collection & Preprocessing/A. Data Labelling.ipynb`

***

## 🎯 Current Work: Phase 1C - Master Timeline Creation

**Objective:** Consolidate all 12 merged case files into a single unified dataset for machine learning.

**Strategy:**
1.  Load all 12 merged case files (`XX-PE-Merged.csv`).
2.  Concatenate vertically, maintaining `case_id`.
3.  Sort chronologically by `case_id` and `eventtime`.
4.  Final validation: Verify 247 timestomped events across the complete dataset.

**Expected Results:**
* Single unified dataset: **~825K events**.
* 247 timestomped events preserved.
* Ready for Phase 2: Feature Engineering.

**Output:** `data/processed/Phase 1 - Data Collection & Preprocessing/C. Master Timeline/master_timeline.csv`

***

## 🔬 Methodology Overview

| Sub-Phase | Objective | Key Process / Design Insight | Results |
| :--- | :--- | :--- | :--- |
| **A. Data Labelling** ✅ | Annotate artifacts with ground truth labels. | **Critical Design Decision**: Separate `is_timestomped` (LABEL) from `timestomp_tool_executed` (FEATURE). Only actual manipulation is the target variable. | 252 timestomped events. 16 tool executions as features. |
| **B. Data Case Merging** ✅ | Create a unified timeline per case (LogFile + UsnJrnl). | **Solution**: Smart UsnJrnl Aggregation (3.1M $\rightarrow$ 631K) + Prioritized Outer Join. Preserved 98.0% of labels with zero duplication. | Dataset reduction: 3.4M $\rightarrow$ 825K rows (75.5% reduction). 12 merged datasets. |
| **C. Master Timeline Creation** 🔄 | Aggregate all 12 cases into a single training dataset. | Vertical concatenation and chronological sorting. | Single unified dataset (~825K events) ready for Phase 2. |

***

## 🛠️ Project Structure (Updated)
Digital-Detectives_Thesis/
├── data/
│   ├── raw/
│   │   └── suspicious/                      # Ground truth labels
│   └── processed/                          # Cleaned & engineered data
│       └── Phase 1 - Data Collection & Preprocessing/
│           ├── A. Data Labelled/           # ✅ Phase 1A output
│           │   ├── XX-PE-LogFile-Labelled.csv
│           │   └── XX-PE-UsnJrnl-Labelled.csv
│           ├── B. Data Case Merging/       # ✅ Phase 1B output
│           │   └── XX-PE-Merged.csv        # 12 merged case files
│           └── C. Master Timeline/         # 🔄 Phase 1C output
│               └── master_timeline.csv     # Unified dataset
├── notebooks/
│   ├── Phase 1 - Data Collection & Preprocessing/
│   │   ├── A. Data Labelling.ipynb         # ✅ Completed
│   │   ├── B. Data Case Merging.ipynb      # ✅ Completed
│   │   └── C. Master Timeline Creation.ipynb # 🔄 In Progress
│   └── Phase 2 - Feature Engineering.ipynb
├── models/
└── outputs/