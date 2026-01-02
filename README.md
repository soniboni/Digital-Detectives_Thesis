# Digital Detectives - NTFS Timestomping Detection

**Machine Learning-Based Detection of Timestamp Manipulation in NTFS File Systems**

This repository contains a thesis project developing machine learning models to detect timestamp manipulation (timestomping) in NTFS filesystems using $LogFile and $UsnJrnl artifacts, based on Oh et al. (2024) methodology.

---

## Project Overview

### Research Objective

Develop an ML-based system to automatically detect timestamp manipulation in NTFS filesystems by analyzing cross-artifact patterns in $LogFile and $UsnJrnl transaction logs.

### Base Methodology

**Oh, Lee, and Hwang (2024)** - "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"  
Published in IEEE Access ([DOI: 10.1109/ACCESS.2024.10517044](https://ieeexplore.ieee.org/document/10517044))

### Detection Approach

- **Core Logic**: Compare Redo vs Undo timestamps in $LogFile and cross-check MAC timestamps in $UsnJrnl. Flag suspicious files.
- **$FN Checker**: Detect $FN timestamp manipulation:
  1. Timestamp change occurs → file moved in same volume → timestamp change again (NtSetInformationFile API)
  2. Only MAC timestamps changed via SetFileTime() or PowerShell Get-Item. No re-manipulation.
- **Combined Logic**: Core logic first, then $FN checker for additional confirmation. Determines Malicious vs Suspicious.

---

## Dataset Structure

### Training Datasets (19 total)

**PE Cases (12)**: 01-PE → 12-PE  
**APT Cases (10)**: 01-APT17, 03-APT21, 04-APT28, 05-APT29, 06-APT30, 07-APT37, 08-APT38, 10-DarkHotel663, 11-DarkHotelbbd, 14-Winnti53b

Each dataset contains:
- `LogFile.csv` – NTFS transaction log with timestamp change events  
- `UsnJrnl.csv` – NTFS change journal with file modification events  
- `MFT.csv` – file metadata  
- `Suspicious.csv` – ground truth labels  

### Validation Datasets (5 total)

- **Lone Wolf**: 12 timestomped files  
- **APT40 / Kimusky / Winnti / APT19**: individual validation files

**Notes:** Patterns include zero nanoseconds, file move, or combination.

---

## Columns Needed Per Parsed File

### LogFile.csv

| Column | Source | Purpose |
| --- | --- | --- |
| LSN | $LogFile | Map record to MFT entry |
| Redo OP | $LogFile | Detect timestamp change operations |
| Record Offset | $LogFile | Check proper record type |
| Attribute Offset | $LogFile | Determine which timestamps to read |
| Undo Data (4 columns) | $LogFile | Extract “before” timestamps ($SI-C/M/E/A) |
| Redo Data (4 columns) | $LogFile | Extract “after” timestamps ($SI-C/M/E/A) |
| Target VCN | $LogFile | Identify target file cluster if no LSN match |
| MFT Cluster Index | $LogFile | Identify target file entry within cluster |
| Timestamp | $LogFile | Event timestamp for ordering / FST check |

### UsnJrnl.csv

| Column | Source | Purpose |
| --- | --- | --- |
| USN | $UsnJrnl | Record identifier for ordering & reference |
| FRN | $UsnJrnl | Map record to target file |
| Reason Flag | $UsnJrnl | Detect BASIC_INFO_CHANGE, CLOSE, FILE_CREATE |
| Timestamp | $UsnJrnl | Event ordering / time delta for cross-check |
| File Name | $UsnJrnl | Reporting / FST cross-check |
| File Path | $UsnJrnl | Reporting / FST cross-check |
| Parent FRN | $UsnJrnl | Optional: reconstruct path, parent-child relations |
| Record Status | Derived | Indicates active / deleted / renamed status |

### MFT.csv

| Column | Source | Purpose |
| --- | --- | --- |
| Entry Number / FRN | MFT | Map to LogFile LSN or USN FRN |
| $SI-C | MFT | Check for creation timestamp change |
| $SI-M | MFT | Check for modification timestamp |
| $SI-E | MFT | Check for entry modification timestamp |
| $SI-A | MFT | Optional, mostly for enrichment |
| File Name | MFT | Identify target file |
| File Path | MFT | Identify target file |
| Entry Active/Inactive | MFT | Avoid false positives |
| LSN | MFT | Map LogFile records to MFT entry |

---

## Project Phases

### Phase 1: Raw Data Parsing
- Preserve nanoseconds.
- Parse MFT, $LogFile, $UsnJrnl.
- Output: `MFT.csv`, `LogFile.csv`, `UsnJrnl.csv`.

### Phase 2: Data Preprocessing & Joining
- Join LogFile/UsnJrnl with MFT.
- Normalize timestamps (UTC, nanoseconds).
- Handle missing/incomplete events.
- Group events **per file, ordered by time**.
- Output: `grouped_events.csv`.

### Phase 3: Feature Engineering & Labeling
- Sliding windows: per file, per N events, per time interval.
- Features:
  - Redo vs Undo timestamp differences
  - Aggregated stats: min/max/mean per window
  - USN reason counts per window
  - Event density / temporal anomalies
- Label windows or files as **timestomped** or **normal** using ground truth (Suspicious.csv from Oh et al.).
- Output: `features.csv` (row = file/window, columns = features + labels)

### Phase 4: Model Training & Evaluation
- Split: train/validation/test (70/15/15)
- Algorithms: Random Forest, LightGBM, XGBoost, Logistic Regression
- Evaluate per file/window: precision, recall, F1
- Validate using Lone Wolf dataset

### Phase 5: Hyperparameter Tuning
- Fine-tune best model
- Re-validate

### Phase 6: Autopsy Integration
- Wrap parser + feature engineering + ML model in Python module
- Input: NTFS volume
- Output: flagged files/events with timestamps, reason, confidence
- Optional visualization/report
- Test plugin with multiple datasets

---

## Model Output

- Row = **file or window**, not individual event
- Columns = engineered features + label
- Provides reason and confidence
- Individual events traceable via LSN/USN
- CSV format preferred for Autopsy and ML training

---

**Important Notes:**
- Core logic + $FN checker is sufficient to detect a significant portion of timestomped files.
- Event sequencing occurs in Phase 2 (grouping per file, ordered by time)
- Feature engineering operates on grouped events (Phase 3)
- All timestamps preserved at nanosecond precision