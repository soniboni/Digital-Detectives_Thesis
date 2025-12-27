# Model Training Explanation

## Overview

This document explains the machine learning approach used for NTFS timestomping detection, based on the forensic methodology by Oh, Lee, and Hwang (2024).

---

## Methodology Foundation

### Base Research

**Oh, Lee, and Hwang (2024)** - "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"
- Published in IEEE Access (DOI: 10.1109/ACCESS.2024.10517044)
- Proposes rule-based detection algorithms using $LogFile and $UsnJrnl artifacts
- Identifies 31 forensic features and cross-artifact validation approach
- Uses algorithmic pattern matching and heuristics (not machine learning)

### Our ML Implementation

This thesis applies **machine learning** to Oh et al.'s forensic methodology:
- Uses Oh et al.'s 31 forensic features as ML input
- Trains LightGBM classifier to automatically learn timestomping patterns
- Achieves comparable detection performance with automated pattern recognition

**Key Difference:**
- Oh et al.: Rule-based algorithms with manual pattern matching
- Our work: ML-based classification learning patterns from training data

---

## File-Level vs Event-Level Validation

### Why File-Level Validation is Correct

**Forensic Objective**: The goal of timestomping detection is to answer: "Which FILES have been tampered with?"

This is fundamentally different from: "Which specific EVENTS represent timestomping?"

### Rationale

**1. Evidence Admissibility**
- In forensic investigations, analysts need to determine if a file's metadata can be trusted
- A file that has been timestomped even once is compromised as evidence
- The number of timestomping events or which specific event is less relevant than the fact that tampering occurred

**2. Journaling Artifacts Document Events**
- $LogFile records operations that modify file metadata (events)
- $UsnJrnl records file system changes including timestamp modifications (events)
- These artifacts provide EVIDENCE that timestomping occurred, not the detection targets themselves

**3. Multiple Events Per File**
- A single file can undergo multiple timestamp manipulation operations
- Example from Lone Wolf validation:
DeathToll.jpg has 6 Basic_Info_Changed events: - USN 239046184 (2018-04-05 10:21:01) - USN 239046272 (2018-04-05 10:21:01) <- Oh et al. flagged this - USN 239081712 (2018-04-05 10:21:11) - USN 239081800 (2018-04-05 10:21:11) - USN 249181480 (2018-04-06 20:35:35) - USN 249181568 (2018-04-06 20:35:35) <- Our model detected this

- All 6 events provide evidence that DeathToll.jpg was timestomped
- Detecting ANY of these events correctly identifies the file as timestomped

**4. Oh et al. Methodology Supports File-Level**
- Oh et al.'s pipeline includes explicit file-level aggregation step
- Detection approach: Aggregate events by filename to create one prediction per file
- Forensic features calculated at file level (not event level)

### Oh et al. Detection Pipeline Overview

Raw Artifacts ($LogFile, $UsnJrnl) | v Event Filtering (Basic_Info_Changed, Time Reversal) | v Cross-Artifact Merging (Join LogFile + UsnJrnl by filename) | v FILE-LEVEL AGGREGATION (Group by filename, keep one row per file) | v Feature Extraction (31 forensic features per FILE) | v Detection Algorithm (Rule-based patterns OR ML classification) | v File-Level Predictions


### Validation Criteria

**Correct Detection**: Model flags a file that has been timestomped, regardless of which specific event triggered the detection.

**Example**:
- Ground truth: "DeathToll.jpg was timestomped" (based on any of 6 events)
- Oh et al. detection: Flagged via USN 239046272
- Our detection: Flagged via USN 249181568
- Both are CORRECT - same file identified as timestomped

**Incorrect Detection**: Model fails to flag a file that was timestomped, OR flags a file that was never timestomped.

---

## How Our Model Works

### Step 1: Data Collection and Merging

**Input Artifacts**:
- $LogFile CSV: Contains time reversal events and file metadata changes
- $UsnJrnl CSV: Contains file system change records including Basic_Info_Changed events
- Suspicious CSV (training only): Ground truth labels

**Merging Strategy** (Oh et al. approach):
1. Filter LogFile for time reversal events
2. Filter UsnJrnl for Basic_Info_Changed events
3. Merge LogFile + UsnJrnl by filename (outer join to preserve all evidence)
4. Attach ground truth labels where available

**Key Insight**: Filtering BEFORE merging (not after) preserves suspicious events that might otherwise be lost.

### Step 2: File-Level Aggregation

**Purpose**: Reduce multiple events per file to ONE ROW per file for ML training.

**Aggregation Logic**:
- Group all events by filename
- Keep the event with highest USN/LSN (most recent)
- Preserve all forensic indicators from any event (OR logic for boolean features)
- Carry forward ground truth label if ANY event was suspicious

**Result**: One training example per file, with aggregated evidence from all events.

### Step 3: Feature Engineering

**31 Forensic Features** extracted per file (from Oh et al. methodology):

**Timestamp-Based Features**:
- `zero_in_nanoseconds`: Timestamp has exactly 0 nanoseconds (SetFileTime() signature)
- `zero_in_nanoseconds_lf`: LogFile-specific zero nanoseconds detection
- `time_reversal_event`: Timestamp changed to earlier value (from LogFile)
- `timestamp_changed_to_past`: Boolean indicator of time reversal
- `using_another_timestamp`: File using another file's timestamp pattern

**Cross-Artifact Validation**:
- `cross_artifact_validation_score`: Weighted score from multiple evidence sources
- `cross_artifact_detected`: Boolean - evidence from BOTH LogFile AND UsnJrnl
- `has_logfile_evidence`: File appears in filtered LogFile events
- `has_usnjrnl_evidence`: File appears in filtered UsnJrnl events

**UsnJrnl-Specific Features**:
- `basic_info_changed`: Basic_Info_Changed event detected
- `modified_creationtime`: Creation time was modified
- `modified_modifiedtime`: Modified time was modified
- `modified_accessedtime`: Access time was modified

**Combined Indicators**:
- `zero_nano_time_reversal`: Both zero nanoseconds AND time reversal detected

**File Characteristics**:
- `path_depth`: Nesting level in directory structure
- `filename_length`: Length of filename
- `is_executable`: .exe, .dll, .sys extensions
- `is_document`: .docx, .pdf, .txt extensions
- `is_archive`: .zip, .rar, .7z extensions
- `is_image`: .jpg, .png, .gif extensions

**Plus 12 additional features** capturing timestamp relationships and metadata patterns.

### Step 4: Model Training (LightGBM)

**Algorithm Selection**:
- LightGBM chosen as best performer among 4 algorithms tested
- Compared against: Random Forest, XGBoost, Logistic Regression
- LightGBM advantages: Fast training, handles imbalanced data well, good interpretability

**Training Configuration**:
- 18 training datasets (PE + APT cases)
- 192 timestomped files, 70,538 benign files (0.27% class imbalance)
- Class weighting applied to handle imbalance
- Features: 31 forensic indicators (no label leakage)

**Training Results**:
- Recall: 100% (74/74 suspicious files detected on test set)
- Precision: 46.3% (86 false positives out of 17,578 benign files)
- F1-Score: 0.632
- False Positive Rate: 0.49%

### Step 5: Prediction and Validation

**Production Pipeline** (Notebooks 01-04):

**Notebook 01 - Load Data**:
- Load LogFile and UsnJrnl CSVs
- Filter for suspicious events (Basic_Info_Changed, Time Reversal)
- Merge artifacts by filename
- Aggregate to one row per file

**Notebook 02 - Feature Engineering**:
- Extract 31 forensic features
- Create cross-artifact validation scores
- Output: data_features.csv

**Notebook 03 - Run Detection**:
- Load trained LightGBM model
- Generate probability predictions
- Convert to confidence percentage
- Flag files above threshold (default: 70%)

**Notebook 04 - View Results**:
- Load predictions and ground truth (if available)
- Calculate performance metrics (file-level)
- Generate visualizations
- Create summary report

**Validation on Lone Wolf (Held-Out Test Set)**:
- Ground truth: 12 known timestomped files
- Detection: 12/12 files detected (100% recall)
- Precision: 50-62.5% (12 known + 12 additional detections)
- F1-Score: 0.67-0.77
- All detections at 70%+ confidence

---

## Model Behavior and Interpretation

### What the Model Detects

The LightGBM model identifies files with patterns consistent with timestamp manipulation by:

1. **Primary indicators**: Zero nanoseconds + time reversal events
2. **Cross-validation**: Evidence from multiple artifacts increases confidence
3. **File patterns**: Executable files with timestamp anomalies rank higher
4. **Metadata inconsistencies**: Timestamps that violate expected relationships

### Confidence Scores

**High Confidence (90-100%)**:
- Multiple forensic indicators present
- Cross-artifact validation (both LogFile and UsnJrnl)
- Strong zero nanoseconds pattern
- Time reversal documented

**Medium Confidence (70-89%)**:
- Some forensic indicators present
- Single artifact evidence (LogFile OR UsnJrnl)
- Timestamp patterns consistent with manipulation

**Low Confidence (<70%)**:
- Weak or ambiguous indicators
- May be benign file system operations
- Not flagged by default threshold

### Why 100% Recall is Achievable

Perfect recall (100%) on a small test set is possible because:

1. **Strong forensic signatures**: SetFileTime() API creates consistent patterns
2. **Multiple evidence sources**: Cross-artifact validation provides redundancy
3. **Conservative threshold**: 70% confidence captures all true positives while accepting some false positives
4. **Effective feature engineering**: 31 features capture various manifestations of timestomping

**Important**: 100% recall on a 12-file test set validates that the model correctly learned the timestomping patterns. It does not guarantee perfect performance on all future datasets.

### Precision Trade-off

**Current precision (50-62.5%)** reflects the conservative detection strategy:
- Prioritizes catching all timestomped files (high recall)
- Accepts some false positives (lower precision)
- Suitable for forensic triage where false negatives are more costly than false positives

**Hyperparameter tuning (Phase 4)** aims to improve precision to 75-80% while maintaining 95%+ recall.

---

## Comparison: Oh et al. vs Our Implementation

| Aspect | Oh et al. (2024) | Our Implementation |
|--------|------------------|-------------------|
| Approach | Rule-based algorithms | Machine Learning (LightGBM) |
| Detection Method | Pattern matching + heuristics | Supervised classification |
| Features | 31 forensic indicators | Same 31 features as input |
| Training | N/A (no training) | 192 suspicious + 70,538 benign files |
| File System Tunneling | Algorithmic identification | Learned during training |
| Zero Nanoseconds | Manual rule checking | Feature for ML model |
| Cross-Artifact Validation | Manual score calculation | Feature for ML model |
| Validation Level | File-level | File-level |
| Scalability | Requires manual rule updates | Learns from new data |
| Performance | Not reported (rule-based) | Recall: 100%, Precision: 46.3%, F1: 0.632 |

**Conclusion**: Our implementation advances Oh et al.'s methodology by applying machine learning to their forensic feature framework, enabling automated pattern learning rather than manual rule definition.

---

## Validation Approach

### File-Level Metrics

**True Positive (TP)**: A file that was timestomped AND was flagged by the model
- Example: DeathToll.jpg (known timestomped, model flagged it)

**False Negative (FN)**: A file that was timestomped BUT was NOT flagged by the model
- Lone Wolf result: 0 false negatives (100% recall)

**False Positive (FP)**: A file that was NOT timestomped BUT was flagged by the model
- Lone Wolf result: 12 additional detections requiring investigation

**True Negative (TN)**: A file that was NOT timestomped AND was NOT flagged
- Majority of files in any dataset

### Why Event-Level Matching Would Be Incorrect

**Hypothetical event-level validation**:
- Ground truth: USN 239046272 for DeathToll.jpg
- Our detection: USN 249181568 for DeathToll.jpg
- Event-level result: MISS (different USN)
- File-level result: HIT (same file detected)

**Problem**: Event-level validation would penalize correct file identification simply because we detected a different event from the same timestomping operation.

**Forensic reality**: Both USNs represent valid evidence that DeathToll.jpg was timestomped. The specific event number is irrelevant to the forensic conclusion.

---

## Practical Application

### For Forensic Investigators

**Workflow**:
1. Export $LogFile and $UsnJrnl from suspect system to CSV
2. Run Prototype Tool notebooks (01-04)
3. Review flagged_files.csv (sorted by confidence)
4. Investigate high-confidence detections first
5. Use LSN/USN identifiers to cross-reference events
6. Examine lf_detail field for exact timestamp manipulation details

**Output Interpretation**:
- Confidence 90-100%: Very likely timestomped, prioritize investigation
- Confidence 70-89%: Likely timestomped, investigate with context
- Confidence <70%: Not flagged, considered benign

**Manual Verification**:
- Check lf_detail for "Zero in 100-nanoseconds" pattern
- Verify time reversal (timestamp changed to past)
- Cross-reference LSN (LogFile) and USN (UsnJrnl) in original artifacts
- Consider file context (system files vs user files)

### Limitations

1. **Requires NTFS journaling artifacts**: Does not work on other file systems
2. **Post-incident detection only**: Cannot prevent timestomping in real-time
3. **Precision trade-off**: Some false positives require manual review
4. **Training data bias**: Performance depends on similarity to training cases
5. **Sophisticated evasion**: Advanced attackers may find ways to avoid detection patterns
6. **Limited detection window**: $LogFile retains 2-3 hours, $UsnJrnl retains 30-40 hours of events

---

## References

Oh, J., Lee, S., & Hwang, Y. (2024). Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation. IEEE Access, 12, 65021-65035. DOI: 10.1109/ACCESS.2024.10517044
