# Phase 2A v2.0: Location-Agnostic Feature Engineering

## Overview

This notebook addresses the critical location-based overfitting discovered in Phase 4 v1.0 model evaluation, where the model achieved 99.6% F1-score on internal data but **0% detection on external APT datasets**.

## Critical Problem Identified

**v1.0 Model Overfitting Analysis:**
- `in_temp_dir`: 30.01% feature importance (PRIMARY overfitting source)
- `in_program_files`: 3.16%
- `in_windows_dir`: 2.50%
- `in_system32`: 0.06%
- `in_users_dir`: 0.52%
- **Total location feature importance: 36.25%**

The v1.0 model learned environmental bias ("files in C:\Temp\ are timestomped") instead of intrinsic manipulation characteristics. This caused complete failure on external datasets with different filesystem layouts.

## v2.0 Solution Strategy

This notebook creates **23 location-agnostic features** that capture:
1. **Temporal patterns** (42% importance in v1.0) - RETAINED ALL
2. **Cross-artifact validation** - Evidence from multiple forensic sources
3. **File system patterns** - Manipulation signatures (BASIC_INFO_CHANGE, tunneling)
4. **File type characteristics** - Executable analysis
5. **NEW forensic features** - Composite scoring systems

## Input/Output

- **Input**: `/data/processed/Phase 1 - V2 Data Cleaning/all_cases_combined_v2.csv`
  - Records: 283,118
  - Columns: 48
  - Timestomped events: Variable by case

- **Output**: `/data/processed/Phase 2A - V2 Location Agnostic Features/all_cases_combined_v2_phase2a.csv`
  - Records: 283,118 (unchanged)
  - Columns: 71 (48 base + 23 features)
  - Expected size: ~150-200 MB

## Features Created (23 Total)

### Group 1: File Type Features (6 features)
```python
1. is_executable               # .exe, .dll, .sys, .bat, .cmd, .ps1
2. is_system_file              # System attribute present
3. is_hidden_file              # Hidden attribute present
4. is_archive                  # Archive attribute present
5. filename_length             # Character count
6. has_suspicious_extension    # .tmp, .log, .bak
```

**Rationale**: File type analysis is location-agnostic. APT malware targets executables (Oh et al. 2024 Table 8).

### Group 2: Temporal Features (6 features) - CRITICAL
```python
7.  event_frequency_per_file           # v1.0 importance: 5.85%
8.  event_frequency_per_case           # Context for batch detection
9.  events_in_1min_window              # v1.0 importance: 21.31% (HIGHEST)
10. events_in_5min_window              # v1.0 importance: 10.53%
11. time_since_previous_event_seconds  # v1.0 importance: 2.44%
12. time_until_next_event_seconds      # v1.0 importance: 1.59%
```

**Rationale**: Combined 42% importance in v1.0. Captures automated timestomping tool signatures (batch operations, rapid-fire changes). These are environment-independent.

### Group 3: Cross-Artifact Features (3 features)
```python
13. source_confidence_score    # both=2, single=1, none=0
14. has_logfile_evidence       # Time reversal events (SetFileTime API)
15. has_usnjrnl_evidence       # BASIC_INFO_CHANGE events (MFT changes)
```

**Rationale**: Multi-source validation is core forensic principle (Oh et al. 2024). Evidence from BOTH artifacts is stronger.

### Group 4: UsnJrnl Pattern Features (3 features)
```python
16. usn_basic_info_change              # BASIC_INFO_CHANGE in usn_event_info
17. usn_file_closed                    # Close in usn_event_info
18. usn_complete_manipulation_pattern  # Both patterns present
```

**Rationale**: UsnJrnl event sequences reveal manipulation signatures. BASIC_INFO_CHANGE + Close = complete timestomping operation.

### Group 5: Path Depth (1 feature - CORRECTED)
```python
19. path_depth  # Count of backslashes in filepath
```

**Rationale**: Depth is location-agnostic structural measure. Shallow paths (2-3) vs deep paths (8+) indicate different risk profiles.

**v1.0 Bug Fix**: Original used incorrect regex, returned 0 for all paths. v2.0 uses `r'\\\\'` to count actual backslashes.

### Group 6: Event-Time Comparison (1 feature)
```python
20. event_vs_modified_after_days  # Days between event time and manipulated timestamp
```

**Rationale**: Large positive delta = backdating, large negative delta = forward-dating, small delta = normal.

### Group 7: NEW v2.0 Forensic Features (3 features)
```python
21. cross_artifact_validation_score (0-3 points)
    # 3 pts: source='both' (Time Reversal + BASIC_INFO_CHANGE)
    # 2 pts: source='logfile_only' (Time Reversal Event)
    # 1 pt:  source='usnjrnl_only' AND has BASIC_INFO_CHANGE
    # 0 pts: Otherwise

22. timestamp_manipulation_pattern_score (0-3 points)
    # +1 if rapid sequential manipulation (>10 events in 1min window)
    # +1 if complete UsnJrnl pattern (BASIC_INFO_CHANGE + CLOSE)
    # +1 if multiple manipulations on same file
    # Max 3 points

23. file_system_tunneling_confidence (0.0-1.0)
    # Uses Phase 1 is_tunneling column
    # 1.0 = tunneling detected, 0.0 = no tunneling
```

**Rationale**: Composite features combine existing signals into forensic scoring systems. Based on Oh et al. (2024) multi-artifact validation and automation detection principles.

## Features REMOVED (Location-Based Overfitting)

**DO NOT CREATE** - These caused 36.25% location-based overfitting:
```python
# REMOVED from v1.0:
- in_temp_dir        # 30.01% importance - PRIMARY overfitting source
- in_program_files   # 3.16%
- in_windows_dir     # 2.50%
- in_system32        # 0.06%
- in_users_dir       # 0.52%
```

## Notebook Structure

1. **Setup & Load Data** (Cells 1-4)
   - Import libraries
   - Configure paths
   - Load 283,118 records
   - Parse timestamps

2. **Group 1: File Type Features** (Cells 5-11)
   - 6 features
   - 100% coverage

3. **Group 2: Temporal Features** (Cells 12-19)
   - 6 features (CRITICAL - 42% importance)
   - Includes 2-3 minute processing for temporal clustering
   - Progress bars for long operations

4. **Group 3: Cross-Artifact Features** (Cells 20-24)
   - 3 features
   - Multi-source validation

5. **Group 4: UsnJrnl Pattern Features** (Cells 25-29)
   - 3 features
   - Manipulation signature detection

6. **Group 5: Path Depth** (Cells 30-32)
   - 1 feature (CORRECTED from v1.0)

7. **Group 6: Event-Time Comparison** (Cells 33-35)
   - 1 feature
   - Backdating/forward-dating detection

8. **Group 7: NEW v2.0 Forensic Features** (Cells 36-40)
   - 3 NEW features
   - Composite scoring systems

9. **Validation & Summary** (Cells 41-48)
   - Data integrity checks
   - Missing value analysis
   - **CRITICAL**: Validation that NO location features created
   - Feature coverage on timestomped events

10. **Save Dataset** (Cells 49-50)
    - Export to CSV
    - File size verification

11. **Phase 2A v2.0 Summary Report** (Cells 51-52)
    - Complete feature list
    - Improvements over v1.0
    - Next steps

## Expected Runtime

- **Total**: 3-5 minutes on 283,118 records
- **Longest operation**: Temporal clustering (cells 9-10) - 2-3 minutes
- **Memory usage**: ~300-400 MB peak

## Validation Checks

The notebook includes critical validation checks:

1. **Record count**: 283,118 (unchanged from input)
2. **Column count**: 71 (48 base + 23 features)
3. **NO location features**: Validates none of the 5 removed features present
4. **Missing values**: Reports coverage for all features
5. **Timestomped coverage**: Shows feature distribution on timestomped events

## Key Improvements Over v1.0

### 1. Location Independence
- Removed ALL location-based features (36.25% importance)
- Model will learn intrinsic manipulation patterns
- Should generalize to external APT datasets

### 2. Temporal Focus
- Retained ALL 6 temporal features (42% combined importance)
- Captures automated tool signatures
- Works on any filesystem layout

### 3. NEW Forensic Features
- `cross_artifact_validation_score`: Multi-source evidence strength
- `timestamp_manipulation_pattern_score`: Automation detection
- `file_system_tunneling_confidence`: Tunneling vs manipulation

### 4. Bug Fixes
- `path_depth`: Fixed regex (was returning 0 for all paths in v1.0)

## Research Foundation

Based on Oh et al. (2024) "Forensic Detection of Timestamp Manipulation for Digital Investigations":

1. **Multi-artifact validation** (Section 3.2)
   - Cross-artifact features leverage this principle
   - Evidence from multiple sources strengthens detection

2. **Temporal clustering** (Section 4.3)
   - Automated tools create temporal patterns
   - Batch operations detectable via event clustering

3. **File system patterns** (Section 3.3)
   - BASIC_INFO_CHANGE indicates MFT timestamp modification
   - File system tunneling can mimic manipulation

## Usage

```bash
# Open Jupyter notebook
jupyter notebook "01_Location_Agnostic_Feature_Engineering_v2.ipynb"

# Run all cells (Kernel -> Restart & Run All)
# Expected runtime: 3-5 minutes

# Output will be saved to:
# /data/processed/Phase 2A - V2 Location Agnostic Features/all_cases_combined_v2_phase2a.csv
```

## Next Steps: Phase 3 - Model Training v2.0

With location-agnostic features, the next phase will:
1. Train Random Forest on 23 features (NO location features)
2. Evaluate on internal validation set (Cases 1-12)
3. **CRITICAL**: Evaluate on external APT datasets (Alharbi et al. 2016)
4. Compare v2.0 vs v1.0 generalization performance

**Expected outcome**: Better generalization to external datasets (>0% detection rate)

## Troubleshooting

### Memory Issues
If notebook crashes due to memory:
- Close other applications
- Process cases in smaller batches for temporal clustering
- Reduce data types (e.g., int64 -> int32 where appropriate)

### Long Runtime
Temporal clustering (cells 9-10) processes 283,118 records:
- Expected: 2-3 minutes
- Progress bar shows case-by-case progress
- DO NOT interrupt - will need to restart

### File Not Found
Ensure Phase 1 v2.0 output exists:
```bash
ls -lh "/data/processed/Phase 1 - V2 Data Cleaning/all_cases_combined_v2.csv"
```

Should show ~283,118 records.

## References

1. Oh, J., Lee, S., & Lee, S. (2024). Forensic Detection of Timestamp Manipulation for Digital Investigations. *Forensic Science International: Digital Investigation*.

2. Alharbi, S., Weber, J., & Chen, X. (2016). Anti-forensics of file system tunneling. *Digital Investigation*, 16, S109-S119.

## Author

Created for Digital Detectives Thesis - Phase 2A v2.0
Date: December 2024
Version: 2.0 (Location-Agnostic)
