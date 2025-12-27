# Phase 2A v2.0 Execution Guide

## Quick Start

```bash
# Navigate to notebook directory
cd "/Users/soni/Github/Digital-Detectives_Thesis/notebooks/Phase 2A - V2 Location Agnostic Features"

# Launch Jupyter
jupyter notebook 01_Location_Agnostic_Feature_Engineering_v2.ipynb

# In Jupyter: Kernel -> Restart & Run All
# Expected runtime: 3-5 minutes
```

## Pre-Execution Checklist

### 1. Verify Input Data
```bash
# Check that Phase 1 v2.0 output exists
ls -lh "/Users/soni/Github/Digital-Detectives_Thesis/data/processed/Phase 1 - V2 Data Cleaning/all_cases_combined_v2.csv"

# Should show:
# - File size: ~100-150 MB
# - Record count: 283,118 (use: wc -l <file>)
```

### 2. Verify Output Directory
```bash
# Output directory should exist (created by notebook)
ls -ld "/Users/soni/Github/Digital-Detectives_Thesis/data/processed/Phase 2A - V2 Location Agnostic Features"
```

### 3. System Requirements
- **RAM**: 4 GB minimum, 8 GB recommended
- **Python**: 3.8+ with pandas, numpy, tqdm
- **Disk space**: 500 MB free (for output + intermediate files)

## Cell-by-Cell Execution Guide

### Section 1: Setup & Load Data (Cells 1-4)
**Expected runtime**: 10-15 seconds

**What happens**:
- Imports libraries (pandas, numpy, datetime)
- Configures input/output paths
- Loads 283,118 records from CSV
- Parses timestamps

**Expected output**:
```
Dataset loaded successfully:
  Records: 283,118
  Columns: 48
  Timestomped events: [varies by case]
  Memory usage: ~200-250 MB
```

**Troubleshooting**:
- If "File not found": Verify Phase 1 v2.0 completed
- If "Memory error": Close other applications

---

### Section 2: Group 1 - File Type Features (Cells 5-11)
**Expected runtime**: 5-10 seconds

**Features created**: 6
1. is_executable
2. is_system_file
3. is_hidden_file
4. is_archive
5. filename_length
6. has_suspicious_extension

**Expected output**:
```
GROUP 1: FILE TYPE FEATURES (6 features)
[1/6] Creating is_executable...
  Total TRUE: ~65,000 (23%)
  ...
Group 1 complete: 6 file type features created
```

**Validation**:
- All 6 features should report 0 missing values
- Coverage should be 100%

---

### Section 3: Group 2 - Temporal Features (Cells 12-19)
**Expected runtime**: 2-3 MINUTES (longest section)

**Features created**: 6
7. event_frequency_per_file
8. event_frequency_per_case
9. events_in_1min_window (LONGEST - 2 min)
10. events_in_5min_window (LONGEST - 2 min)
11. time_since_previous_event_seconds
12. time_until_next_event_seconds

**Expected output**:
```
GROUP 2: TEMPORAL FEATURES (6 features) - CRITICAL
[9-10/24] Creating temporal clustering features...
Processing cases: 100%|██████████| 12/12 [02:30<00:00, 12.5s/it]

  events_in_1min_window (v1.0 importance: 21.31%):
    Mean: ~1,800
    ...
```

**Critical note**: Cells 9-10 process ALL 283,118 records:
- Progress bar shows case-by-case progress (12 cases total)
- DO NOT interrupt - will need to restart
- Expected: 2-3 minutes on modern laptop

**Troubleshooting**:
- If taking >5 minutes: Check CPU usage (should be 100%)
- If memory error: Reduce batch size in code

---

### Section 4: Group 3 - Cross-Artifact Features (Cells 20-24)
**Expected runtime**: 3-5 seconds

**Features created**: 3
13. source_confidence_score
14. has_logfile_evidence
15. has_usnjrnl_evidence

**Expected output**:
```
GROUP 3: CROSS-ARTIFACT FEATURES (3 features)
Source distribution:
  usnjrnl_only   : ~270,000 (95%)
  both           : ~10,000 (3%)
  logfile_only   : ~3,000 (1%)
```

---

### Section 5: Group 4 - UsnJrnl Pattern Features (Cells 25-29)
**Expected runtime**: 3-5 seconds

**Features created**: 3
16. usn_basic_info_change
17. usn_file_closed
18. usn_complete_manipulation_pattern

**Expected output**:
```
GROUP 4: USNJRNL PATTERN FEATURES (3 features)
[16/24] Creating usn_basic_info_change...
  Total TRUE: ~200,000 (70%)
  ...
```

---

### Section 6: Group 5 - Path Depth (Cells 30-32)
**Expected runtime**: 2-3 seconds

**Features created**: 1
19. path_depth (CORRECTED from v1.0)

**Expected output**:
```
GROUP 5: PATH DEPTH (1 feature) - CORRECTED
v1.0 bug: Used incorrect regex, returned 0 for all paths
v2.0 fix: Use raw string r'\\' to count actual backslashes

  Non-zero values: ~100,000 (35%)
  Mean depth: 5.2
  Median depth: 5
```

**Validation**:
- In v1.0, mean was 0.00 (BUG)
- In v2.0, mean should be 4-6 (FIXED)

---

### Section 7: Group 6 - Event-Time Comparison (Cells 33-35)
**Expected runtime**: 2-3 seconds

**Features created**: 1
20. event_vs_modified_after_days

**Expected output**:
```
GROUP 6: EVENT-TIME COMPARISON (1 feature)
  Non-null values: ~10,000 (3%)
  Mean: -500 days (negative = backdating)
```

---

### Section 8: Group 7 - NEW v2.0 Forensic Features (Cells 36-40)
**Expected runtime**: 30-45 seconds

**Features created**: 3 NEW
21. cross_artifact_validation_score (0-3 points)
22. timestamp_manipulation_pattern_score (0-3 points)
23. file_system_tunneling_confidence (0.0-1.0)

**Expected output**:
```
GROUP 7: NEW v2.0 FORENSIC FEATURES (3 features)
[21/24] Creating cross_artifact_validation_score...
  Distribution:
    Score 0: ~250,000 (88%)
    Score 1: ~20,000 (7%)
    Score 2: ~3,000 (1%)
    Score 3: ~10,000 (3%)
```

**Critical**: These features use `.apply()` which processes all 283,118 rows:
- Expected: 30-45 seconds
- Progress bar may not show (pandas limitation)

---

### Section 9: Validation & Summary (Cells 41-48)
**Expected runtime**: 5-10 seconds

**What happens**:
- Data integrity check (record count unchanged)
- Missing value analysis (should all be 0% or expected)
- **CRITICAL**: Validates NO location features created
- Feature coverage on timestomped events

**Expected output**:
```
CRITICAL VALIDATION: NO LOCATION FEATURES
Location features REMOVED from v1.0:
  in_temp_dir        : Not present (CORRECT)
  in_program_files   : Not present (CORRECT)
  in_windows_dir     : Not present (CORRECT)
  in_system32        : Not present (CORRECT)
  in_users_dir       : Not present (CORRECT)

Validation: PASSED - No location features present
```

**Critical check**: If ANY location feature shows "FOUND (ERROR!)", STOP and investigate.

---

### Section 10: Save Dataset (Cells 49-50)
**Expected runtime**: 30-60 seconds

**What happens**:
- Saves 283,118 records to CSV
- Reports file size and statistics

**Expected output**:
```
SAVING DATASET
Writing to: all_cases_combined_v2_phase2a.csv
This may take 30-60 seconds for 283,118 records...

Dataset saved successfully:
  File: all_cases_combined_v2_phase2a.csv
  Path: /Users/soni/.../Phase 2A - V2 Location Agnostic Features/...
  Size: 150-200 MB
  Records: 283,118
  Columns: 71
  New features: 23
```

**Validation**:
```bash
# Verify output file
ls -lh "/Users/soni/Github/Digital-Detectives_Thesis/data/processed/Phase 2A - V2 Location Agnostic Features/all_cases_combined_v2_phase2a.csv"

# Should show:
# - Size: 150-200 MB
# - 283,119 lines (283,118 data + 1 header)
wc -l <file>
```

---

### Section 11: Summary Report (Cells 51-52)
**Expected runtime**: 1-2 seconds

**What happens**:
- Comprehensive summary of all features created
- Comparison with v1.0
- Next steps

**Expected output**:
```
PHASE 2A v2.0 SUMMARY REPORT

FEATURES CREATED (23 total)
Group 1: File Type Features (6 features)
  - is_executable
  ...

FEATURES REMOVED (NO LOCATION FEATURES)
  - in_temp_dir (REMOVED)
  ...

KEY IMPROVEMENTS OVER v1.0
1. LOCATION INDEPENDENCE
2. TEMPORAL FOCUS
3. NEW FORENSIC FEATURES
4. BUG FIXES
```

---

## Post-Execution Validation

### 1. Verify Output File
```bash
# Check file exists and has correct size
ls -lh "/Users/soni/Github/Digital-Detectives_Thesis/data/processed/Phase 2A - V2 Location Agnostic Features/all_cases_combined_v2_phase2a.csv"

# Count records (should be 283,119 = 283,118 data + 1 header)
wc -l "/Users/soni/Github/Digital-Detectives_Thesis/data/processed/Phase 2A - V2 Location Agnostic Features/all_cases_combined_v2_phase2a.csv"
```

### 2. Verify Column Count
```python
import pandas as pd
df = pd.read_csv('.../all_cases_combined_v2_phase2a.csv')
print(f"Columns: {len(df.columns)}")  # Should be 71
print(f"Records: {len(df)}")          # Should be 283,118
```

### 3. Verify NO Location Features
```python
location_features = ['in_temp_dir', 'in_program_files', 'in_windows_dir',
                     'in_system32', 'in_users_dir']
for feat in location_features:
    if feat in df.columns:
        print(f"ERROR: {feat} found in output!")
    else:
        print(f"OK: {feat} not present")
```

### 4. Verify New Features Present
```python
new_features = [
    'cross_artifact_validation_score',
    'timestamp_manipulation_pattern_score',
    'file_system_tunneling_confidence'
]
for feat in new_features:
    if feat in df.columns:
        print(f"OK: {feat} present")
    else:
        print(f"ERROR: {feat} missing!")
```

---

## Expected Total Runtime

| Section | Runtime | Progress Indicator |
|---------|---------|-------------------|
| Setup (1-4) | 10-15 sec | Loading bar |
| Group 1 (5-11) | 5-10 sec | Print statements |
| Group 2 (12-19) | **2-3 min** | **tqdm progress bar** |
| Group 3 (20-24) | 3-5 sec | Print statements |
| Group 4 (25-29) | 3-5 sec | Print statements |
| Group 5 (30-32) | 2-3 sec | Print statements |
| Group 6 (33-35) | 2-3 sec | Print statements |
| Group 7 (36-40) | 30-45 sec | (No progress bar) |
| Validation (41-48) | 5-10 sec | Print statements |
| Save (49-50) | 30-60 sec | Print statements |
| Summary (51-52) | 1-2 sec | Print statements |
| **TOTAL** | **3-5 min** | - |

---

## Troubleshooting

### Issue: "File not found" error in Cell 3
**Cause**: Phase 1 v2.0 output missing
**Solution**:
```bash
# Verify Phase 1 output exists
ls "/Users/soni/Github/Digital-Detectives_Thesis/data/processed/Phase 1 - V2 Data Cleaning/all_cases_combined_v2.csv"

# If missing, run Phase 1 v2.0 notebook first
```

### Issue: Memory error during temporal clustering
**Cause**: Insufficient RAM (need 4+ GB)
**Solution**:
1. Close other applications
2. Restart Jupyter kernel
3. If persists, modify cells 9-10 to process in batches:
   ```python
   # Instead of processing all cases at once,
   # process 3 cases at a time
   for case_batch in [cases[:3], cases[3:6], cases[6:9], cases[9:]]:
       # process batch
   ```

### Issue: Cells 9-10 taking >5 minutes
**Cause**: Slow CPU or background processes
**Solution**:
1. Check CPU usage (should be near 100%)
2. Close background applications
3. If CPU is low, check for thermal throttling

### Issue: "Location features found" validation error
**Cause**: Code modification error
**Solution**:
1. Re-download notebook from repository
2. DO NOT modify feature creation code
3. If intentional, update validation section

### Issue: Output file wrong size (<100 MB or >300 MB)
**Cause**: Data corruption or incomplete execution
**Solution**:
1. Verify all cells executed (check cell numbers)
2. Re-run "Save Dataset" section (cells 49-50)
3. If persists, restart kernel and run all

### Issue: Columns ≠ 71
**Cause**: Feature creation failed or duplicates
**Solution**:
1. Check validation output (cell 41)
2. Look for error messages in feature creation cells
3. Verify no duplicate column names:
   ```python
   df.columns[df.columns.duplicated()].tolist()
   ```

---

## Performance Optimization Tips

### For Faster Execution (if needed)
1. **Use smaller dtype**: Change int64 to int32 where appropriate
2. **Categorical encoding**: Convert string columns to categorical
3. **Parallel processing**: Use multiprocessing for temporal clustering
4. **Reduce batch size**: Process fewer cases at once

### For Lower Memory Usage (if needed)
1. **Process in chunks**: Load data in chunks instead of all at once
2. **Delete intermediate**: Delete temporary columns after use
3. **Garbage collection**: Add `import gc; gc.collect()` after heavy operations

---

## Next Steps After Execution

Once Phase 2A v2.0 completes successfully:

1. **Verify output**: Use validation commands above
2. **Document results**: Save execution time and any issues encountered
3. **Proceed to Phase 3**: Model Training v2.0
   - Input: `all_cases_combined_v2_phase2a.csv` (71 columns)
   - Train Random Forest on 23 location-agnostic features
   - Evaluate on internal + external datasets

---

## Quick Reference: Expected Values

| Metric | Expected Value | Validation Command |
|--------|---------------|-------------------|
| Input records | 283,118 | `wc -l <input>` |
| Output records | 283,118 | `wc -l <output>` |
| Input columns | 48 | `head -1 <input> | tr ',' '\n' | wc -l` |
| Output columns | 71 | `head -1 <output> | tr ',' '\n' | wc -l` |
| New features | 23 | 71 - 48 = 23 |
| Location features | 0 | Manual check |
| Output file size | 150-200 MB | `ls -lh <output>` |
| Execution time | 3-5 min | Manual timing |

---

## Contact

For issues or questions:
1. Check troubleshooting section above
2. Review README.md for feature specifications
3. Verify input data from Phase 1 v2.0
4. Check Jupyter kernel logs for detailed errors

**Critical reminder**: This is v2.0 designed to fix location-based overfitting. Do NOT add location features back in!
