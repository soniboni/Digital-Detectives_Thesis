# Phase 2A Implementation Guide

## 📋 Quick Reference

**Notebook**: `02A_File_Level_and_Behavioral_Features.ipynb`

**Purpose**: Create 18 file-level and behavioral features with 100% coverage to detect timestamp manipulation in the 94% UsnJrnl-only events.

---

## 🎯 What This Notebook Does

### Input:
- `all_cases_combined_clean.csv` (154,550 records, 38 columns, Phase 1B output)

### Output:
- `all_cases_combined_with_phase2a_features.csv` (154,550 records, 56 columns)
- 18 new features added
- 2 visualization plots

### Processing Time:
- Estimated: 2-5 minutes (temporal clustering features are computation-heavy)

---

## 📊 Features Created (18 Total)

### Group 1A: Location-Based (6 features)

| Feature | Type | Formula | Coverage | Rationale |
|---------|------|---------|----------|-----------|
| `in_system32` | Boolean | System32 OR SysWOW64 in path | 100% | APT targets System32 (50%+ from research) |
| `in_windows_dir` | Boolean | Windows in path | 100% | System directory targeting |
| `in_temp_dir` | Boolean | Temp OR AppData/Local/Temp in path | 100% | Malware staging location |
| `in_program_files` | Boolean | Program Files in path | 100% | Application vs system distinction |
| `in_users_dir` | Boolean | Users in path | 100% | User vs system risk profile |
| `path_depth` | Integer | Count of backslashes in path | 100% | Hiding behavior (deep nesting) |

---

### Group 1B: File Type & Attributes (6 features)

| Feature | Type | Formula | Coverage | Rationale |
|---------|------|---------|----------|-----------|
| `is_executable` | Boolean | Extension in [.exe, .dll, .sys, .bat, .ps1, etc.] | 100% | Executables = primary timestomping targets |
| `is_system_file` | Boolean | 'System' in usn_file_attribute | 98% | System files = higher risk |
| `is_hidden_file` | Boolean | 'Hidden' in usn_file_attribute | 98% | Hidden + timestomped = anti-forensics |
| `is_archive` | Boolean | 'Archive' in usn_file_attribute | 98% | Archive attribute analysis |
| `filename_length` | Integer | Length of filename | 100% | Suspicious short/long names |
| `has_suspicious_extension` | Boolean | Extension in [.tmp, .log, .bak, etc.] | 100% | Temp files being manipulated |

---

### Group 1C: Temporal Behavioral (6 features)

| Feature | Type | Formula | Coverage | Rationale |
|---------|------|---------|----------|-----------|
| `event_frequency_per_file` | Integer | Count events per merge_key | 100% | Multiple manipulations on same file |
| `event_frequency_per_case` | Integer | Count events per case_id | 100% | Batch operation detection |
| `events_in_1min_window` | Integer | Events within ±30 seconds | 100% | Rapid batch operations |
| `events_in_5min_window` | Integer | Events within ±2.5 minutes | 100% | Broader temporal clustering |
| `time_since_previous_event_seconds` | Float | Seconds since previous event | 100% | Rapid-fire detection (<1 sec) |
| `time_until_next_event_seconds` | Float | Seconds until next event | 100% | Symmetric temporal context |

---

## 🔧 Implementation Details

### Key Design Decisions:

**1. 100% Coverage Priority**
- All features work on ALL 252 timestomped events
- No missing data except intentional cases (first/last events in temporal features)
- Missing values handled explicitly with fill strategies

**2. Case-Based Processing**
- Temporal features calculated PER CASE (events in different cases don't cluster)
- Sorting by case_id and eventtime_dt for efficient window calculations

**3. Missing Data Handling**
```python
# Filepath-based features: empty string = False
df['filepath'].fillna('').astype(str)

# File attribute features: empty = False (LogFile-only records)
df['usn_file_attribute'].fillna('').astype(str)

# Temporal features: first/last events filled with large value (999999)
df['time_since_previous_event_seconds'].fillna(999999)
```

**4. Boolean vs Continuous**
- Location/file type features: Boolean (TRUE/FALSE)
- Temporal/numeric features: Integer/Float (actual values)
- Enables both decision trees and statistical models

---

## 🚀 How to Run

### Step 1: Open Notebook in Jupyter
```bash
cd /Users/soni/Github/Digital-Detectives_Thesis
jupyter notebook "notebooks/Phase 2 - Feature Engineering/02A_File_Level_and_Behavioral_Features.ipynb"
```

### Step 2: Run All Cells
- Click "Cell" → "Run All" OR
- Shift+Enter through each cell

### Step 3: Verify Output
Check for:
- ✅ 154,550 records (no change)
- ✅ 252 timestomped events (no change)
- ✅ 56 total columns (38 + 18)
- ✅ Output CSV saved
- ✅ 2 plots generated

---

## 📊 Expected Results

### Feature Distributions (Timestomped vs Benign):

**Location Features:**
- `in_system32`: Timestomped likely HIGHER than benign (APT targeting)
- `in_users_dir`: Benign likely HIGHER (user activity)
- `in_temp_dir`: Mixed (both use temp)

**File Type Features:**
- `is_executable`: Timestomped likely HIGHER (executables targeted)
- `is_hidden_file`: Timestomped likely HIGHER (anti-forensics)

**Temporal Features:**
- `events_in_1min_window`: Timestomped likely HIGHER (batch operations)
- `time_since_previous_event_seconds`: Timestomped likely LOWER (rapid-fire)
- `event_frequency_per_file`: Timestomped likely HIGHER (multiple manipulations)

---

## ✅ Validation Checklist

After running the notebook, verify:

- [ ] No errors or warnings (except expected pandas dtype warnings)
- [ ] 154,550 records unchanged
- [ ] 252 timestomped events unchanged
- [ ] 18 new features created
- [ ] All features have <1% missing data (except temporal edge cases)
- [ ] Output CSV saved successfully
- [ ] Plots generated and saved
- [ ] Feature distributions look reasonable (no all-zeros, no all-same values)

---

## 🔍 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution**: Ensure Jupyter is using correct Python environment with pandas installed
```bash
python3 -m pip install pandas numpy matplotlib seaborn
```

### Issue: Temporal clustering features taking too long (>10 minutes)
**Solution**: This is expected for 154,550 records. Coffee break recommended ☕
- Processes ~12,000-13,000 events per case
- 12 cases total
- Should complete in 2-5 minutes on modern hardware

### Issue: Memory warning or system slowdown
**Solution**: The dataset is ~80MB. Should be fine on systems with 8GB+ RAM. If issues occur:
```python
# Add at top of notebook to reduce memory:
df = pd.read_csv(..., dtype={'usn_usn': 'float32', 'lf_lsn': 'float32'})
```

---

## 📈 Next Steps After Phase 2A

Once Phase 2A completes successfully:

1. **Review Feature Distributions**
   - Check the plots: do timestomped vs benign show separation?
   - Identify which features have strongest signals

2. **Proceed to Phase 2B**
   - Add 7 more features (cross-artifact, UsnJrnl patterns, event-time comparisons)
   - Total will be 25 new features

3. **Optional: Quick Model Test**
   - Train a simple Random Forest with just these 18 features
   - Check if they alone can detect timestomping
   - Validates feature quality before adding more

---

## 📚 Research Justification

Every feature is backed by **Oh et al. (2024) "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"**

**Key Citations:**

**Location Features** → Table 8 (Page 14):
- "APT malware samples show distinctive location patterns"
- "50%+ target System32 directory"
- "Windows system directories commonly manipulated"

**File Type Features** → Table 8 (Page 14):
- "Executable files (.exe, .dll, .sys) are primary targets"
- "System files exhibit higher manipulation rates"

**Temporal Features** → Section V.E (Page 11-12):
- "Automated tools create temporal clustering patterns"
- "BASIC_INFO_CHANGE + CLOSE occur within short time (0-1 second)"
- "Batch operations create detectable event frequency spikes"

---

## 💡 Pro Tips

1. **Save Intermediate Results**: The notebook auto-saves after each group. If interrupted, you won't lose all progress.

2. **Check Case-Level Stats**: The temporal clustering prints per-case statistics. Review these to understand case-specific patterns.

3. **Feature Importance Preview**: After Phase 2C, you'll run feature importance analysis. These 18 features should score HIGH importance.

4. **Documentation**: The notebook includes extensive markdown. Read the rationale sections to understand WHY each feature matters.

5. **Plot Interpretation**:
   - Boolean plots: Look for RED (timestomped) bars being notably different from BLUE (benign)
   - Continuous plots: Look for distribution separation (different means, different shapes)

---

## ✅ Success Criteria

Phase 2A is successful if:

1. ✅ All 18 features created without errors
2. ✅ No data loss (252 timestomped events preserved)
3. ✅ Feature distributions show SOME separation between timestomped and benign
4. ✅ Output CSV is valid and loadable
5. ✅ Plots show clear visualization of feature patterns

**If all criteria met → Proceed to Phase 2B**

---

## 📞 Support

If issues arise:
1. Check the error message carefully
2. Verify input file exists: `data/processed/Phase 1B - Column Cleanup/all_cases_combined_clean.csv`
3. Ensure sufficient disk space (output ~85MB)
4. Review the troubleshooting section above

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Notebook**: 02A_File_Level_and_Behavioral_Features.ipynb
