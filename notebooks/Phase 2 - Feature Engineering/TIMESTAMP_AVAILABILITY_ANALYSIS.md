# Timestamp Availability Analysis - Critical Findings

## 🔍 Problem Statement

You correctly identified a major constraint for Phase 2 feature engineering:

**Original Phase 2 Plan Issue:**
- Many proposed features require comparing different timestamp types (e.g., "creation_after_modified", "creation_in_future")
- These features assume we have ALL four NTFS timestamps available for comparison
- **Reality**: We only have timestamps that were ACTUALLY MANIPULATED in the lf_detail field

---

## 📊 What Timestamp Data Do We Actually Have?

### Available Timestamps:

#### 1. **eventtime_dt** (ALWAYS available)
- **Source**: From both LogFile and UsnJrnl
- **Meaning**: When the timestamp manipulation EVENT occurred (logged by the journal)
- **Coverage**: 100% of records (154,550 records)
- **Data type**: Datetime

#### 2. **Parsed LogFile Timestamps** (ONLY for LogFile records)
From `lf_detail` parsing, we have BEFORE/AFTER pairs for timestamps that were **actually manipulated**:

| Timestamp Pair | Availability | Notes |
|----------------|--------------|-------|
| `lf_creation_time_before/after` | ~473 records (0.3%) | Only when CreationTime was manipulated |
| `lf_modified_time_before/after` | ~2,919 records (1.9%) | Only when ModifiedTime was manipulated |
| `lf_accessed_time_before/after` | ~292 records (0.2%) | Only when AccessedTime was manipulated |
| `lf_mft_modified_time_before/after` | ~1,469 records (0.9%) | Only when MFTModifiedTime was manipulated |

**Critical Finding:**
- Each record has ONLY the timestamp(s) that were manipulated
- Example: If only ModifiedTime was changed, we have ONLY modified_time_before/after
- We do NOT have the other three timestamps (Creation, Accessed, MFTModified) for that file
- We CANNOT compare CreationTime vs ModifiedTime if only one was manipulated

#### 3. **UsnJrnl Records** (151,448 records = 98% of dataset)
- **Source**: `usnjrnl_only` records
- **Available timestamps**: ONLY `eventtime_dt` (the event time)
- **No parsed before/after timestamps**: UsnJrnl doesn't provide the actual timestamp values, only the fact that a change occurred

---

## ❌ Features We CANNOT Create (Cross-Timestamp Comparisons)

### From Original Phase 2 Plan - Group 1 (Temporal Anomaly Features):

**Cannot create these features:**
1. ❌ `creation_after_modified` - Requires BOTH creation AND modified timestamps
2. ❌ `creation_in_future` - Can only compare to eventtime, not to modified time
3. ❌ `modified_after_accessed` - Requires BOTH modified AND accessed timestamps
4. ❌ `timestamp_ordering_violation_count` - Requires ALL four timestamps
5. ❌ `impossible_timestamp_sequence` - Requires multiple timestamps

**Why?**
- If a file had only ModifiedTime manipulated, we don't have CreationTime/AccessedTime/MFTModifiedTime to compare against
- Cross-timestamp comparisons only work if MULTIPLE timestamps were manipulated on the SAME file in the SAME event

---

## ✅ Features We CAN Create (Limited But High-Value)

### Category 1: Event Time vs Manipulated Timestamp Comparisons

For each record with LogFile data, we can compare `eventtime_dt` (when manipulation occurred) against the manipulated timestamp:

1. **event_vs_creation_after_days**
   - Formula: `(eventtime_dt - lf_creation_time_after).days`
   - Meaning: How far in the past/future is the manipulated creation time relative to when the manipulation occurred?
   - Coverage: 473 records with lf_creation_time_after

2. **event_vs_modified_after_days**
   - Formula: `(eventtime_dt - lf_modified_time_after).days`
   - Coverage: 2,919 records

3. **event_vs_accessed_after_days**
   - Formula: `(eventtime_dt - lf_accessed_time_after).days`
   - Coverage: 292 records

4. **event_vs_mft_modified_after_days**
   - Formula: `(eventtime_dt - lf_mft_modified_time_after).days`
   - Coverage: 1,469 records

**Rationale**: Large discrepancies indicate timestomping to distant past/future.

---

### Category 2: Delta-Based Features (Already Created in Phase 1B)

✅ These are ALREADY in the dataset:

1. **creation_time_delta_days** - How many days the creation time was changed (before - after)
2. **modified_time_delta_days** - How many days the modified time was changed
3. **accessed_time_delta_days** - How many days the accessed time was changed
4. **mft_modified_time_delta_days** - How many days the MFT modified time was changed

5. **creation_time_changed_to_past** (boolean)
6. **modified_time_changed_to_past** (boolean)
7. **accessed_time_changed_to_past** (boolean)
8. **mft_modified_time_changed_to_past** (boolean)

**These are HIGH-VALUE features** - they directly capture the manipulation behavior.

---

### Category 3: Manipulation Signatures (Already in Dataset)

✅ Already created in Phase 1B:

1. **zero_in_nanoseconds** - Indicator of SetFileTime() API usage (tool signature)
2. **copied_from_file** - Indicator of SetMACE-like behavior (copied from another file)

**Research backing**: Oh et al. (2024) identified these as strong indicators of tool-based manipulation.

---

### Category 4: Cross-Artifact Correlation Features (High Priority)

✅ Can create without timestamp comparisons:

1. **source_confidence_score**
   - Mapping: `{'both': 2, 'logfile_only': 1, 'usnjrnl_only': 1}`
   - Research-backed: Cross-artifact agreement = HIGH confidence

2. **has_logfile_data** (boolean)
   - `source in ['both', 'logfile_only']`
   - Indicates presence of LogFile Time Reversal event

3. **has_usnjrnl_data** (boolean)
   - `source in ['both', 'usnjrnl_only']`
   - Indicates presence of UsnJrnl BASIC_INFO_CHANGE pattern

4. **usn_basic_info_change** (boolean)
   - Parse from `usn_event_info` column
   - Pattern: contains "Basic_Info_Change"

5. **usn_close_event** (boolean)
   - Parse from `usn_event_info` column
   - Pattern: contains "File_Close"

6. **usn_manipulation_pattern** (boolean)
   - `usn_basic_info_change AND usn_close_event`
   - Research-backed complete detection pattern

---

### Category 5: Behavioral Pattern Features (Medium Priority)

1. **event_frequency_per_file**
   - Count of events per unique `merge_key` (filepath + filename)
   - Rationale: Multiple timestamp manipulations on same file = suspicious

2. **event_frequency_per_case**
   - Count of events per `case_id`
   - Rationale: Batch timestomping creates spikes

3. **time_since_previous_event_seconds**
   - Time delta between consecutive events (sorted by eventtime_dt)
   - Rationale: Rapid-fire timestamp changes indicate automated tools

4. **events_in_5min_window**
   - Count of events within ±5 minutes of current event
   - Rationale: Temporal clustering = batch operations

---

### Category 6: File-Level Features (High Priority for APT Detection)

These require NO timestamp comparisons, only file path/attribute analysis:

1. **in_system32** - `'\\Windows\\System32\\' in filepath OR '\\Windows\\SysWOW64\\' in filepath`
2. **in_windows_dir** - `'\\Windows\\' in filepath`
3. **in_temp_dir** - `'\\Temp\\' in filepath OR '\\AppData\\Local\\Temp\\' in filepath`
4. **in_program_files** - `'\\Program Files' in filepath`
5. **in_users_dir** - `'\\Users\\' in filepath`

6. **path_depth** - Count of `\\` in filepath
7. **filename_length** - Length of filename

8. **is_executable** - Extension in ['.exe', '.dll', '.sys', '.bat', '.ps1', '.vbs']
9. **is_document** - Extension in ['.doc', '.docx', '.pdf', '.xls', '.xlsx']
10. **is_hidden** - Parse from `usn_file_attribute` column (contains 'Hidden')
11. **is_system** - Parse from `usn_file_attribute` column (contains 'System')

**Research backing**: Oh et al. (2024) Table 8 - APT malware targets specific locations (System32, Windows, Temp)

---

## 🎯 Revised Feature Engineering Strategy (High Value, Low Noise)

### Priority 1: Features with DIRECT Detection Value (10-15 features)

**Already in dataset (Phase 1B):**
1. ✅ `creation_time_delta_days`
2. ✅ `modified_time_delta_days`
3. ✅ `accessed_time_delta_days`
4. ✅ `mft_modified_time_delta_days`
5. ✅ `creation_time_changed_to_past`
6. ✅ `modified_time_changed_to_past`
7. ✅ `accessed_time_changed_to_past`
8. ✅ `mft_modified_time_changed_to_past`
9. ✅ `zero_in_nanoseconds`
10. ✅ `copied_from_file`

**Need to create:**
11. `source_confidence_score` (Cross-artifact)
12. `usn_manipulation_pattern` (Cross-artifact)
13. `event_vs_modified_after_days` (Temporal - most coverage: 2,919 records)

---

### Priority 2: Behavioral Context Features (5-8 features)

14. `event_frequency_per_file`
15. `event_frequency_per_case`
16. `time_since_previous_event_seconds`
17. `events_in_5min_window`

---

### Priority 3: File-Level Features (8-12 features)

18. `in_system32`
19. `in_windows_dir`
20. `in_temp_dir`
21. `path_depth`
22. `filename_length`
23. `is_executable`
24. `is_hidden`
25. `is_system`

---

## 📉 What We're NOT Including (To Reduce Noise)

### Removed from original plan:

1. ❌ Cross-timestamp comparisons (impossible with our data)
2. ❌ Z-score normalization (adds complexity, not necessarily better than raw deltas)
3. ❌ Statistical aggregations (per-case means/stds) - adds leakage risk
4. ❌ Future timestamp detection (can use eventtime_dt comparison instead)
5. ❌ Complex timestamp ordering violations (need multiple timestamps)

---

## 📊 Expected Final Feature Count

**Total: ~25-30 features** (down from 61-78 in original plan)

**Breakdown:**
- Priority 1 (Direct detection): 13 features ← **MOST IMPORTANT**
- Priority 2 (Behavioral): 5-8 features
- Priority 3 (File-level): 8-12 features

**Benefit:**
- ✅ Focused on features we can ACTUALLY calculate
- ✅ Research-backed features only
- ✅ No noisy/speculative features
- ✅ Better signal-to-noise ratio for ML model

---

## 🚀 Recommendation: Proceed with Revised Plan?

**Suggested approach:**
1. Create Priority 1 features first (13 features) - these have DIRECT detection value
2. Train baseline Random Forest with Priority 1 only
3. Add Priority 2 features (behavioral context)
4. Add Priority 3 features (file-level)
5. Use feature importance analysis to identify which features actually help

**This iterative approach ensures:**
- We focus on high-value features first
- We validate each feature group's contribution
- We avoid feature bloat and overfitting
- We maintain forensic interpretability

---

## ✅ Next Steps

1. **Update PHASE_2_PLAN.md** with revised feature list (25-30 features)
2. **Create Phase 2A notebook** to implement Priority 1 features
3. **Validate feature quality** on timestomped vs benign records
4. **Proceed to baseline model** once Priority 1 features are validated

Let me know if this revised strategy makes sense, and I'll update the Phase 2 plan accordingly!
