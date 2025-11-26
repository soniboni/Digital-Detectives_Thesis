# Phase 2: Feature Engineering - REVISED Plan (Data-Driven Approach)

## 📋 Document Purpose

This is a **REVISED** feature engineering plan based on comprehensive data analysis that revealed critical insights about the dataset composition and feature coverage. The original plan prioritized timestamp delta features, but data analysis showed that **94% of timestomped events lack LogFile data**, making file-level and behavioral features far more critical.

---

## 🔍 Key Data Insights That Drove This Revision

### Critical Finding #1: Source Distribution Asymmetry

**Timestomped Events (252 total):**
- UsnJrnl-only: 238 (94.4%) ← **Cannot use LogFile timestamp features**
- LogFile-only: 8 (3.2%)
- Both artifacts: 6 (2.4%)

**Benign Events (154,298 total):**
- UsnJrnl-only: 151,210 (98.0%)
- Both artifacts: 2,858 (1.9%)
- LogFile-only: 230 (0.1%)

**Implication:**
- ⚠️ **94% of timestomped events have NO parsed timestamp deltas** (no lf_detail)
- ⚠️ LogFile features (creation_time_delta_days, etc.) only work on 14 timestomped events (5.6%)
- ✅ **File-level and behavioral features work on ALL 252 events (100%)**

**Reasoning:**
Prioritizing LogFile delta features (original plan) would leave 94% of timestomped events with minimal detection signals. We must prioritize features that provide context for UsnJrnl-only events.

---

### Critical Finding #2: Event Pattern Overlap

**UsnJrnl Event Patterns:**

| Pattern | Timestomped Count | Benign Count | Overlap Ratio |
|---------|-------------------|--------------|---------------|
| `Basic_Info_Changed / File_Closed` | 9 | 23,782 | 1:2,642 |
| `Basic_Info_Changed` | 6 | 25,329 | 1:4,221 |
| `File_Created / Basic_Info_Changed / Data_Added / Data_Overwritten / File_Closed` | 229 | 3,819 | 1:17 |

**LogFile Event Types:**

| Event Type | Timestomped Count | Benign Count | Overlap Ratio |
|------------|-------------------|--------------|---------------|
| `Time Reversal Event` | 14 | 2,800 | 1:200 |

**Implication:**
- ⚠️ **Event patterns alone CANNOT distinguish timestomping from benign behavior**
- ⚠️ Massive overlap - benign events vastly outnumber timestomped events with same patterns
- ✅ **We need contextual features** to separate malicious from benign `Basic_Info_Changed` events

**Reasoning:**
Since event patterns overlap heavily, we need **file location, file type, temporal clustering, and behavioral patterns** to distinguish APT timestomping from legitimate system operations. This is supported by Oh et al. (2024) Table 8, which shows APT malware has distinctive behavioral signatures (System32 targeting, executable files, batch operations).

---

### Critical Finding #3: Tool Signature Coverage

**zero_in_nanoseconds = True:**
- Timestomped: 3 / 252 (1.2%)
- Benign: 1,295 / 154,298 (0.8%)

**copied_from_file = True:**
- Limited coverage in timestomped events

**Implication:**
- ⚠️ Tool signatures have **very low coverage** (only 1.2% of timestomped events)
- ✅ When present, they are valuable (high precision), but cannot be relied upon as primary detection features

**Reasoning:**
Tool signatures should be included but cannot be prioritized. Most timestomping in this dataset does NOT leave these signatures.

---

## 🎯 REVISED Feature Engineering Strategy

### Design Philosophy:

1. **Coverage First**: Prioritize features that work on ALL 252 timestomped events (100% coverage)
2. **Contextual Separation**: Use file-level and behavioral features to separate overlapping event patterns
3. **Research-Backed**: Every feature justified by Oh et al. (2024) APT behavioral analysis
4. **Handle Missing Data**: Model must work when LogFile features are unavailable (94% of cases)
5. **Feature Interactions**: Design features that combine well (Random Forest learns interaction patterns)

---

## 📊 REVISED Feature Groups (Priority Order)

### **Priority 1: File-Level & Behavioral Features (15-18 features)** ✅ START HERE

**Rationale:**
- ✅ Work on 100% of events (both LogFile and UsnJrnl)
- ✅ Capture APT behavioral patterns from research (Oh et al., 2024, Table 8)
- ✅ Provide context to separate overlapping event patterns
- ✅ Most critical for detecting the 94% UsnJrnl-only timestomped events

**Research Support:**
Oh et al. (2024) Table 8 shows APT malware has distinctive patterns:
- 50%+ target System32 directory
- Focus on executable files (.exe, .dll, .sys)
- Batch operations (multiple files in short time)
- Specific locations: Windows, Temp, System32

---

#### Feature Group 1A: Location-Based Features (6 features)

```python
# Feature: in_system32
# Formula: ('\\Windows\\System32\\' in filepath) OR ('\\Windows\\SysWOW64\\' in filepath)
# Why: Research shows 50%+ of APT malware targets System32
# Coverage: 100% of records
# Research: Oh et al. (2024) Table 8 - APT malware location analysis
```

```python
# Feature: in_windows_dir
# Formula: '\\Windows\\' in filepath
# Why: APT malware often targets Windows system directories
# Coverage: 100% of records
# Research: Oh et al. (2024) Table 8
```

```python
# Feature: in_temp_dir
# Formula: ('\\Temp\\' in filepath) OR ('\\AppData\\Local\\Temp\\' in filepath)
# Why: APT malware stages files in temp directories before timestomping
# Coverage: 100% of records
# Research: Oh et al. (2024) Table 8
```

```python
# Feature: in_program_files
# Formula: '\\Program Files' in filepath
# Why: Distinguish between system vs application files
# Coverage: 100% of records
```

```python
# Feature: in_users_dir
# Formula: '\\Users\\' in filepath
# Why: User directories have different risk profile than system directories
# Coverage: 100% of records
```

```python
# Feature: path_depth
# Formula: filepath.count('\\')
# Why: Deeply nested paths may indicate hiding behavior; shallow paths (system root) have different risk
# Coverage: 100% of records (NaN for records without filepath → fill with 0)
```

---

#### Feature Group 1B: File Type & Attribute Features (6 features)

```python
# Feature: is_executable
# Formula: filename extension in ['.exe', '.dll', '.sys', '.bat', '.ps1', '.vbs', '.com', '.scr']
# Why: Executables are primary targets for timestomping (APT malware)
# Coverage: 100% of records
# Research: Oh et al. (2024) Table 8 - APT malware targets executables
```

```python
# Feature: is_system_file
# Formula: 'System' in usn_file_attribute
# Why: System files being manipulated is higher risk than user files
# Coverage: UsnJrnl records only (~98% of dataset)
# Missing data handling: Fill False for LogFile-only records
```

```python
# Feature: is_hidden_file
# Formula: 'Hidden' in usn_file_attribute
# Why: Hidden files + timestamp manipulation = potential anti-forensics
# Coverage: UsnJrnl records only (~98% of dataset)
# Missing data handling: Fill False for LogFile-only records
```

```python
# Feature: is_archive
# Formula: 'Archive' in usn_file_attribute
# Why: Archive attribute analysis (normal vs suspicious)
# Coverage: UsnJrnl records only (~98% of dataset)
```

```python
# Feature: filename_length
# Formula: len(filename)
# Why: Very short or very long filenames may be suspicious
# Coverage: 100% of records
```

```python
# Feature: has_suspicious_extension
# Formula: extension in ['.tmp', '.log', '.bak', '.old', '.$$$']
# Why: Temporary files being timestomped may indicate cleanup operations
# Coverage: 100% of records
```

---

#### Feature Group 1C: Temporal Behavioral Features (6 features)

```python
# Feature: event_frequency_per_file
# Formula: df.groupby('merge_key').size()
# Why: Multiple timestamp manipulations on SAME file = highly suspicious
# Coverage: 100% of records
# Rationale: Benign operations typically touch file once; malware may manipulate multiple timestamps
```

```python
# Feature: event_frequency_per_case
# Formula: df.groupby('case_id').size()
# Why: Batch timestomping creates spikes in event counts
# Coverage: 100% of records
# Rationale: APT malware often manipulates many files at once
```

```python
# Feature: events_in_1min_window
# Formula: Count events within ±30 seconds of current event (by eventtime_dt)
# Why: Temporal clustering indicates automated batch operations
# Coverage: 100% of records
# Research: Automated tools (SetMACE, NewFileTime) process files rapidly
```

```python
# Feature: events_in_5min_window
# Formula: Count events within ±2.5 minutes of current event
# Why: Broader temporal clustering detection (less sensitive than 1min)
# Coverage: 100% of records
```

```python
# Feature: time_since_previous_event_seconds
# Formula: (eventtime_dt - previous_eventtime_dt).total_seconds() (sorted by eventtime_dt per case)
# Why: Rapid-fire timestamp changes (<1 second apart) indicate automated tools
# Coverage: 100% of records (first event in case = NaN → fill with large value)
```

```python
# Feature: time_until_next_event_seconds
# Formula: (next_eventtime_dt - eventtime_dt).total_seconds()
# Why: Symmetric temporal context (before and after)
# Coverage: 100% of records (last event in case = NaN → fill with large value)
```

---

### **Priority 2: Cross-Artifact Correlation Features (3 features)**

**Rationale:**
- ✅ Research-backed (Oh et al., 2024) - cross-artifact agreement = HIGH confidence
- ✅ Work on 100% of records
- ✅ Provide confidence weighting for predictions

---

```python
# Feature: source_confidence_score
# Formula: {'both': 2, 'logfile_only': 1, 'usnjrnl_only': 1}
# Why: Cross-artifact validation increases detection confidence
# Coverage: 100% of records
# Research: Oh et al. (2024) emphasizes cross-artifact correlation for HIGH confidence detection
```

```python
# Feature: has_logfile_evidence
# Formula: source in ['both', 'logfile_only']
# Why: Presence of LogFile Time Reversal event = direct evidence
# Coverage: 100% of records
```

```python
# Feature: has_usnjrnl_evidence
# Formula: source in ['both', 'usnjrnl_only']
# Why: Presence of UsnJrnl BASIC_INFO_CHANGE pattern
# Coverage: 100% of records
```

---

### **Priority 3: UsnJrnl Pattern Features (3 features)**

**Rationale:**
- ✅ Parse structured information from usn_event_info column
- ✅ Research-backed detection pattern (Oh et al., 2024, Section V.E.1)
- ✅ 98% coverage (UsnJrnl records)

---

```python
# Feature: usn_basic_info_change
# Formula: 'Basic_Info_Change' in usn_event_info
# Why: Core detection pattern from research
# Coverage: 98% of records (UsnJrnl records)
# Research: Oh et al. (2024) Section V.E.1 - timestamp manipulation creates BASIC_INFO_CHANGE
```

```python
# Feature: usn_file_closed
# Formula: 'File_Close' in usn_event_info
# Why: Complete detection pattern (BASIC_INFO_CHANGE + CLOSE)
# Coverage: 98% of records
# Research: Oh et al. (2024) Section V.E.1
```

```python
# Feature: usn_complete_manipulation_pattern
# Formula: usn_basic_info_change AND usn_file_closed
# Why: Research-backed complete detection pattern
# Coverage: 98% of records
# Note: This pattern appears in BOTH timestomped and benign, but provides signal for interaction
```

---

### **Priority 4: Timestamp Delta Features (8 features)** - Already Exist from Phase 1B

**Rationale:**
- ✅ Already created in Phase 1B
- ✅ Direct evidence of manipulation when available
- ⚠️ Low coverage (only 6% of timestomped events have LogFile data)
- ✅ High signal when present

**Existing Features (from Phase 1B):**
1. `creation_time_delta_days` - Coverage: 473 records
2. `creation_time_changed_to_past` - Coverage: 473 records
3. `modified_time_delta_days` - Coverage: 2,919 records
4. `modified_time_changed_to_past` - Coverage: 2,919 records
5. `accessed_time_delta_days` - Coverage: 292 records
6. `accessed_time_changed_to_past` - Coverage: 292 records
7. `mft_modified_time_delta_days` - Coverage: 1,469 records
8. `mft_modified_time_changed_to_past` - Coverage: 1,469 records

**No additional features needed** - these already exist and provide direct evidence of manipulation magnitude and direction.

---

### **Priority 5: Event-Time vs Manipulated-Time Features (1 feature)**

**Rationale:**
- ✅ Temporal anomaly detection for LogFile records
- ⚠️ Only works when LogFile data available (~6% of timestomped)
- ✅ High signal when present

---

```python
# Feature: event_vs_modified_after_days
# Formula: (eventtime_dt - lf_modified_time_after).total_seconds() / (24*3600)
# Why: Large discrepancies indicate timestamp changed far into past/future relative to manipulation event
# Coverage: 2,919 records with modified_time_after
# Rationale: If file was "modified" in 2000 but manipulation event occurred in 2023, delta = 23 years
```

**Why only modified_time?**
- Modified time has best coverage (2,919 records vs 473 creation, 292 accessed)
- Focus on single highest-coverage feature to avoid sparse feature space

---

### **Priority 6: Tool Signature Features (2 features)** - Already Exist from Phase 1B

**Rationale:**
- ✅ Already created in Phase 1B
- ⚠️ Very low coverage (1.2% of timestomped events)
- ✅ High precision when present

**Existing Features (from Phase 1B):**
1. `zero_in_nanoseconds` - SetFileTime() API signature
2. `copied_from_file` - SetMACE-like behavior (timestamp copied from another file)

**No additional features needed** - keep these for high-precision signal when present.

---

### **Priority 7: False Positive Filter (1 feature)** - Already Exists from Phase 1B

**Existing Feature:**
- `is_tunneling` - File system tunneling detection (15-second window)

**Rationale:**
- ✅ Research-backed (Oh et al., 2024, Section V.D.3, Algorithm 4)
- ✅ Reduces false positives from legitimate Windows caching behavior
- ✅ Already implemented in Phase 1

---

## 📊 Final Feature Count Summary

| Priority | Feature Group | Count | Coverage | Status |
|----------|---------------|-------|----------|--------|
| **1** | File-Level & Behavioral | **18** | 100% | **NEW - To Create** |
| **2** | Cross-Artifact Correlation | **3** | 100% | **NEW - To Create** |
| **3** | UsnJrnl Pattern Features | **3** | 98% | **NEW - To Create** |
| **4** | Timestamp Deltas | **8** | 2-19% | ✅ **Exist (Phase 1B)** |
| **5** | Event vs Manipulated Time | **1** | 1.9% | **NEW - To Create** |
| **6** | Tool Signatures | **2** | 1.2% | ✅ **Exist (Phase 1B)** |
| **7** | False Positive Filter | **1** | 100% | ✅ **Exist (Phase 1B)** |
| | **TOTAL** | **36** | | **25 NEW + 11 Existing** |

---

## 🚀 Implementation Plan (Phased Approach)

### Phase 2A: File-Level & Behavioral Features (Week 1)
**Objective**: Create features that work on 100% of records and capture APT behavioral patterns

**Tasks:**
1. Implement Location-Based Features (6 features)
   - in_system32, in_windows_dir, in_temp_dir, in_program_files, in_users_dir, path_depth
2. Implement File Type & Attribute Features (6 features)
   - is_executable, is_system_file, is_hidden_file, is_archive, filename_length, has_suspicious_extension
3. Implement Temporal Behavioral Features (6 features)
   - event_frequency_per_file, event_frequency_per_case, events_in_1min_window, events_in_5min_window, time_since_previous_event_seconds, time_until_next_event_seconds

**Validation:**
- Check feature distributions for timestomped vs benign
- Verify no data leakage
- Validate 100% coverage (no NaN except expected cases)

**Output**: 18 new features added to dataset

---

### Phase 2B: Cross-Artifact & UsnJrnl Pattern Features (Week 2)
**Objective**: Add research-backed detection patterns and confidence scoring

**Tasks:**
1. Implement Cross-Artifact Features (3 features)
   - source_confidence_score, has_logfile_evidence, has_usnjrnl_evidence
2. Implement UsnJrnl Pattern Features (3 features)
   - usn_basic_info_change, usn_file_closed, usn_complete_manipulation_pattern
3. Implement Event vs Manipulated Time Feature (1 feature)
   - event_vs_modified_after_days

**Validation:**
- Verify pattern extraction accuracy
- Check confidence score distribution
- Validate against ground truth labels

**Output**: 7 new features added (25 total new features)

---

### Phase 2C: Feature Quality Analysis & Selection (Week 3)
**Objective**: Validate feature quality and remove low-value features

**Tasks:**
1. **Feature Distribution Analysis**
   - Plot distributions for timestomped vs benign
   - Identify features with good separation

2. **Correlation Analysis**
   - Calculate feature correlation matrix
   - Remove highly correlated features (>0.95)

3. **Missing Data Analysis**
   - Document missing data patterns
   - Verify missing data handling strategy

4. **Preliminary Feature Importance** (using Random Forest)
   - Train baseline Random Forest with ALL features
   - Calculate feature importance scores
   - Identify top 20-25 features

5. **Feature Documentation**
   - Document each feature's coverage, rationale, research backing
   - Create feature importance report

**Validation:**
- Verify all 252 timestomped events preserved
- Check for data leakage
- Validate no target leakage

**Output**:
- Feature quality report
- Feature correlation heatmap
- Feature importance rankings
- Curated feature set (top 25-30 features)

---

### Phase 2D: Final Dataset Preparation (Week 4)
**Objective**: Create clean, ML-ready dataset for Phase 3 (baseline model training)

**Tasks:**
1. **Handle Missing Data**
   - Appropriate imputation strategies per feature
   - Document all imputation decisions

2. **Feature Scaling** (if needed for Neural Network)
   - StandardScaler for continuous features
   - Document scaling strategy

3. **Create Feature Subsets**
   - All features (36 total)
   - Top features only (based on importance analysis)
   - File-level + Behavioral only (test Priority 1 independently)

4. **Save Clean Dataset**
   - Output: `all_cases_combined_features.csv`
   - Include feature documentation CSV

5. **Update README & Documentation**

**Output**:
- ML-ready dataset with 36 features
- Feature documentation
- Ready for Phase 3: Baseline Model Training

---

## ✅ Success Criteria

### Phase 2A Success:
- ✅ 18 file-level & behavioral features created
- ✅ 100% coverage validated
- ✅ No NaN values except expected cases (with imputation strategy documented)
- ✅ All 252 timestomped events preserved

### Phase 2B Success:
- ✅ 7 additional features created (25 new features total)
- ✅ UsnJrnl pattern extraction validated
- ✅ Cross-artifact confidence scoring validated

### Phase 2C Success:
- ✅ Feature correlation analysis complete
- ✅ Preliminary feature importance calculated
- ✅ Top 25-30 features identified
- ✅ Feature quality report generated

### Phase 2D Success:
- ✅ Clean ML-ready dataset created
- ✅ Feature documentation complete
- ✅ Missing data handled appropriately
- ✅ Ready for Phase 3: Baseline Model Training

---

## 🎯 Key Design Decisions & Rationale

### Decision 1: Prioritize File-Level Features FIRST
**Reasoning:**
- 94% of timestomped events are UsnJrnl-only (no LogFile data)
- Event patterns overlap heavily between timestomped and benign
- File-level context provides behavioral separation
- Research (Oh et al., 2024, Table 8) validates APT behavioral patterns

**Alternative Considered:**
- Start with timestamp deltas (original plan)
- **Rejected**: Only 6% coverage on timestomped events

---

### Decision 2: Limit Event-Time Comparison Features
**Reasoning:**
- Only one feature (event_vs_modified_after_days) instead of multiple
- Modified time has best coverage (2,919 records)
- Avoid sparse feature space from low-coverage features

**Alternative Considered:**
- Create event_vs_creation, event_vs_accessed, event_vs_mft_modified
- **Rejected**: Too much feature sparsity (coverage 292-1,469 records)

---

### Decision 3: Remove Cross-Timestamp Comparisons
**Reasoning:**
- Cannot compare creation vs modified (only have manipulated timestamps, not all four NTFS timestamps)
- Example: If only ModifiedTime was manipulated, we don't have CreationTime to compare against

**Alternative Considered:**
- Features like "creation_after_modified", "modified_after_accessed"
- **Rejected**: Data doesn't support these features (see TIMESTAMP_AVAILABILITY_ANALYSIS.md)

---

### Decision 4: Include Temporal Clustering Features
**Reasoning:**
- Automated timestomping tools process files in batches
- Temporal clustering (events_in_1min_window, events_in_5min_window) captures this behavior
- Provides context even when individual event patterns overlap

**Research Support:**
- APT malware campaigns manipulate multiple files rapidly
- Time-based clustering is common anti-forensics technique detection method

---

### Decision 5: Keep Tool Signatures Despite Low Coverage
**Reasoning:**
- When present (1.2%), they are high-precision indicators
- Model can learn: "IF zero_in_nanoseconds=True → HIGH confidence timestomping"
- No harm in keeping (doesn't add noise, just low recall)

---

## 📚 Research References

All features are justified by:

**Oh, J., Lee, S., & Hwang, H. (2024)**. "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation." *IEEE Access*, DOI: 10.1109/ACCESS.2024.3395644

**Key Sections:**
- Section V: Detection algorithm based on NTFS journals
- Section V.E.1 (Page 11-12): UsnJrnl detection pattern (BASIC_INFO_CHANGE + CLOSE)
- Section V.D.3 (Page 7): File system tunneling detection (Algorithm 4)
- Section VII: Case study with APT malware
- Table 8: APT malware behavioral patterns (System32 targeting, executables, batch operations)

---

## 📋 Next Steps

1. **Review this revised plan** - Confirm approach makes sense given data analysis
2. **Create Phase 2A notebook** - Start implementing file-level & behavioral features
3. **Iterative validation** - Test each feature group, analyze distributions, verify quality
4. **Proceed to Phase 3** - Baseline model training once features validated

---

## 🔄 Changelog

**2025-01-XX - REVISED Plan Created**
- Analyzed dataset composition (94% UsnJrnl-only, 6% LogFile data)
- Discovered event pattern overlap (Basic_Info_Changed appears in both timestomped and benign)
- Reprioritized features based on coverage and separability
- Reduced feature count from 61-78 to 36 (25 new + 11 existing)
- Emphasized file-level and behavioral features for 100% coverage
- Documented all design decisions with clear rationale

**Previous Version:**
- Original plan prioritized timestamp deltas (Group 1: Temporal Anomaly Features)
- Assumed all records would have LogFile data
- Did not account for UsnJrnl-only event coverage gap
- See: `PHASE_2_PLAN.md` (original, not revised)
