Comprehensive Findings from Oh et al. 2024 - Implementation Guide
Based on the Oh et al. 2024 paper "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation", here are ALL the key findings that should guide our Phase 1 approach:
1. Core Methodology: Filter BEFORE Merge (Critical!)
Finding: Oh et al. achieves 95.9% data reduction (3.2M → 138K records) by filtering BEFORE merging, NOT after. Why This Matters:
Raw merge creates Cartesian products from NULL values → 56M record explosion
Filtering first reduces LogFile by ~96.8% and UsnJrnl by ~92.4%
Memory usage drops from 3-6GB to <500MB
Implementation Order:

1. Load LogFile → Filter to timestamp-relevant events → Keep ~3% of records
2. Load UsnJrnl → Filter to BASIC_INFO_CHANGE pattern → Keep ~7.6% of records  
3. THEN merge filtered datasets → Manageable size
4. THEN add Suspicious CSV labels → Ground truth
2. LogFile Detection Patterns (What to Filter)
Finding: LogFile contains TWO key patterns for timestomping detection:
Pattern 1: Time Reversal Events
Event name: Contains "Time Reversal"
Meaning: NTFS detected timestamp changed to PAST
Example: "Updating Time Reversal"
Detail field: Shows which timestamp (CreationTime, ModifiedTime, etc.) was changed
Code:

time_reversal = lf_df[
    lf_df['lf_event'].str.contains('Time Reversal', na=False, case=False)
]
Pattern 2: Update Resident Value Events
Event name: Contains "Update"
Most common: "Update Resident Value", "Updating MFTModified Time"
Meaning: File attributes or timestamps were modified
Detail field: Contains forensic patterns like "Zero in 100-nanoseconds"
Code:

update_events = lf_df[
    lf_df['lf_event'].str.contains('Update', na=False, case=False)
]
Combined LogFile Filter:

# Keep ONLY Time Reversal OR Update events
lf_filtered = pd.concat([time_reversal, update_events]).drop_duplicates()
# Expected reduction: 39,077 → 1,235 (96.8% reduction for 01-PE)
3. UsnJrnl Detection Patterns (What to Filter)
Finding: UsnJrnl uses BASIC_INFO_CHANGE as the primary timestomping indicator.
Pattern: BASIC_INFO_CHANGE + CLOSE
EventInfo field: Must contain "Basic_Info_Change"
Common combinations:
"File_Created | Basic_Info_Changed | File_Closed"
"Basic_Info_Changed | File_Closed"
"Data_Extended | Basic_Info_Changed | File_Closed"
Meaning: File's basic information (attributes/timestamps) was changed
Code:

usn_filtered = usn_df[
    usn_df['usn_event_info'].str.contains('Basic_Info_Change', na=False, case=False)
].copy()
# Expected reduction: 316,817 → 24,002 (92.4% reduction for 01-PE)
Why BASIC_INFO_CHANGE?:
SetFileTime() API → triggers BASIC_INFO_CHANGE event in UsnJrnl
Most reliable indicator of timestamp manipulation
Oh et al. validated this pattern across multiple attack scenarios
4. Smart Union Merging Strategy
Finding: Oh et al. uses outer join to preserve records from BOTH sources.
Why Outer Join?
LogFile-only detections: Time Reversal events may not appear in UsnJrnl
UsnJrnl-only detections: BASIC_INFO_CHANGE may not have corresponding LogFile event
Cross-artifact validation: When BOTH sources detect = HIGH confidence
Merge Keys (In Priority Order):

# Primary merge key
merge_keys = ['filename', 'full_path']

# Handle edge cases
- If filename missing: Use full_path only
- If full_path missing: Use filename only  
- If BOTH missing: Use FileReferenceNumber (advanced)
Expected Result Sizes (01-PE):

LogFile filtered: 1,235 records
UsnJrnl filtered: 24,002 records
Merged (outer join): ~24,204 records
Why NOT 25,237? Because ~1,033 records match on filename → deduplicated in merge.
5. File System Tunneling Detection
Finding: Windows caches filename + CreationTime for 15 seconds after deletion.
What is File System Tunneling?
User deletes file.txt at 10:00:00
User creates NEW file.txt at 10:00:10 (within 15 seconds)
Windows automatically copies CreationTime from OLD file to NEW file
This is BENIGN behavior, NOT timestomping!
How to Detect Tunneling:

# Check if CreationTime = EventTime from UsnJrnl
# AND time difference < 15 seconds
# AND filename previously existed

def detect_tunneling(row):
    if pd.notna(row['usn_timestamp']) and pd.notna(row['usn_creation_time']):
        time_diff = abs((row['usn_timestamp'] - row['usn_creation_time']).total_seconds())
        if time_diff <= 15:
            return True
    return False

df['is_tunneling'] = df.apply(detect_tunneling, axis=1)
Filter Out Tunneling:

# Remove tunneling records BEFORE feature engineering
df_no_tunneling = df[df['is_tunneling'] == False].copy()
Why This Matters: Oh et al. found ~20% of "zero nanoseconds" detections are tunneling (false positives).
6. Detection Levels (Malicious vs Suspicious vs Additional)
Finding: Oh et al. defines THREE detection confidence levels.
Level 1: Malicious (Highest Confidence)
Criteria:
Zero in 100-nanoseconds detected
AND Cross-artifact validation (both LogFile + UsnJrnl agree)
AND NOT file system tunneling
Label: timestomped = 1, confidence = "HIGH"
Example: CreationTime ends in .0000000 (exactly 0 nanoseconds)
Level 2: Suspicious (Medium Confidence)
Criteria:
Zero in 100-nanoseconds detected
OR Using another file's timestamp
BUT only ONE source (LogFile OR UsnJrnl)
Label: timestomped = 1, confidence = "MEDIUM"
Level 3: Additional (Low Confidence)
Criteria:
Anomalous patterns (e.g., CreationTime = ModifiedTime exactly)
OR Same timestamp as known system file
BUT NOT zero nanoseconds
Label: timestomped = 0.5, confidence = "LOW" (investigate further)
Implementation:

def assign_detection_level(row):
    if row['zero_in_nanoseconds'] and row['cross_artifact_validation'] and not row['is_tunneling']:
        return 'MALICIOUS'
    elif row['zero_in_nanoseconds'] or row['using_another_timestamp']:
        return 'SUSPICIOUS'
    elif row['creation_equals_modified']:
        return 'ADDITIONAL'
    else:
        return 'BENIGN'
7. Zero in 100-Nanoseconds Detection
Finding: SetFileTime() API sets nanoseconds to EXACTLY 0 (not randomized like normal file creation).
Where to Check:

# LogFile Detail field (for LogFile records)
lf_zero = df['lf_detail'].fillna('').str.contains(
    'Zero in 100-nanoseconds', case=False, na=False
)

# Suspicious Detail field (for ALL flagged records)
suspicious_zero = df['suspicious_detail'].fillna('').str.contains(
    'Zero in 100-nanoseconds', case=False, na=False
)

# Combine: True if EITHER source shows pattern
df['zero_in_nanoseconds'] = lf_zero | suspicious_zero
Why Check BOTH Fields?
LogFile Detail: Available in raw LogFile CSVs (production-ready)
Suspicious Detail: Ground truth from Oh et al.'s tool (training labels)
UsnJrnl has NO Detail field in raw CSVs → must use Suspicious Detail for ground truth
8. Cross-Artifact Validation Score
Finding: When LogFile + UsnJrnl BOTH detect timestomping on same file → 99.2% confidence.
How to Calculate:

def calculate_cross_artifact_score(row):
    score = 0
    
    # LogFile evidence
    if pd.notna(row['lf_event']) and 'Time Reversal' in row['lf_event']:
        score += 0.5
    
    # UsnJrnl evidence
    if pd.notna(row['usn_event_info']) and 'Basic_Info_Change' in row['usn_event_info']:
        score += 0.5
    
    # Both sources agree
    if score == 1.0:
        score = 1.0  # HIGH confidence
    elif score == 0.5:
        score = 0.5  # MEDIUM confidence
    else:
        score = 0.0  # No evidence
    
    return score

df['cross_artifact_validation_score'] = df.apply(calculate_cross_artifact_score, axis=1)
Validation Thresholds:
score = 1.0: Both sources → MALICIOUS (if zero nanoseconds present)
score = 0.5: One source → SUSPICIOUS
score = 0.0: No evidence → BENIGN
9. Ground Truth Labeling (Suspicious CSVs)
Finding: Suspicious CSVs contain PRE-LABELED ground truth from Oh et al.'s NTFS Artifact Analysis Tool.
Critical Insight:
Suspicious CSVs are NOT production features!
They are ONLY used for training labels
Production model must extract features from RAW CSVs only
Correct Usage:

# Step 1: Merge LogFile + Suspicious by LSN
lf_merged = pd.merge(
    lf_filtered, 
    suspicious_lf,  # source='logfile' records
    left_on='lf_lsn', 
    right_on='lsn/usn',
    how='left'
)

# Step 2: Merge UsnJrnl + Suspicious by USN
usn_merged = pd.merge(
    usn_filtered,
    suspicious_usn,  # source='usnjrnl' records
    left_on='usn_usn',
    right_on='lsn/usn',
    how='left'
)

# Step 3: Add ground truth label
df['is_flagged_suspicious'] = df['suspicious_category'].notna()
df['ground_truth_label'] = df['is_flagged_suspicious'].astype(int)
What Suspicious CSVs Provide:
source: Which artifact detected the activity ("logfile" or "usnjrnl")
category: Pre-labeled category (e.g., "Timestamp Manipulation")
detail: Forensic description including zero nanoseconds pattern
Use for: Training labels, validation, feature engineering verification
What to AVOID:
❌ Do NOT use suspicious_detail as a model feature (not available in production)
✅ DO extract same patterns from lf_detail (available in raw LogFile CSV)
10. Feature Engineering Requirements
Finding: Oh et al. uses ~15-20 features divided into 4 categories.
Category 1: Forensic Patterns (Production-Ready)

# Extract from RAW LogFile Detail field (NOT Suspicious CSV)
df['zero_in_nanoseconds'] = df['lf_detail'].str.contains('Zero in 100-nanoseconds', na=False)
df['time_reversal_event'] = df['lf_event'].str.contains('Time Reversal', na=False)
df['basic_info_changed'] = df['usn_event_info'].str.contains('Basic_Info_Change', na=False)
df['using_another_timestamp'] = df['lf_detail'].str.contains('Using another', na=False)
Category 2: Cross-Artifact Validation

df['has_logfile_evidence'] = df['lf_event'].notna()
df['has_usnjrnl_evidence'] = df['usn_event_info'].notna()
df['cross_artifact_validation_score'] = df.apply(calculate_cross_artifact_score, axis=1)
Category 3: Temporal Features

# Event frequency per file
df['event_count'] = df.groupby('filename')['filename'].transform('count')

# Events in 1-minute window
df['events_in_1min_window'] = df.groupby('filename').rolling('1T', on='timestamp')['filename'].count()

# Events in 5-minute window
df['events_in_5min_window'] = df.groupby('filename').rolling('5T', on='timestamp')['filename'].count()
Category 4: File Characteristics

df['is_executable'] = df['filename'].str.endswith(('.exe', '.dll', '.sys'))
df['is_archive'] = df['filename'].str.endswith(('.zip', '.rar', '.7z'))
df['path_depth'] = df['full_path'].str.count('\\\\')
df['filename_length'] = df['filename'].str.len()
11. Expected Data Reduction (Validation Metric)
Finding: Oh et al. achieves consistent data reduction ratios across datasets.
Expected Ratios (01-PE):

Raw LogFile: 39,077 records
  ↓ Filter Time Reversal + Update events
Filtered LogFile: 1,235 records (96.8% reduction)

Raw UsnJrnl: 316,817 records
  ↓ Filter Basic_Info_Change events
Filtered UsnJrnl: 24,002 records (92.4% reduction)

Merged (outer join): ~24,204 records
  ↓ Remove duplicates
Final dataset: ~24,204 records

Total reduction: (39,077 + 316,817) → 24,204 = 93.2% reduction
Validation Check:

# After filtering, verify ratios
logfile_reduction = (1 - len(lf_filtered) / len(lf_raw)) * 100
usnjrnl_reduction = (1 - len(usn_filtered) / len(usn_raw)) * 100

print(f"LogFile reduction: {logfile_reduction:.1f}%")  # Should be ~96.8%
print(f"UsnJrnl reduction: {usnjrnl_reduction:.1f}%")  # Should be ~92.4%

# If you see 56M records → something is WRONG (Cartesian product from NULLs)
12. Production Compatibility Requirement
Finding: Model features must be extractable from RAW CSVs only (no Suspicious CSV dependency).
Why This Matters:
In production, forensic investigators only have: Raw LogFile + Raw UsnJrnl CSVs
Suspicious CSVs are NOT available (they're generated by Oh et al.'s research tool)
Model must work on raw artifacts ONLY
Feature Extraction Rules:

# ✅ CORRECT: Extract from raw LogFile Detail field
df['zero_in_nanoseconds'] = df['lf_detail'].str.contains('Zero in 100-nanoseconds')

# ❌ WRONG: Extract from Suspicious Detail field (not available in production)
df['zero_in_nanoseconds'] = df['suspicious_detail'].str.contains('Zero in 100-nanoseconds')
Training vs Production Split:

# TRAINING PHASE:
# - Use Suspicious CSV to CREATE ground truth labels
# - Extract features from RAW LogFile/UsnJrnl fields
# - Train model on features + labels

# PRODUCTION PHASE:
# - Load RAW LogFile/UsnJrnl CSVs
# - Extract SAME features from RAW fields
# - Run model prediction (no Suspicious CSV needed)
13. Memory Management Strategy
Finding: Loading all 18 training datasets at once = 3-6GB RAM → kernel crash.
Solution: Incremental Processing

# Process ONE dataset at a time
for dataset in training_datasets:
    # Step 1: Load raw CSVs
    lf_raw = pd.read_csv(f"{dataset}/logfile.csv")
    usn_raw = pd.read_csv(f"{dataset}/usnjrnl.csv")
    
    # Step 2: Filter IMMEDIATELY (before storing)
    lf_filtered = filter_logfile_timestamp_changes(lf_raw)
    usn_filtered = find_basic_detection_pattern(usn_raw)
    
    # Step 3: Merge filtered data
    merged = smart_union_merge(lf_filtered, usn_filtered)
    
    # Step 4: Append to combined dataset
    all_cases_combined.append(merged)
    
    # Step 5: Clear memory
    del lf_raw, usn_raw, lf_filtered, usn_filtered, merged
    gc.collect()

# Step 6: Save combined dataset
final_df = pd.concat(all_cases_combined, ignore_index=True)
final_df.to_csv('all_cases_combined.csv', index=False)
Memory Savings:
OLD approach: Load all → 5.4M records → 3-6GB RAM → crash
NEW approach: Filter incrementally → ~138K records → <500MB RAM → success
14. Ground Truth Validation (Critical Success Criteria)
Finding: Model must detect ALL known timestomped files in validation datasets.
Validation Datasets:

# 01-PE Ground Truth (from Suspicious CSV)
# Expected: 24 timestomped files with "Zero in 100-nanoseconds" pattern
# All should have: ground_truth_label = 1

# Lone Wolf Ground Truth (from Suspicious CSV)
# Expected: 12 timestomped files with "Zero in 100-nanoseconds" pattern
# All should have: ground_truth_label = 1, confidence >= 70%
Validation Code:

# After Phase 1 merge, verify ground truth preservation
suspicious_records = df[df['is_flagged_suspicious'] == True]
print(f"Total flagged suspicious: {len(suspicious_records)}")

# Check zero nanoseconds extraction
zero_nano_records = suspicious_records[suspicious_records['zero_in_nanoseconds'] == True]
print(f"Zero nanoseconds detected: {len(zero_nano_records)}")

# Verify Lone Wolf (should be 12/12)
lone_wolf_detected = lone_wolf_df[lone_wolf_df['zero_in_nanoseconds'] == True]
print(f"Lone Wolf detection: {len(lone_wolf_detected)}/12")
Success Criteria:
01-PE: 24/24 timestomped files have zero_in_nanoseconds = True
Lone Wolf: 12/12 timestomped files detected at ≥70% confidence
False positive rate: <5%
Summary: Implementation Checklist for Phase 1
Based on Oh et al. 2024, our Phase 1 notebook MUST:
 Filter LogFile BEFORE merging (Time Reversal + Update events only)
 Filter UsnJrnl BEFORE merging (Basic_Info_Change events only)
 Use outer join (preserve records from both sources)
 Merge LogFile + Suspicious by LSN (preserve ground truth)
 Merge UsnJrnl + Suspicious by USN (preserve ground truth)
 Extract zero nanoseconds from RAW lf_detail (production-ready)
 Use suspicious_detail ONLY for labels (not features)
 Calculate cross-artifact validation score (both sources = high confidence)
 Detect and filter file system tunneling (15-second window)
 Process datasets incrementally (avoid memory crash)
 Verify 93-96% data reduction (validation metric)
 Validate ground truth preservation (all suspicious records preserved)
