# Oh et al. Methodology Adaptation for ML Training

## Problem Statement

Oh et al.'s methodology requires:
1. Raw $MFT file access (we don't have this)
2. Event-level detection (we're doing file-level aggregation)
3. Cross-artifact timestamp comparison (UsnJrnl doesn't have file metadata in CSV)

## Current Phase 1 Issues

```python
# PROBLEM: We're aggregating events by file
lf_aggregated = lf_filtered.groupby('filename').agg(...)
usn_aggregated = usn_filtered.groupby('filename').agg(...)

# This creates ONE row per file with stats like:
# - event_count, has_time_reversal, has_basic_info_change
# - But LOSES the specific LSN/USN of the manipulation event!

final = pd.merge(lf_aggregated, usn_aggregated, on='filename', how='outer')
# Result: LSN/USN in output don't match Suspicious CSV ground truth
```

## Solution: Keep Event-Level Granularity

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Event-Level Merging                      │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Filter Events (BEFORE merging)
├── LogFile: Keep Time Reversal + Update events only
└── UsnJrnl: Keep Basic_Info_Change events only

Step 2: Add Ground Truth Labels (by exact LSN/USN)
├── Merge LogFile with Suspicious (left join on LSN)
└── Merge UsnJrnl with Suspicious (left join on USN)

Step 3: Cross-Artifact Join (by filename + time window)
├── For each UsnJrnl event
│   └── Find LogFile events for same file within ±5 minutes
├── Create event pairs (UsnJrnl event + matching LogFile events)
└── Keep BOTH UsnJrnl-only and LogFile-only events (outer join logic)

Step 4: Extract Features at EVENT level
├── From LogFile: zero_in_nanoseconds, time_reversal_event, timestamps
├── From UsnJrnl: basic_info_changed, event_time
├── Cross-artifact: has_both_sources, time_difference
└── Temporal: event_count_in_1min_window, event_count_in_5min_window

Result: Each row = ONE event (with LSN/USN preserved!)
```

### Implementation Strategy

#### Phase 1A: Event-Level Merge (NEW!)

```python
def event_level_cross_artifact_merge(lf_filtered, usn_filtered, suspicious_df):
    """
    Event-level merge preserving LSN/USN for exact ground truth matching.

    Unlike file-level aggregation, this keeps each event as a separate row,
    allowing us to identify the SPECIFIC manipulation event by LSN/USN.
    """

    # Step 1: Split suspicious CSV by source
    suspicious_lf = suspicious_df[suspicious_df['source'] == 'logfile'].copy()
    suspicious_usn = suspicious_df[suspicious_df['source'] == 'usnjrnl'].copy()

    # Step 2: Add ground truth labels (exact LSN/USN match)
    lf_with_labels = pd.merge(
        lf_filtered,
        suspicious_lf[['lsn/usn', 'category', 'detail']],
        left_on='lf_lsn',
        right_on='lsn/usn',
        how='left',
        suffixes=('', '_suspicious')
    )
    lf_with_labels['is_flagged_suspicious'] = lf_with_labels['category'].notna()
    lf_with_labels['source_artifact'] = 'logfile'

    usn_with_labels = pd.merge(
        usn_filtered,
        suspicious_usn[['lsn/usn', 'category', 'detail']],
        left_on='usn_usn',
        right_on='lsn/usn',
        how='left',
        suffixes=('', '_suspicious')
    )
    usn_with_labels['is_flagged_suspicious'] = usn_with_labels['category'].notna()
    usn_with_labels['source_artifact'] = 'usnjrnl'

    # Step 3: Prepare for cross-artifact matching
    # Convert event times to datetime for time window matching
    lf_with_labels['event_datetime'] = pd.to_datetime(
        lf_with_labels['lf_event_time'],
        errors='coerce'
    )
    usn_with_labels['event_datetime'] = pd.to_datetime(
        usn_with_labels['usn_event_time'],
        errors='coerce'
    )

    # Step 4: Create unified event dataset
    # Rename columns to common schema
    lf_events = lf_with_labels.rename(columns={
        'lf_lsn': 'event_id',
        'lf_filename': 'filename',
        'lf_full_path': 'full_path',
        'lf_event': 'event_type',
        'lf_detail': 'detail',
        'lf_event_time': 'event_time'
    })

    usn_events = usn_with_labels.rename(columns={
        'usn_usn': 'event_id',
        'usn_filename': 'filename',
        'usn_full_path': 'full_path',
        'usn_event_info': 'event_type',
        'usn_event_time': 'event_time'
    })

    # Step 5: Add cross-artifact validation BEFORE concatenating
    # For each file, check if it appears in BOTH LogFile and UsnJrnl
    files_in_lf = set(lf_events['filename'].dropna())
    files_in_usn = set(usn_events['filename'].dropna())
    files_in_both = files_in_lf & files_in_usn

    lf_events['has_logfile_evidence'] = True
    lf_events['has_usnjrnl_evidence'] = lf_events['filename'].isin(files_in_both)

    usn_events['has_logfile_evidence'] = usn_events['filename'].isin(files_in_both)
    usn_events['has_usnjrnl_evidence'] = True

    # Step 6: Concatenate events (union of LogFile + UsnJrnl events)
    all_events = pd.concat([lf_events, usn_events], ignore_index=True)

    # Step 7: Calculate cross-artifact validation score
    all_events['cross_artifact_validation_score'] = (
        all_events['has_logfile_evidence'].astype(int) * 0.5 +
        all_events['has_usnjrnl_evidence'].astype(int) * 0.5
    )

    return all_events
```

#### Phase 1B: Feature Extraction (Event-Level)

```python
def extract_event_level_features(events_df):
    """
    Extract features at EVENT level (not file level).
    Each row = one event with its own features.
    """

    # 1. FORENSIC PATTERNS (from raw fields - production ready!)

    # Zero in 100-nanoseconds (LogFile Detail field)
    events_df['zero_in_nanoseconds_lf'] = events_df['detail'].fillna('').str.contains(
        'Zero in 100-nanoseconds',
        case=False,
        na=False
    )

    # Zero in nanoseconds (Suspicious detail - ground truth only)
    events_df['zero_in_nanoseconds_gt'] = events_df['detail_suspicious'].fillna('').str.contains(
        'Zero in 100-nanoseconds',
        case=False,
        na=False
    )

    # Combined: True if EITHER source shows zero nanoseconds
    events_df['zero_in_nanoseconds'] = (
        events_df['zero_in_nanoseconds_lf'] |
        events_df['zero_in_nanoseconds_gt']
    )

    # Time Reversal Event
    events_df['time_reversal_event'] = events_df['event_type'].fillna('').str.contains(
        'Time Reversal',
        case=False,
        na=False
    )

    # Basic Info Changed
    events_df['basic_info_changed'] = events_df['event_type'].fillna('').str.contains(
        'Basic_Info_Change',
        case=False,
        na=False
    )

    # Update Resident Value
    events_df['update_resident_value'] = events_df['event_type'].fillna('').str.contains(
        'Update',
        case=False,
        na=False
    )

    # 2. CROSS-ARTIFACT VALIDATION
    # (Already calculated in merge function)

    # 3. TEMPORAL FEATURES (event frequency per file)

    # Events per file (how many events this file has)
    events_df['event_count_per_file'] = events_df.groupby('filename')['filename'].transform('count')

    # Events in time windows (requires event_datetime column)
    if 'event_datetime' in events_df.columns:
        events_df = events_df.sort_values(['filename', 'event_datetime'])

        # Rolling 1-minute window
        events_df['events_in_1min_window'] = events_df.groupby('filename').rolling(
            '1T',
            on='event_datetime'
        )['filename'].count().reset_index(drop=True)

        # Rolling 5-minute window
        events_df['events_in_5min_window'] = events_df.groupby('filename').rolling(
            '5T',
            on='event_datetime'
        )['filename'].count().reset_index(drop=True)

    # 4. FILE CHARACTERISTICS

    events_df['is_executable'] = events_df['filename'].fillna('').str.endswith(
        ('.exe', '.dll', '.sys', '.bat', '.cmd', '.ps1')
    )

    events_df['is_document'] = events_df['filename'].fillna('').str.endswith(
        ('.doc', '.docx', '.pdf', '.txt', '.xls', '.xlsx', '.ppt', '.pptx')
    )

    events_df['is_image'] = events_df['filename'].fillna('').str.endswith(
        ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    )

    events_df['is_archive'] = events_df['filename'].fillna('').str.endswith(
        ('.zip', '.rar', '.7z', '.tar', '.gz')
    )

    events_df['path_depth'] = events_df['full_path'].fillna('').str.count('\\\\')
    events_df['filename_length'] = events_df['filename'].fillna('').str.len()

    # 5. PARSE TIMESTAMPS FROM LOGFILE (when available)

    # Extract timestamps from LogFile columns (these are the "current $MFT" values)
    if 'lf_creation_time' in events_df.columns:
        events_df['current_creation_time'] = pd.to_datetime(
            events_df['lf_creation_time'],
            errors='coerce'
        )

    if 'lf_modified_time' in events_df.columns:
        events_df['current_modified_time'] = pd.to_datetime(
            events_df['lf_modified_time'],
            errors='coerce'
        )

    # Parse BEFORE → AFTER from Detail field
    # Example: "ModifiedTime : 2018-04-06 20:35:25 -> 2018-04-05 10:13:47(Zero in 100-nanoseconds)"
    detail_pattern = r':\s*([^-]+)\s*->\s*([^(]+)'
    events_df['timestamp_before'] = events_df['detail'].str.extract(detail_pattern)[0].str.strip()
    events_df['timestamp_after'] = events_df['detail'].str.extract(detail_pattern)[1].str.strip()

    return events_df
```

### Ground Truth Labeling Strategy

```python
def create_ground_truth_labels(events_df):
    """
    Create ground truth labels at EVENT level.

    CRITICAL: Only the events with LSN/USN matching Suspicious CSV
    should be labeled as suspicious!
    """

    # Label = 1 if this SPECIFIC event is in Suspicious CSV
    events_df['ground_truth_label'] = events_df['is_flagged_suspicious'].astype(int)

    # Confidence level based on detection factors
    def assign_confidence(row):
        if row['is_flagged_suspicious']:
            # Count detection factors
            factors = 0
            if row['zero_in_nanoseconds']:
                factors += 1
            if row['cross_artifact_validation_score'] >= 1.0:  # Both sources
                factors += 1
            if row['time_reversal_event'] or row['basic_info_changed']:
                factors += 1

            # Oh et al. levels
            if factors >= 2:
                return 'MALICIOUS'  # High confidence
            else:
                return 'SUSPICIOUS'  # Medium confidence
        else:
            return 'BENIGN'

    events_df['confidence_level'] = events_df.apply(assign_confidence, axis=1)

    return events_df
```

## Key Advantages of Event-Level Approach

1. **LSN/USN Preserved**: Each row has the exact LSN/USN of the event
2. **Ground Truth Alignment**: Can match Suspicious CSV by exact LSN/USN
3. **Event-Specific Features**: Can extract before/after timestamps from Detail
4. **Cross-Artifact Validation**: Can identify which specific events are detected by both sources
5. **Production-Ready**: Features extracted from raw LogFile/UsnJrnl fields only

## Handling UsnJrnl-Only Detections (Without $MFT)

For events detected ONLY by UsnJrnl (like DeathToll.jpg USN 239046272):

**Option 1: Use Suspicious CSV labels as ground truth**
```python
# Accept that we cannot EXTRACT the detection pattern from UsnJrnl alone
# BUT we can LEARN from labeled examples in training data
# The model learns: "files with this pattern of UsnJrnl events = suspicious"
```

**Option 2: Partial feature extraction**
```python
# We CAN extract:
# - BASIC_INFO_CHANGE pattern ✓
# - Event timestamp ✓
# - Event frequency ✓
# - File characteristics ✓
#
# We CANNOT extract:
# - Current $MFT timestamps ✗
# - Before/after timestamp comparison ✗
# - Zero nanoseconds from UsnJrnl ✗ (only from LogFile Detail)
```

**Option 3: Request $MFT files for datasets**
```python
# If we want full Oh et al. replication, we'd need:
# - Raw $MFT files for each dataset
# - Custom parser to extract $SI attributes
# - This is complex and may not be available
```

## Recommended Path Forward

1. **Redesign Phase 1**: Event-level merge (not file-level aggregation)
2. **Use Suspicious CSV**: Exact LSN/USN matching for ground truth
3. **Extract LogFile features**: Focus on LogFile Detail field for zero nanoseconds
4. **Train on labeled events**: Let the model learn from Suspicious CSV labels
5. **Accept UsnJrnl limitations**: We can't fully replicate UsnJrnl-only detections without $MFT

This approach:
- ✅ Preserves LSN/USN for ground truth matching
- ✅ Extracts production-ready features from raw CSVs
- ✅ Implements cross-artifact validation
- ✅ Trains on event-level patterns (not file-level)
- ⚠️  Limited for UsnJrnl-only detections (requires Suspicious CSV labels for training)