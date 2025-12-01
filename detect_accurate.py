#!/usr/bin/env python3
"""
Digital Detectives - Accurate Detection Pipeline
Following EXACT methodology from Phase 1-3 notebooks

This script replicates the notebook workflow for testing on new datasets.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# COLORS (for terminal output)
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Print tool banner."""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    DIGITAL DETECTIVES                            ║")
    print("║         Accurate Detection Pipeline (Notebook-Based)             ║")
    print("║                                                                  ║")
    print("║  Phase 1: Smart Union Merging (No Aggressive Filtering!)        ║")
    print("║  Phase 2: Feature Engineering (26 features)                     ║")
    print("║  Phase 3: Model Inference & Prediction                          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")


# ============================================================================
# PHASE 1: SMART UNION MERGING (FROM NOTEBOOK)
# ============================================================================

def find_basic_detection_pattern(usn_df):
    """
    Filter UsnJrnl to BASIC_INFO_CHANGE pattern.
    Based on: Phase 1 notebook, cell 5
    """
    result = usn_df[
        usn_df['usn_event_info'].str.contains('Basic_Info_Change', na=False, case=False)
    ].copy()
    return result


def filter_logfile_timestamp_changes(lf_df):
    """
    Filter LogFile to timestamp-relevant events.
    Based on: Phase 1 notebook, cell 6
    """
    # Keep Time Reversal events
    time_reversal = lf_df[
        lf_df['lf_event'].str.contains('Time Reversal', na=False, case=False)
    ]

    # Keep Update events
    update_events = lf_df[
        lf_df['lf_event'].str.contains('Update', na=False, case=False)
    ]

    # Combine and remove duplicates
    result = pd.concat([time_reversal, update_events]).drop_duplicates()
    return result


def phase1_smart_union(logfile_path: str, usnjrnl_path: str) -> pd.DataFrame:
    """
    Phase 1: Smart Union Merging
    EXACTLY following Phase 1 notebook methodology (cell 11)
    """
    print(f"\n{Colors.OKBLUE}[PHASE 1] Smart Union Merging{Colors.ENDC}")

    # Load files
    print(f"  • Loading LogFile...")
    lf_raw = pd.read_csv(logfile_path, low_memory=False)
    print(f"    ✓ {len(lf_raw):,} records")

    print(f"  • Loading UsnJrnl...")
    usn_raw = pd.read_csv(usnjrnl_path, low_memory=False)
    print(f"    ✓ {len(usn_raw):,} records")

    # Standardize columns (matching notebook)
    print(f"  • Standardizing columns...")
    lf_df = pd.DataFrame()
    lf_df['lf_lsn'] = lf_raw['LSN'] if 'LSN' in lf_raw.columns else None
    lf_df['eventtime'] = lf_raw['EventTime(UTC+8)'] if 'EventTime(UTC+8)' in lf_raw.columns else lf_raw.get('EventTime', None)
    lf_df['lf_event'] = lf_raw.get('Event', None)
    lf_df['lf_detail'] = lf_raw.get('Detail', None)
    lf_df['filename'] = lf_raw.get('File/Directory Name', None)
    lf_df['filepath'] = lf_raw.get('Full Path', lf_raw.get('FullPath', None))
    lf_df['lf_redo'] = lf_raw.get('Redo', None)
    lf_df['lf_target_vcn'] = lf_raw.get('Target VCN', None)
    lf_df['lf_cluster_index'] = lf_raw.get('Cluster Index', None)

    usn_df = pd.DataFrame()
    usn_df['eventtime'] = usn_raw['TimeStamp(UTC+8)'] if 'TimeStamp(UTC+8)' in usn_raw.columns else usn_raw.get('TimeStamp', None)
    usn_df['usn_usn'] = usn_raw.get('USN', None)
    usn_df['filename'] = usn_raw.get('File/Directory Name', None)
    usn_df['filepath'] = usn_raw.get('FullPath', usn_raw.get('Full Path', None))
    usn_df['usn_event_info'] = usn_raw.get('EventInfo', None)
    usn_df['usn_source_info'] = usn_raw.get('SourceInfo', None)
    usn_df['usn_file_attribute'] = usn_raw.get('FileAttribute', None)
    usn_df['usn_file_reference_number'] = usn_raw.get('FileReferenceNumber', None)
    usn_df['usn_parent_file_reference_number'] = usn_raw.get('ParentFileReferenceNumber', None)

    # Parse timestamps
    lf_df['eventtime_dt'] = pd.to_datetime(lf_df['eventtime'], errors='coerce')
    usn_df['eventtime_dt'] = pd.to_datetime(usn_df['eventtime'], errors='coerce')

    # Create merge keys
    lf_df['merge_key'] = (lf_df['filepath'].fillna('').astype(str) + '|' +
                          lf_df['filename'].fillna('').astype(str))
    usn_df['merge_key'] = (usn_df['filepath'].fillna('').astype(str) + '|' +
                           usn_df['filename'].fillna('').astype(str))

    # Add case_id
    lf_df['case_id'] = 999
    usn_df['case_id'] = 999

    print(f"  • After standardization: {len(lf_df):,} LogFile, {len(usn_df):,} UsnJrnl")

    # STEP 2: Filter for matching (but keep originals!)
    print(f"  • Filtering for cross-artifact matching...")
    lf_filtered = filter_logfile_timestamp_changes(lf_df)
    usn_filtered = find_basic_detection_pattern(usn_df)
    print(f"    ✓ LogFile: {len(lf_df):,} → {len(lf_filtered):,} (for matching)")
    print(f"    ✓ UsnJrnl: {len(usn_df):,} → {len(usn_filtered):,} (for matching)")

    # STEP 3: Match with ±1 second window
    print(f"  • Matching LogFile ↔ UsnJrnl (±1 second window)...")
    matched_records = []
    matched_lf_indices = set()
    matched_usn_indices = set()

    for lf_idx, lf_row in lf_filtered.iterrows():
        potential_matches = usn_filtered[
            (usn_filtered['merge_key'] == lf_row['merge_key']) &
            (usn_filtered['eventtime_dt'] >= lf_row['eventtime_dt'] - timedelta(seconds=1)) &
            (usn_filtered['eventtime_dt'] <= lf_row['eventtime_dt'] + timedelta(seconds=1))
        ]

        if len(potential_matches) > 0:
            # Take closest match
            potential_matches = potential_matches.copy()
            potential_matches['time_diff'] = (potential_matches['eventtime_dt'] - lf_row['eventtime_dt']).abs()
            closest = potential_matches.nsmallest(1, 'time_diff').iloc[0]

            # Merge records
            merged_row = lf_row.copy()
            for col in usn_filtered.columns:
                if col not in merged_row.index and col not in ['eventtime', 'eventtime_dt', 'merge_key', 'case_id', 'filename', 'filepath']:
                    merged_row[col] = closest[col]

            merged_row['source'] = 'both'
            merged_row['time_diff_seconds'] = closest['time_diff'].total_seconds()
            matched_records.append(merged_row)

            matched_lf_indices.add(lf_idx)
            matched_usn_indices.add(closest.name)

    matched_df = pd.DataFrame(matched_records) if matched_records else pd.DataFrame()
    print(f"    ✓ Matched: {len(matched_df):,} (source='both')")

    # STEP 4: Unmatched LogFile (from FILTERED set only - these are the ones we tried to match)
    lf_only = lf_filtered[~lf_filtered.index.isin(matched_lf_indices)].copy()
    lf_only['source'] = 'logfile_only'
    lf_only['time_diff_seconds'] = np.nan
    print(f"    ✓ LogFile-only: {len(lf_only):,}")

    # STEP 5: Unmatched UsnJrnl (from FILTERED set only)
    usn_only = usn_filtered[~usn_filtered.index.isin(matched_usn_indices)].copy()
    usn_only['source'] = 'usnjrnl_only'
    usn_only['time_diff_seconds'] = np.nan
    print(f"    ✓ UsnJrnl-only: {len(usn_only):,}")

    # STEP 6: Smart Union
    print(f"  • Creating smart union...")
    final_df = pd.concat([matched_df, lf_only, usn_only], ignore_index=True)

    # Sort by time
    final_df = final_df.sort_values('eventtime_dt').reset_index(drop=True)

    print(f"  {Colors.OKGREEN}✓ Phase 1 Complete: {len(final_df):,} events{Colors.ENDC}")
    print(f"    • both: {(final_df['source'] == 'both').sum():,}")
    print(f"    • logfile_only: {(final_df['source'] == 'logfile_only').sum():,}")
    print(f"    • usnjrnl_only: {(final_df['source'] == 'usnjrnl_only').sum():,}")

    return final_df


# ============================================================================
# PHASE 2: FEATURE ENGINEERING (SIMPLIFIED - FROM PROCESSED DATA)
# ============================================================================

def phase2_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Engineer ALL 16 features required by the model
    Replicates Phase 2A + 2B feature engineering from notebooks
    """
    print(f"\n{Colors.OKBLUE}[PHASE 2] Feature Engineering{Colors.ENDC}")
    print(f"  • Creating all 16 required features...")

    # ===== FILE-LEVEL FEATURES =====

    # Feature 1-2: Path-based location features
    df['in_temp_dir'] = df['filepath'].str.contains(r'\\Temp\\|\\temp\\', na=False, regex=True).astype(int)
    df['in_windows_dir'] = df['filepath'].str.contains(r'\\Windows\\', na=False, regex=True).astype(int)
    df['in_program_files'] = df['filepath'].str.contains(r'\\Program Files', na=False, regex=True).astype(int)
    df['in_users_dir'] = df['filepath'].str.contains(r'\\Users\\', na=False, regex=True).astype(int)

    # Feature 3: Path depth
    df['path_depth'] = df['filepath'].str.count(r'\\').fillna(0).astype(int)

    # Feature 4: Filename length
    df['filename_length'] = df['filename'].str.len().fillna(0).astype(int)

    # Feature 5-6: File type features
    df['file_extension'] = df['filename'].str.split('.').str[-1].fillna('').str.lower()
    df['is_executable'] = df['file_extension'].isin(['exe', 'dll', 'sys']).astype(int)
    df['is_archive'] = df['file_extension'].isin(['zip', 'rar', '7z', 'tar', 'gz']).astype(int)

    # Feature 7: Event frequency per file
    file_counts = df.groupby('filepath').size()
    df['event_frequency_per_file'] = df['filepath'].map(file_counts).fillna(1).astype(int)

    # Feature 8: Event frequency per case
    df['event_frequency_per_case'] = len(df)  # Total events in this dataset

    # ===== TEMPORAL FEATURES =====

    # Sort by time for temporal features
    df = df.sort_values('eventtime_dt').reset_index(drop=True)

    # Feature 9: Time since previous event
    df['time_since_previous_event_seconds'] = df['eventtime_dt'].diff().dt.total_seconds()
    df['time_since_previous_event_seconds'] = df['time_since_previous_event_seconds'].fillna(999999.0)

    # Feature 10: Time until next event
    df['time_until_next_event_seconds'] = df['eventtime_dt'].diff(-1).abs().dt.total_seconds()
    df['time_until_next_event_seconds'] = df['time_until_next_event_seconds'].fillna(0.0)

    # Feature 11-12: Events in time windows
    print(f"  • Computing windowed features (this may take a moment)...")
    df['events_in_1min_window'] = 0
    df['events_in_5min_window'] = 0

    from datetime import timedelta
    window_1min = timedelta(minutes=1)
    window_5min = timedelta(minutes=5)

    for idx in range(len(df)):
        current_time = df.at[idx, 'eventtime_dt']
        start_1min = current_time - window_1min
        end_1min = current_time + window_1min
        start_5min = current_time - window_5min
        end_5min = current_time + window_5min

        mask_1min = (df['eventtime_dt'] >= start_1min) & (df['eventtime_dt'] <= end_1min)
        mask_5min = (df['eventtime_dt'] >= start_5min) & (df['eventtime_dt'] <= end_5min)

        df.at[idx, 'events_in_1min_window'] = mask_1min.sum()
        df.at[idx, 'events_in_5min_window'] = mask_5min.sum()

        if idx % 1000 == 0 and idx > 0:
            print(f"    • Processed {idx:,}/{len(df):,} events...")

    # ===== CROSS-ARTIFACT FEATURES =====

    # Feature 13: Has LogFile evidence
    logfile_files = set(df[df['source'].isin(['logfile_only', 'both'])]['filepath'].unique())
    df['has_logfile_evidence'] = df['filepath'].apply(lambda x: int(x in logfile_files))

    # Feature 14: USN complete manipulation pattern
    # Check for BASIC_INFO_CHANGE in USN events
    manipulation_keywords = ['basic_info_change', 'basic_info_changed']
    df['usn_complete_manipulation_pattern'] = df['usn_event_info'].apply(
        lambda x: int(any(kw in str(x).lower() for kw in manipulation_keywords)) if pd.notna(x) else 0
    )

    print(f"  {Colors.OKGREEN}✓ Phase 2 Complete: All 16 features engineered{Colors.ENDC}")

    return df


# ============================================================================
# PHASE 3: MODEL INFERENCE
# ============================================================================

def phase3_predict(df: pd.DataFrame, model_path: str, threshold: float = 0.5) -> pd.DataFrame:
    """Phase 3: Run model inference."""
    print(f"\n{Colors.OKBLUE}[PHASE 3] Model Inference & Prediction{Colors.ENDC}")

    # Load model
    print(f"  • Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Get expected features
    expected_features = model.get_booster().feature_names
    print(f"  • Model expects {len(expected_features)} features")

    # Check for missing features (shouldn't happen with complete feature engineering)
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        print(f"  {Colors.WARNING}⚠ Missing {len(missing)} features: {missing}{Colors.ENDC}")
        print(f"  {Colors.WARNING}⚠ Filling with zeros (may reduce accuracy){Colors.ENDC}")
        for feat in missing:
            df[feat] = 0
    else:
        print(f"  {Colors.OKGREEN}✓ All features present{Colors.ENDC}")

    # Prepare feature matrix
    X = df[expected_features]

    print(f"  • Running inference on {len(X):,} events...")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Add predictions
    df['probability'] = probabilities
    df['prediction'] = predictions

    # Confidence levels
    def classify_confidence(prob):
        if prob >= 0.7:
            return 'HIGH'
        elif prob >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'

    df['confidence_level'] = df['probability'].apply(classify_confidence)

    # Count results
    num_flagged = predictions.sum()
    high_conf = (df['probability'] >= 0.7).sum()
    medium_conf = ((df['probability'] >= 0.5) & (df['probability'] < 0.7)).sum()
    low_conf = (df['probability'] < 0.5).sum()

    print(f"  {Colors.OKGREEN}✓ Phase 3 Complete{Colors.ENDC}")
    print(f"    {Colors.FAIL}HIGH confidence: {high_conf:,} (≥0.7){Colors.ENDC}")
    print(f"    {Colors.WARNING}MEDIUM confidence: {medium_conf:,} (0.5-0.7){Colors.ENDC}")
    print(f"    {Colors.OKGREEN}LOW confidence: {low_conf:,} (<0.5){Colors.ENDC}")
    print(f"    Total Flagged (≥threshold): {num_flagged:,}")

    return df


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_outputs(df: pd.DataFrame, output_dir: str):
    """Generate output files."""
    print(f"\n{Colors.OKBLUE}[OUTPUT] Generating Results{Colors.ENDC}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Output columns
    output_cols = [
        'filepath', 'filename', 'eventtime', 'source',
        'lf_lsn', 'usn_usn', 'lf_event', 'usn_event_info',
        'probability', 'prediction', 'confidence_level'
    ]

    # 1. Full predictions
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    df[output_cols].to_csv(predictions_path, index=False)
    print(f"  ✓ {predictions_path} ({len(df):,} records)")

    # 1b. Debug output with all features (for analysis)
    debug_path = os.path.join(output_dir, 'predictions_with_features.csv')
    df.to_csv(debug_path, index=False)
    print(f"  ✓ {debug_path} ({len(df):,} records with all features)")

    # 2. Flagged events
    flagged = df[df['prediction'] == 1].copy()
    flagged = flagged.sort_values('probability', ascending=False)
    flagged_path = os.path.join(output_dir, 'flagged_files.csv')
    flagged[output_cols].to_csv(flagged_path, index=False)
    print(f"  ✓ {flagged_path} ({len(flagged):,} records)")

    # 3. Summary
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    total = len(df)
    flagged_count = len(flagged)
    high_conf = (df['confidence_level'] == 'HIGH').sum()
    medium_conf = (df['confidence_level'] == 'MEDIUM').sum()
    low_conf = (df['confidence_level'] == 'LOW').sum()

    report = f"""
{'='*70}
          TIMESTAMP MANIPULATION DETECTION REPORT
{'='*70}

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
                     SUMMARY STATISTICS
{'='*70}

Total Events Analyzed:          {total:,}

Confidence Level Breakdown:
  HIGH (≥0.7):                  {high_conf:,} ({high_conf/total*100:.2f}%)
  MEDIUM (0.5-0.7):             {medium_conf:,} ({medium_conf/total*100:.2f}%)
  LOW (<0.5):                   {low_conf:,} ({low_conf/total*100:.2f}%)

Total Flagged:                  {flagged_count:,} ({flagged_count/total*100:.2f}%)

Average Probability:            {df['probability'].mean():.4f}
Maximum Probability:            {df['probability'].max():.4f}

{'='*70}
                  TOP 10 FLAGGED FILES
{'='*70}

"""

    if len(flagged) > 0:
        for i, (idx, row) in enumerate(flagged.head(10).iterrows(), 1):
            report += f"{i}. {row['filepath']}\n"
            report += f"   Probability: {row['probability']:.4f} | Confidence: {row['confidence_level']}\n"
            report += f"   Source: {row['source']}\n\n"
    else:
        report += "No suspicious files detected.\n\n"

    report += f"""
{'='*70}
                        END OF REPORT
{'='*70}
"""

    with open(summary_path, 'w') as f:
        f.write(report)

    print(f"  ✓ {summary_path}")
    print(f"  {Colors.OKGREEN}✓ All outputs generated{Colors.ENDC}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print_banner()

    print(f"{Colors.BOLD}This tool follows the EXACT notebook methodology.{Colors.ENDC}\n")

    # Get inputs
    logfile_path = input(f"{Colors.BOLD}LogFile CSV Path: {Colors.ENDC}").strip()
    usnjrnl_path = input(f"{Colors.BOLD}UsnJrnl CSV Path: {Colors.ENDC}").strip()
    threshold = float(input(f"{Colors.BOLD}Detection Threshold [default: 0.5]: {Colors.ENDC}").strip() or "0.5")
    output_dir = input(f"{Colors.BOLD}Output Directory [default: ./results]: {Colors.ENDC}").strip() or "./results"

    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  LogFile:   {logfile_path}")
    print(f"  UsnJrnl:   {usnjrnl_path}")
    print(f"  Threshold: {threshold}")
    print(f"  Output:    {output_dir}")
    print(f"  Model:     models/xgboost_model.pkl")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

    proceed = input(f"{Colors.WARNING}Proceed with analysis? (y/n): {Colors.ENDC}").strip().lower()
    if proceed != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Run pipeline
    try:
        df = phase1_smart_union(logfile_path, usnjrnl_path)
        df = phase2_engineer_features(df)
        df = phase3_predict(df, 'models/xgboost_model.pkl', threshold)
        generate_outputs(df, output_dir)

        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ Analysis Complete!{Colors.ENDC}\n")

    except Exception as e:
        print(f"\n{Colors.FAIL}ERROR: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
