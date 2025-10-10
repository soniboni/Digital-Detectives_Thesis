#!/usr/bin/env python3
"""
=============================================================================
TIMESTOMPING DETECTION - FULL PIPELINE DEMO (OPTION 2)
=============================================================================

This script demonstrates the COMPLETE timestomping detection pipeline from
raw NTFS artifacts ($LogFile + $UsnJrnl CSVs) to final predictions.

USAGE:
------
    python full_pipeline_demo.py <logfile.csv> <usnjrnl.csv> [options]

REQUIRED INPUT:
---------------
    1. $LogFile CSV - Parsed $LogFile artifact from NTFS
       Required columns: eventtime, filename, filepath, lf_* columns

    2. $UsnJrnl CSV - Parsed $UsnJrnl artifact from NTFS
       Required columns: eventtime, filename, filepath, usn_* columns

    NOTE: Input files do NOT need to be labeled. The model will predict
          which files are likely timestomped.

EXPECTED OUTPUT:
----------------
    1. master_timeline.csv - Merged timeline from both artifacts
    2. features_engineered.csv - ML-ready features
    3. predictions.csv - All predictions with confidence scores
    4. flagged_files.csv - Only timestomped predictions (sorted by confidence)
    5. summary_report.txt - Detection summary and statistics
    6. pipeline_log.txt - Full pipeline execution log

EXAMPLE USAGE:
--------------
    # Basic usage
    python full_pipeline_demo.py LogFile.csv UsnJrnl.csv

    # Specify output directory
    python full_pipeline_demo.py LogFile.csv UsnJrnl.csv --output-dir ./case_001_results

    # Set custom confidence threshold
    python full_pipeline_demo.py LogFile.csv UsnJrnl.csv --threshold 0.5

    # Verbose mode
    python full_pipeline_demo.py LogFile.csv UsnJrnl.csv --verbose

PIPELINE STAGES:
----------------
    Stage 1: Load raw $LogFile and $UsnJrnl CSVs
    Stage 2: Create master timeline (merge artifacts)
    Stage 3: Feature engineering (extract ML features)
    Stage 4: Load trained model and make predictions
    Stage 5: Save results and generate report

REQUIREMENTS:
-------------
    - Trained model: data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib
    - Python packages: pandas, numpy, scikit-learn, joblib, imblearn (for SMOTE)
    - Input CSVs must follow standard $LogFile/$UsnJrnl parser output format

IMPORTANT NOTES:
----------------
    ⚠️ This script assumes input CSVs follow the format from Phase 1 parsers
    ⚠️ Large datasets (>1M events) may take 5-15 minutes to process
    ⚠️ Ensure sufficient memory (~2GB recommended for 1M events)

=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import math
import warnings

warnings.filterwarnings('ignore')

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

# =============================================================================
# STAGE 1: LOAD RAW ARTIFACTS
# =============================================================================

def load_raw_artifacts(logfile_path, usnjrnl_path, verbose=False):
    """
    Load raw $LogFile and $UsnJrnl CSV files

    Args:
        logfile_path: Path to $LogFile CSV
        usnjrnl_path: Path to $UsnJrnl CSV
        verbose: Print detailed information

    Returns:
        lf_df: LogFile DataFrame
        usn_df: UsnJrnl DataFrame
    """
    print_info(f"Loading $LogFile: {logfile_path}")

    if not Path(logfile_path).exists():
        print_error(f"$LogFile not found: {logfile_path}")
        sys.exit(1)

    lf_df = pd.read_csv(logfile_path, encoding='utf-8-sig')
    print_success(f"Loaded {len(lf_df):,} LogFile entries")

    if verbose:
        print_info(f"   Columns: {', '.join(lf_df.columns[:5].tolist())}... ({len(lf_df.columns)} total)")

    print_info(f"Loading $UsnJrnl: {usnjrnl_path}")

    if not Path(usnjrnl_path).exists():
        print_error(f"$UsnJrnl not found: {usnjrnl_path}")
        sys.exit(1)

    usn_df = pd.read_csv(usnjrnl_path, encoding='utf-8-sig')
    print_success(f"Loaded {len(usn_df):,} UsnJrnl entries")

    if verbose:
        print_info(f"   Columns: {', '.join(usn_df.columns[:5].tolist())}... ({len(usn_df.columns)} total)")

    return lf_df, usn_df

# =============================================================================
# STAGE 2: CREATE MASTER TIMELINE
# =============================================================================

def create_master_timeline(lf_df, usn_df, case_id=1, verbose=False):
    """
    Merge $LogFile and $UsnJrnl into master timeline

    Args:
        lf_df: LogFile DataFrame
        usn_df: UsnJrnl DataFrame
        case_id: Case identifier (default: 1)
        verbose: Print detailed information

    Returns:
        master_df: Merged master timeline
    """
    print_info("Creating master timeline...")

    # Add case_id
    lf_df['case_id'] = case_id
    usn_df['case_id'] = case_id

    # Standardize column names (prefix with lf_ and usn_)
    lf_df_prefixed = lf_df.copy()
    usn_df_prefixed = usn_df.copy()

    # Add merge columns
    lf_df_prefixed['merge_type'] = 'logfile_only'
    usn_df_prefixed['merge_type'] = 'usnjrnl_only'

    # Add default labels (unlabeled data)
    for df in [lf_df_prefixed, usn_df_prefixed]:
        df['is_timestomped'] = 0.0
        df['is_timestomped_lf'] = np.nan
        df['is_timestomped_usn'] = np.nan
        df['timestomp_tool_executed'] = np.nan
        df['timestomp_tool_executed_lf'] = np.nan
        df['timestomp_tool_executed_usn'] = np.nan
        df['suspicious_tool_name_lf'] = np.nan
        df['suspicious_tool_name_usn'] = np.nan
        df['suspicious_tool_name'] = np.nan
        df['label_source'] = np.nan
        df['label_source_lf'] = np.nan
        df['label_source_usn'] = np.nan

    # Simple concatenation (basic merge - can be improved)
    print_info("   Merging LogFile and UsnJrnl...")
    master_df = pd.concat([lf_df_prefixed, usn_df_prefixed], ignore_index=True)

    # Sort by eventtime
    master_df = master_df.sort_values('eventtime').reset_index(drop=True)

    print_success(f"Master timeline created: {len(master_df):,} events")

    if verbose:
        print_info(f"   LogFile events: {(master_df['merge_type'] == 'logfile_only').sum():,}")
        print_info(f"   UsnJrnl events: {(master_df['merge_type'] == 'usnjrnl_only').sum():,}")

    return master_df

# =============================================================================
# STAGE 3: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df, verbose=False):
    """
    Extract ML features from master timeline

    This replicates the feature engineering from Phase 2.

    Args:
        df: Master timeline DataFrame
        verbose: Print detailed information

    Returns:
        df_engineered: DataFrame with engineered features
    """
    print_info("Engineering features...")

    df_processed = df.copy()

    # 1. Convert eventtime to datetime
    if verbose:
        print_info("   Converting timestamps...")

    df_processed['eventtime_dt'] = pd.to_datetime(
        df_processed['eventtime'],
        format='%m/%d/%y %H:%M:%S',
        errors='coerce'
    )

    # 2. Temporal features
    if verbose:
        print_info("   Extracting temporal features...")

    df_processed['hour_of_day'] = df_processed['eventtime_dt'].dt.hour
    df_processed['day_of_week'] = df_processed['eventtime_dt'].dt.dayofweek
    df_processed['day_of_month'] = df_processed['eventtime_dt'].dt.day
    df_processed['month'] = df_processed['eventtime_dt'].dt.month
    df_processed['year'] = df_processed['eventtime_dt'].dt.year
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    df_processed['is_off_hours'] = ((df_processed['hour_of_day'] < 7) |
                                     (df_processed['hour_of_day'] >= 22)).astype(int)

    # 3. Time deltas
    if verbose:
        print_info("   Calculating time deltas...")

    df_processed = df_processed.sort_values(
        by=['case_id', 'filepath', 'eventtime_dt'],
        na_position='last'
    ).reset_index(drop=True)

    df_processed['prev_eventtime'] = df_processed.groupby(
        ['case_id', 'filepath'], dropna=False
    )['eventtime_dt'].shift(1)

    df_processed['time_delta_seconds'] = (
        df_processed['eventtime_dt'] - df_processed['prev_eventtime']
    ).dt.total_seconds().fillna(0)

    df_processed = df_processed.drop('prev_eventtime', axis=1)

    # 4. Event frequency
    if verbose:
        print_info("   Calculating event frequencies...")

    df_processed['events_per_file'] = df_processed.groupby(
        ['case_id', 'filepath'], dropna=False
    )['filepath'].transform('size')

    # Calculate events per minute
    file_stats = df_processed.groupby(['case_id', 'filepath'], dropna=False).agg({
        'eventtime_dt': ['min', 'max', 'count']
    }).reset_index()

    file_stats.columns = ['case_id', 'filepath', 'first_time', 'last_time', 'event_count']
    file_stats['timespan_minutes'] = (
        (file_stats['last_time'] - file_stats['first_time']).dt.total_seconds() / 60
    ).fillna(0)

    file_stats['events_per_minute'] = np.where(
        file_stats['timespan_minutes'] > 0,
        file_stats['event_count'] / file_stats['timespan_minutes'],
        file_stats['event_count']
    )

    df_processed = df_processed.merge(
        file_stats[['case_id', 'filepath', 'events_per_minute']],
        on=['case_id', 'filepath'],
        how='left'
    )

    df_processed['is_high_activity'] = (df_processed['events_per_minute'] > 10).astype(int)

    # 5. Timestamp anomalies
    if verbose:
        print_info("   Detecting timestamp anomalies...")

    # Convert MAC timestamps if they exist
    mac_cols = ['lf_creation_time', 'lf_modified_time', 'lf_mft_modified_time', 'lf_accessed_time']
    for col in mac_cols:
        if col in df_processed.columns:
            df_processed[f'{col}_dt'] = pd.to_datetime(df_processed[col], errors='coerce')

    # Anomaly detection
    if 'lf_creation_time_dt' in df_processed.columns and 'lf_modified_time_dt' in df_processed.columns:
        df_processed['creation_after_modification'] = (
            (df_processed['lf_creation_time_dt'] > df_processed['lf_modified_time_dt']) &
            df_processed['lf_creation_time_dt'].notna() &
            df_processed['lf_modified_time_dt'].notna()
        ).astype(int)
    else:
        df_processed['creation_after_modification'] = 0

    if 'lf_accessed_time_dt' in df_processed.columns and 'lf_creation_time_dt' in df_processed.columns:
        df_processed['accessed_before_creation'] = (
            (df_processed['lf_accessed_time_dt'] < df_processed['lf_creation_time_dt']) &
            df_processed['lf_accessed_time_dt'].notna() &
            df_processed['lf_creation_time_dt'].notna()
        ).astype(int)
    else:
        df_processed['accessed_before_creation'] = 0

    if all(col in df_processed.columns for col in ['lf_creation_time', 'lf_modified_time', 'lf_accessed_time']):
        df_processed['mac_all_identical'] = (
            (df_processed['lf_creation_time'] == df_processed['lf_modified_time']) &
            (df_processed['lf_modified_time'] == df_processed['lf_accessed_time']) &
            df_processed['lf_creation_time'].notna()
        ).astype(int)
    else:
        df_processed['mac_all_identical'] = 0

    # Future timestamps
    mac_time_cols = [col for col in ['lf_creation_time_dt', 'lf_modified_time_dt', 'lf_accessed_time_dt']
                     if col in df_processed.columns]

    if mac_time_cols:
        df_processed['has_future_timestamp'] = (
            df_processed[mac_time_cols].apply(lambda row: any(row > df_processed['eventtime_dt']), axis=1)
        ).astype(int)
    else:
        df_processed['has_future_timestamp'] = 0

    # Year deltas
    if 'lf_creation_time_dt' in df_processed.columns:
        df_processed['creation_year_delta'] = (
            df_processed['eventtime_dt'].dt.year - df_processed['lf_creation_time_dt'].dt.year
        ).abs()
    else:
        df_processed['creation_year_delta'] = 0

    if 'lf_modified_time_dt' in df_processed.columns:
        df_processed['modified_year_delta'] = (
            df_processed['eventtime_dt'].dt.year - df_processed['lf_modified_time_dt'].dt.year
        ).abs()
    else:
        df_processed['modified_year_delta'] = 0

    df_processed['nanosec_is_zero'] = 0  # Placeholder
    df_processed['missing_eventtime_flag'] = 0  # Placeholder

    # Clean up MAC datetime columns
    for col in mac_cols:
        if f'{col}_dt' in df_processed.columns:
            df_processed = df_processed.drop(f'{col}_dt', axis=1)

    # 6. Path features
    if verbose:
        print_info("   Extracting path features...")

    df_processed['path_depth'] = df_processed['filepath'].fillna('').str.count('\\\\')

    def is_system_path(path):
        if pd.isna(path):
            return 0
        path_lower = str(path).lower()
        return int(any(ind in path_lower for ind in ['\\windows\\', '\\system32\\', '\\program files\\', '\\syswow64\\']))

    def is_temp_path(path):
        if pd.isna(path):
            return 0
        path_lower = str(path).lower()
        return int(any(ind in path_lower for ind in ['\\temp\\', '\\tmp\\', '\\appdata\\local\\temp', '\\cache\\']))

    def is_user_path(path):
        if pd.isna(path):
            return 0
        return int('\\users\\' in str(path).lower())

    df_processed['is_system_path'] = df_processed['filepath'].apply(is_system_path)
    df_processed['is_temp_path'] = df_processed['filepath'].apply(is_temp_path)
    df_processed['is_user_path'] = df_processed['filepath'].apply(is_user_path)

    df_processed['filename_length'] = df_processed['filename'].fillna('').astype(str).str.len()

    df_processed['file_extension'] = df_processed['filename'].fillna('').astype(str).str.extract(r'\.([^.]+)$')[0].fillna('none')

    suspicious_exts = ['exe', 'dll', 'sys', 'bat', 'cmd', 'ps1', 'vbs', 'js']
    df_processed['is_executable'] = df_processed['file_extension'].str.lower().isin(suspicious_exts).astype(int)

    # Path entropy
    def calculate_entropy(text):
        if pd.isna(text) or len(str(text)) == 0:
            return 0
        text = str(text)
        counter = Counter(text)
        length = len(text)
        entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
        return entropy

    df_processed['path_entropy'] = df_processed['filepath'].apply(calculate_entropy)
    df_processed['filename_entropy'] = df_processed['filename'].apply(calculate_entropy)

    # 7. Event pattern features
    if verbose:
        print_info("   Encoding event patterns...")

    # Event encoding
    from sklearn.preprocessing import LabelEncoder

    le_lf = LabelEncoder()
    le_usn = LabelEncoder()

    if 'lf_event' in df_processed.columns:
        df_processed['lf_event_encoded'] = le_lf.fit_transform(df_processed['lf_event'].fillna('unknown'))
    else:
        df_processed['lf_event_encoded'] = 0

    if 'usn_event_info' in df_processed.columns:
        df_processed['usn_event_encoded'] = le_usn.fit_transform(df_processed['usn_event_info'].fillna('unknown'))
    else:
        df_processed['usn_event_encoded'] = 0

    # Rare events (using simplified threshold)
    if 'lf_event' in df_processed.columns:
        lf_event_counts = df_processed['lf_event'].value_counts()
        rare_threshold = len(df_processed) * 0.001
        lf_rare_events = set(lf_event_counts[lf_event_counts < rare_threshold].index)
        df_processed['is_rare_lf_event'] = df_processed['lf_event'].isin(lf_rare_events).astype(int)
    else:
        df_processed['is_rare_lf_event'] = 0

    if 'usn_event_info' in df_processed.columns:
        usn_event_counts = df_processed['usn_event_info'].value_counts()
        rare_threshold = len(df_processed) * 0.001
        usn_rare_events = set(usn_event_counts[usn_event_counts < rare_threshold].index)
        df_processed['is_rare_usn_event'] = df_processed['usn_event_info'].isin(usn_rare_events).astype(int)
    else:
        df_processed['is_rare_usn_event'] = 0

    df_processed['event_count_per_file'] = df_processed.groupby(
        ['case_id', 'filepath']
    )['filepath'].transform('count')

    if 'lf_event' in df_processed.columns:
        df_processed['prev_lf_event'] = df_processed.groupby(['case_id', 'filepath'])['lf_event'].shift(1)
        df_processed['is_consecutive_same_event'] = (
            df_processed['lf_event'] == df_processed['prev_lf_event']
        ).astype(int)
        df_processed = df_processed.drop('prev_lf_event', axis=1)
    else:
        df_processed['is_consecutive_same_event'] = 0

    # 8. One-hot encode categorical columns
    if verbose:
        print_info("   One-hot encoding categorical features...")

    # usn_file_attribute
    if 'usn_file_attribute' in df_processed.columns:
        usn_attr_dummies = pd.get_dummies(df_processed['usn_file_attribute'], prefix='usn_attr', dummy_na=True)
        df_processed = pd.concat([df_processed, usn_attr_dummies], axis=1)

    # 9. Cross-artifact features
    if verbose:
        print_info("   Creating cross-artifact features...")

    # Merge type one-hot
    merge_type_dummies = pd.get_dummies(df_processed['merge_type'], prefix='merge')
    df_processed = pd.concat([df_processed, merge_type_dummies], axis=1)

    # Artifact presence flags
    df_processed['has_logfile_data'] = df_processed.get('lf_lsn', pd.Series([np.nan]*len(df_processed))).notna().astype(int)
    df_processed['has_usnjrnl_data'] = df_processed.get('usn_usn', pd.Series([np.nan]*len(df_processed))).notna().astype(int)
    df_processed['has_both_artifacts'] = (
        (df_processed['has_logfile_data'] == 1) & (df_processed['has_usnjrnl_data'] == 1)
    ).astype(int)

    # Label source (for compatibility)
    label_source_dummies = pd.get_dummies(df_processed['label_source'], prefix='label_source', dummy_na=True)
    df_processed = pd.concat([df_processed, label_source_dummies], axis=1)

    print_success(f"Feature engineering complete: {len(df_processed.columns)} columns")

    return df_processed

# =============================================================================
# STAGE 4: MAKE PREDICTIONS
# =============================================================================

def make_predictions(df_engineered, model_path, threshold=0.3, verbose=False):
    """
    Load model and make predictions

    Args:
        df_engineered: DataFrame with engineered features
        model_path: Path to trained model
        threshold: Confidence threshold for flagging
        verbose: Print detailed information

    Returns:
        predictions: Binary predictions
        probabilities: Confidence scores
        risk_levels: Risk categorization
    """
    print_info(f"Loading model: {model_path}")

    if not Path(model_path).exists():
        print_error(f"Model not found: {model_path}")
        sys.exit(1)

    model = joblib.load(model_path)
    print_success("Model loaded successfully")

    # Prepare features
    print_info("Preparing features for prediction...")

    cols_to_exclude = [
        'is_timestomped', 'is_timestomped_lf', 'is_timestomped_usn',
        'timestomp_tool_executed', 'timestomp_tool_executed_lf', 'timestomp_tool_executed_usn',
        'case_id', 'eventtime_dt', 'label_source_both', 'label_source_logfile',
        'label_source_usnjrnl', 'label_source_nan',
        'eventtime', 'filename', 'filepath', 'merge_type', 'label_source',
        'file_extension', 'lf_event', 'usn_event_info', 'usn_file_attribute',
        'suspicious_tool_name_lf', 'suspicious_tool_name_usn', 'suspicious_tool_name',
        'lf_creation_time', 'lf_modified_time', 'lf_mft_modified_time', 'lf_accessed_time',
        'lf_detail', 'lf_redo', 'lf_target_vcn', 'lf_cluster_index',
        'usn_file_reference_number', 'usn_parent_file_reference_number',
        'lf_lsn', 'usn_usn',
    ]

    feature_cols = [col for col in df_engineered.columns if col not in cols_to_exclude]

    X = df_engineered[feature_cols].copy()

    # Convert bool to int
    bool_cols = X.select_dtypes(include='bool').columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    X = X.fillna(0)

    print_info(f"   Feature matrix: {X.shape}")
    print_info(f"   Model expects: {model.n_features_in_} features")

    # Make predictions
    print_info("Generating predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Risk levels
    risk_levels = []
    for prob in probabilities:
        if prob < threshold:
            risk_levels.append('LOW')
        elif prob < 0.7:
            risk_levels.append('MEDIUM')
        else:
            risk_levels.append('HIGH')

    print_success(f"Predictions complete!")
    print_info(f"   Flagged as timestomped: {predictions.sum():,} ({predictions.sum()/len(predictions)*100:.2f}%)")
    print_info(f"   High risk: {risk_levels.count('HIGH'):,}")
    print_info(f"   Medium risk: {risk_levels.count('MEDIUM'):,}")

    return predictions, probabilities, risk_levels

# =============================================================================
# STAGE 5: SAVE RESULTS
# =============================================================================

def save_results(df_original, df_engineered, predictions, probabilities, risk_levels,
                 output_dir, save_intermediates=False, verbose=False):
    """
    Save prediction results

    Args:
        df_original: Original master timeline
        df_engineered: Engineered features
        predictions: Binary predictions
        probabilities: Confidence scores
        risk_levels: Risk categorization
        output_dir: Output directory
        save_intermediates: Save intermediate files (master timeline, features)
        verbose: Print detailed information

    Returns:
        output_paths: Dictionary of output paths
    """
    print_info(f"Saving results to: {output_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save intermediate files if requested
    if save_intermediates:
        if verbose:
            print_info("   Saving intermediate files...")

        master_path = output_dir / 'master_timeline.csv'
        df_original.to_csv(master_path, index=False)
        print_success(f"Saved master timeline: {master_path}")

        features_path = output_dir / 'features_engineered.csv'
        df_engineered.to_csv(features_path, index=False)
        print_success(f"Saved engineered features: {features_path}")

    # Create results DataFrame
    results_df = df_original.copy()
    results_df['prediction'] = predictions
    results_df['confidence'] = probabilities
    results_df['risk_level'] = risk_levels

    # Save predictions
    predictions_path = output_dir / 'predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    print_success(f"Saved all predictions: {predictions_path}")

    # Save flagged files
    flagged_df = results_df[results_df['prediction'] == 1].copy()
    flagged_df = flagged_df.sort_values('confidence', ascending=False)
    flagged_path = output_dir / 'flagged_files.csv'
    flagged_df.to_csv(flagged_path, index=False)
    print_success(f"Saved flagged files: {flagged_path} ({len(flagged_df):,} files)")

    # Generate summary report
    summary_path = output_dir / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TIMESTOMPING DETECTION - FULL PIPELINE SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Random Forest (v3 Final - Minimal SMOTE)\n\n")

        f.write("INPUT:\n")
        f.write(f"  Total events analyzed: {len(results_df):,}\n\n")

        f.write("PREDICTIONS:\n")
        f.write(f"  Flagged as timestomped: {predictions.sum():,} ({predictions.sum()/len(predictions)*100:.2f}%)\n")
        f.write(f"  Predicted benign: {(predictions == 0).sum():,}\n\n")

        f.write("RISK BREAKDOWN:\n")
        f.write(f"  HIGH risk (≥70% confidence): {risk_levels.count('HIGH'):,}\n")
        f.write(f"  MEDIUM risk (30-70% confidence): {risk_levels.count('MEDIUM'):,}\n")
        f.write(f"  LOW risk (<30% confidence): {risk_levels.count('LOW'):,}\n\n")

        if len(flagged_df) > 0:
            f.write("TOP 10 HIGHEST CONFIDENCE DETECTIONS:\n")
            f.write("-"*80 + "\n")
            for idx, row in flagged_df.head(10).iterrows():
                filename = row.get('filename', 'N/A')
                f.write(f"  {row['confidence']:.3f} | {row['risk_level']:6s} | {filename}\n")
            f.write("\n")

        f.write("FORENSIC TRIAGE VALUE:\n")
        f.write(f"  Files requiring investigation: {len(flagged_df):,}\n")
        f.write(f"  Investigation reduction: {(1 - len(flagged_df)/len(results_df))*100:.2f}%\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  1. predictions.csv - All predictions with confidence scores\n")
        f.write(f"  2. flagged_files.csv - Only timestomped predictions\n")
        f.write(f"  3. summary_report.txt - This report\n")
        if save_intermediates:
            f.write(f"  4. master_timeline.csv - Merged timeline\n")
            f.write(f"  5. features_engineered.csv - ML features\n")
        f.write("\n")

        f.write("="*80 + "\n")

    print_success(f"Saved summary report: {summary_path}")

    return {
        'predictions': predictions_path,
        'flagged': flagged_path,
        'summary': summary_path
    }

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description='Timestomping Detection - Full Pipeline Demo (Option 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('logfile_csv', type=str,
                        help='Path to $LogFile CSV')
    parser.add_argument('usnjrnl_csv', type=str,
                        help='Path to $UsnJrnl CSV')
    parser.add_argument('--output-dir', type=str, default='./full_pipeline_results',
                        help='Directory to save results (default: ./full_pipeline_results)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Confidence threshold for flagging (default: 0.3)')
    parser.add_argument('--case-id', type=int, default=1,
                        help='Case identifier (default: 1)')
    parser.add_argument('--model-path', type=str,
                        default='../data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib',
                        help='Path to trained model')
    parser.add_argument('--save-intermediates', action='store_true',
                        help='Save intermediate files (master timeline, features)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress')

    args = parser.parse_args()

    # Print header
    print_header("TIMESTOMPING DETECTION - FULL PIPELINE DEMO")
    print(f"LogFile: {args.logfile_csv}")
    print(f"UsnJrnl: {args.usnjrnl_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Case ID: {args.case_id}")
    print(f"Threshold: {args.threshold}\n")

    try:
        # Stage 1: Load raw artifacts
        print_header("STAGE 1: LOAD RAW ARTIFACTS")
        lf_df, usn_df = load_raw_artifacts(args.logfile_csv, args.usnjrnl_csv, verbose=args.verbose)

        # Stage 2: Create master timeline
        print_header("STAGE 2: CREATE MASTER TIMELINE")
        master_df = create_master_timeline(lf_df, usn_df, case_id=args.case_id, verbose=args.verbose)

        # Stage 3: Feature engineering
        print_header("STAGE 3: FEATURE ENGINEERING")
        df_engineered = engineer_features(master_df, verbose=args.verbose)

        # Stage 4: Make predictions
        print_header("STAGE 4: MAKE PREDICTIONS")
        predictions, probabilities, risk_levels = make_predictions(
            df_engineered, args.model_path, threshold=args.threshold, verbose=args.verbose
        )

        # Stage 5: Save results
        print_header("STAGE 5: SAVE RESULTS")
        output_paths = save_results(
            master_df, df_engineered, predictions, probabilities, risk_levels,
            args.output_dir, save_intermediates=args.save_intermediates, verbose=args.verbose
        )

        # Final summary
        print_header("PIPELINE COMPLETE!")
        print_success("Timestomping detection pipeline completed successfully!")
        print(f"\n{Colors.BOLD}Output Files:{Colors.ENDC}")
        print(f"  📄 All predictions: {output_paths['predictions']}")
        print(f"  🚨 Flagged files: {output_paths['flagged']}")
        print(f"  📊 Summary report: {output_paths['summary']}")

        flagged_count = predictions.sum()
        if flagged_count > 0:
            print(f"\n{Colors.WARNING}{Colors.BOLD}⚠️  {flagged_count:,} files flagged for forensic investigation!{Colors.ENDC}")
            print(f"{Colors.WARNING}   Review {output_paths['flagged']} for details.{Colors.ENDC}\n")
        else:
            print(f"\n{Colors.OKGREEN}✓ No timestomped files detected.{Colors.ENDC}\n")

    except Exception as e:
        print_error(f"Pipeline error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()