# -*- coding: utf-8 -*-

"""
Feature Engineering Module for NTFS Timestomping Detection

This module provides feature engineering functionality for grouped events.
It transforms grouped_events.csv (output from data_preprocessing.py) into forensic features
suitable for machine learning inference.

Coding style and logic adapted from feature_engineering.txt (Phase 3: Batch Feature Engineering).
Single-dataset (non-batch) processing model compatible with Autopsy plugin.

Environment: Python 3.x
Dependencies: pandas, numpy, pathlib, logging

Pipeline Flow (from ProjectOverview.txt):
    Stage 3 (Data Preprocessing) Output -> Stage 4 (This Module) -> Stage 5 (ML Inference)
    
    Input:  "Grouped Events File" directory containing:
            - grouped_events.csv
            
    Output: "File Features" directory containing:
            - file_features.csv

Directory Structure (from ntfs_timestomping_detector.py):
    NTFS Timestomping Detector/
    ├── Exported NTFS Files/      # Stage 1: Raw binary NTFS files
    ├── Parsed Files/             # Stage 2: Parser output
    ├── Grouped Events File/      # Stage 3: Preprocessing output (INPUT for this module)
    ├── File Features/            # Stage 4: This module's output
    └── Detection Results/        # Stage 5: ML inference output
"""

import pandas as pd
import numpy as np
import logging
import warnings

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Orchestrator class for feature engineering.
    
    Coordinates feature extraction from grouped events.
    
    Directory Structure:
        Input:  grouped_events_dir_path = "Grouped Events File"
        Output: output_dir_path = "File Features"
    
    Usage:
        engineer = FeatureEngineer()
        results = engineer.extract_features(
            grouped_events_dir=Path("path/to/Grouped Events File"),
            output_dir=Path("path/to/File Features")
        )
    """
    
    def __init__(self, logger_obj=None):
        """
        Initialize the feature engineer.
        
        Args:
            logger_obj: Optional logger object. If not provided, uses module logger.
        """
        self.logger = logger_obj if logger_obj else logger
    
    def extract_features(self, grouped_events_dir, output_dir):
        """
        Execute the complete feature engineering pipeline.
        
        Args:
            grouped_events_dir: Input directory containing grouped_events.csv
            output_dir: Output directory for file_features.csv
            
        Returns:
            Dict with processing results
        """
        grouped_events_dir = Path(grouped_events_dir)
        output_dir = Path(output_dir)
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: FEATURE ENGINEERING - File Features Extraction")
        self.logger.info("=" * 80)
        self.logger.info("Input directory:  {}".format(grouped_events_dir))
        self.logger.info("Output directory: {}".format(output_dir))
        
        try:
            # Load grouped events
            self.logger.info("\n[Step 1/5] Loading grouped events...")
            grouped_events_file = grouped_events_dir / 'grouped_events.csv'
            if not grouped_events_file.exists():
                self.logger.error("Grouped events file not found: {}".format(grouped_events_file))
                return {
                    'success': False,
                    'message': 'Grouped events file not found',
                    'file_features_csv': None,
                    'file_count': 0
                }
            
            df_events = pd.read_csv(grouped_events_file)
            self.logger.info("  Loaded {} events".format(len(df_events)))
            
            # Compute per-event indicators
            self.logger.info("\n[Step 2/5] Computing per-event indicators...")
            df_events = compute_event_indicators(df_events)
            
            # Get file metadata
            self.logger.info("  Extracting file metadata...")
            file_metadata = df_events.groupby('FileFRN').agg({
                'dataID': 'first',
                'FileName': 'first',
                'FilePath': 'first'
            }).reset_index()
            
            # Aggregate all feature categories
            self.logger.info("\n[Step 3/5] Aggregating feature categories...")
            timestamp_features = df_events.groupby('FileFRN').apply(aggregate_timestamp_features).reset_index()
            structural_features = df_events.groupby('FileFRN').apply(aggregate_structural_features).reset_index()
            cross_artifact_features = df_events.groupby('FileFRN').apply(aggregate_cross_artifact_features).reset_index()
            temporal_features = df_events.groupby('FileFRN').apply(aggregate_temporal_features).reset_index()
            
            # Merge all features
            self.logger.info("\n[Step 4/5] Merging feature sets...")
            df_features = file_metadata.copy()
            df_features = df_features.merge(timestamp_features, on='FileFRN', how='left')
            df_features = df_features.merge(structural_features, on='FileFRN', how='left')
            df_features = df_features.merge(cross_artifact_features, on='FileFRN', how='left')
            df_features = df_features.merge(temporal_features, on='FileFRN', how='left')
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export features
            output_file = output_dir / 'file_features.csv'
            
            # Convert boolean columns to int for CSV export
            for col in df_features.columns:
                if df_features[col].dtype == 'bool':
                    df_features[col] = df_features[col].astype(int)
            
            # Fill NaN values for numeric columns
            numeric_cols = df_features.select_dtypes(include=['float64', 'int64']).columns
            df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
            
            df_features.to_csv(output_file, index=False, encoding='utf-8')
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STAGE 3 COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info("✓ File features exported to: {}".format(output_file))
            self.logger.info("  Total files: {:,}".format(len(df_features)))
            self.logger.info("  Total features: {:,}".format(len(df_features.columns)))
            self.logger.info("=" * 80)
            
            return {
                'success': True,
                'message': 'Successfully extracted features for {} files'.format(len(df_features)),
                'file_features_csv': str(output_file),
                'file_count': len(df_features)
            }
        
        except Exception as e:
            self.logger.error("Error during feature engineering: {}".format(str(e)), exc_info=True)
            return {
                'success': False,
                'message': 'Feature engineering failed: {}'.format(str(e)),
                'file_features_csv': None,
                'file_count': 0
            }

# =============================================================================
# TIMESTAMP ANALYSIS HELPER FUNCTIONS
# =============================================================================

def compute_timestamp_jump(undo_ts, redo_ts):
    """
    Compute time delta between Undo (before) and Redo (after) timestamps.
    Returns delta in seconds (negative = backward jump).
    
    Reference: Oh et al. Algorithm 3 - Checking Timestamp Changes
    """
    if pd.isna(undo_ts) or pd.isna(redo_ts):
        return np.nan
    return (redo_ts - undo_ts).total_seconds()


def has_zero_nanoseconds(timestamp):
    """
    Check if timestamp has zero nanoseconds (microseconds in pandas).
    Many timestomping tools produce zero nanoseconds.
    """
    if pd.isna(timestamp):
        return False
    return timestamp.microsecond == 0


def is_backward_jump(undo_ts, redo_ts):
    """
    Detect backward timestamp jump (moved to past).
    Algorithm 3: event.before_$SI-M > event.after_$SI-M
    """
    if pd.isna(undo_ts) or pd.isna(redo_ts):
        return False
    return redo_ts < undo_ts


def is_forward_jump(undo_ts, redo_ts):
    """
    Detect forward timestamp jump (moved to future).
    """
    if pd.isna(undo_ts) or pd.isna(redo_ts):
        return False
    return redo_ts > undo_ts


# =============================================================================
# PER-EVENT INDICATOR COMPUTATION
# =============================================================================

def compute_event_indicators(df):
    """
    Compute per-event indicators for feature aggregation.
    
    Parameters:
        df: DataFrame with grouped events from preprocessing stage
    
    Returns:
        DataFrame with additional indicator columns
    """
    df = df.copy()
    
    # Parse timestamp columns to datetime if needed
    ts_cols = ['EventTimestamp', 'Undo_$SI-C', 'Undo_$SI-M', 'Undo_$SI-E', 'Undo_$SI-A',
               'Redo_$SI-C', 'Redo_$SI-M', 'Redo_$SI-E', 'Redo_$SI-A']
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Event source flags
    df['is_logfile_event'] = df['EventSource'] == 'LogFile'
    df['is_usnjrnl_event'] = df['EventSource'] == 'UsnJrnl'
    
    # Ensure IsTimestampChange is boolean
    df['IsTimestampChange'] = df['IsTimestampChange'].fillna(False).astype(bool)
    
    # Backward/forward jumps using $SI-M (Modified time)
    df['backward_jump_M'] = df.apply(
        lambda row: is_backward_jump(row.get('Undo_$SI-M'), row.get('Redo_$SI-M'))
        if row['IsTimestampChange'] else False, axis=1
    )
    
    df['forward_jump_M'] = df.apply(
        lambda row: is_forward_jump(row.get('Undo_$SI-M'), row.get('Redo_$SI-M'))
        if row['IsTimestampChange'] else False, axis=1
    )
    
    # Jump magnitude in seconds
    df['jump_seconds_M'] = df.apply(
        lambda row: compute_timestamp_jump(row.get('Undo_$SI-M'), row.get('Redo_$SI-M'))
        if row['IsTimestampChange'] else np.nan, axis=1
    )
    
    # Creation time changes (Algorithm 3)
    df['creation_changed'] = df.apply(
        lambda row: (pd.notna(row.get('Undo_$SI-C')) and 
                    pd.notna(row.get('Redo_$SI-C')) and 
                    row['Undo_$SI-C'] != row['Redo_$SI-C'])
        if row['IsTimestampChange'] else False, axis=1
    )
    
    # Zero nanoseconds detection
    df['redo_zero_ns_C'] = df['Redo_$SI-C'].apply(has_zero_nanoseconds)
    df['redo_zero_ns_M'] = df['Redo_$SI-M'].apply(has_zero_nanoseconds)
    df['redo_zero_ns_E'] = df['Redo_$SI-E'].apply(has_zero_nanoseconds)
    df['redo_zero_ns_A'] = df['Redo_$SI-A'].apply(has_zero_nanoseconds)
    
    df['has_redo_zero_ns'] = (df['redo_zero_ns_C'] | df['redo_zero_ns_M'] | 
                              df['redo_zero_ns_E'] | df['redo_zero_ns_A'])
    
    return df


# =============================================================================
# CATEGORY A: TIMESTAMP CHANGE FEATURE AGGREGATION
# =============================================================================

def aggregate_timestamp_features(group):
    """
    Aggregate timestamp change features for a single file.
    Reference: Oh et al. Algorithm 3
    """
    ts_events = group[group['IsTimestampChange'] == True]
    total_events = len(group)
    
    # A.1 Count of timestamp changes
    num_ts_changes = len(ts_events)
    
    # A.2 Backward jumps (Algorithm 3: before > after)
    num_backward = group['backward_jump_M'].sum()
    
    # A.3 Forward jumps
    num_forward = group['forward_jump_M'].sum()
    
    # A.4 Creation time changes
    num_creation = group['creation_changed'].sum()
    
    # A.5 Max backward jump magnitude
    backward_jumps = group.loc[group['backward_jump_M'], 'jump_seconds_M']
    max_backward = abs(backward_jumps.min()) if len(backward_jumps) > 0 else 0
    
    # A.6 Mean jump magnitude
    all_jumps = group['jump_seconds_M'].dropna()
    mean_jump = all_jumps.abs().mean() if len(all_jumps) > 0 else 0
    
    # A.7 Timestamp change density
    density = num_ts_changes / total_events if total_events > 0 else 0
    
    # A.8 Zero nanoseconds count
    num_zero_ns = group['has_redo_zero_ns'].sum()
    
    return pd.Series({
        'num_timestamp_changes': num_ts_changes,
        'num_backward_jumps': num_backward,
        'num_forward_jumps': num_forward,
        'num_creation_changes': num_creation,
        'max_backward_jump_seconds': max_backward,
        'mean_jump_seconds': mean_jump,
        'timestamp_change_density': density,
        'num_zero_nanosecond_events': num_zero_ns
    })


# =============================================================================
# CATEGORY B: STRUCTURAL NTFS FEATURE AGGREGATION
# =============================================================================

def aggregate_structural_features(group):
    """
    Aggregate structural NTFS pattern features.
    Reference: Oh et al. Algorithms 1-2
    """
    logfile_events = group[group['is_logfile_event']]
    ts_events = group[group['IsTimestampChange'] == True]
    
    # B.1 Only $SI modified (attribute offset 0x18-0x30 = $SI timestamps)
    if len(logfile_events) > 0:
        attr_offsets = logfile_events['AttributeOffset'].dropna()
        only_si = attr_offsets.isin([24, 32, 40, 48]).all() if len(attr_offsets) > 0 else False
    else:
        only_si = False
    
    # B.2 UpdateResidentValue operations (Algorithm 1: redo.op == 0x7)
    if 'RedoOPName' in logfile_events.columns:
        urv_count = (logfile_events['RedoOPName'] == 'UpdateResidentValue').sum()
    else:
        urv_count = 0
    
    # B.3 Repeated UpdateResidentValue
    repeated_urv = urv_count > 1
    
    # B.4 Consecutive timestamp changes
    consecutive = False
    if len(ts_events) >= 2:
        ts_lsns = ts_events['LSN'].dropna().sort_values()
        if len(ts_lsns) >= 2:
            lsn_diffs = ts_lsns.diff().dropna()
            consecutive = (lsn_diffs < 1000).any()
    
    # B.5 Total LogFile events
    num_logfile = len(logfile_events)
    
    return pd.Series({
        'only_SI_modified': only_si,
        'num_update_resident_value': urv_count,
        'repeated_update_resident_value': repeated_urv,
        'consecutive_timestamp_changes': consecutive,
        'num_logfile_events': num_logfile
    })


# =============================================================================
# CATEGORY C: CROSS-ARTIFACT FEATURE AGGREGATION
# =============================================================================

def aggregate_cross_artifact_features(group):
    """
    Aggregate cross-artifact consistency features.
    Reference: Oh et al. Algorithms 5-7
    """
    ts_change_events = group[group['IsTimestampChange'] == True]
    basic_info_events = group[group['HasBasicInfoChange'] == True]
    close_events = group[group['HasClose'] == True]
    create_events = group[group['HasFileCreate'] == True]
    usnjrnl_events = group[group['is_usnjrnl_event']]
    
    # C.1-4 Event type flags
    has_ts_change = len(ts_change_events) > 0
    has_basic_info = len(basic_info_events) > 0
    has_close = len(close_events) > 0
    has_create = len(create_events) > 0
    
    # C.5 Counts
    num_basic_info = len(basic_info_events)
    num_close = len(close_events)
    num_create = len(create_events)
    
    # C.6 LogFile-USN mismatch (timestamp change but no BASIC_INFO_CHANGE)
    mismatch = has_ts_change and not has_basic_info
    
    # C.7 USN basic detection pattern (Algorithm 5)
    has_basic_pattern = has_basic_info and has_close
    
    # C.8 Total USN events
    num_usnjrnl = len(usnjrnl_events)
    
    return pd.Series({
        'has_logfile_ts_change': has_ts_change,
        'has_usn_basic_info': has_basic_info,
        'has_usn_close': has_close,
        'has_usn_file_create': has_create,
        'num_usn_basic_info': num_basic_info,
        'num_usn_close': num_close,
        'num_usn_file_create': num_create,
        'logfile_usn_mismatch': mismatch,
        'has_usn_basic_pattern': has_basic_pattern,
        'num_usnjrnl_events': num_usnjrnl
    })


# =============================================================================
# CATEGORY D: TEMPORAL BEHAVIOR FEATURE AGGREGATION
# =============================================================================

def aggregate_temporal_features(group):
    """
    Aggregate temporal behavior features.
    """
    sorted_group = group.sort_values('EventTimestamp')
    timestamps = sorted_group['EventTimestamp'].dropna()
    
    features = {
        'min_inter_event_delta': np.nan,
        'max_inter_event_delta': np.nan,
        'mean_inter_event_delta': np.nan,
        'burstiness_score': 0.0,
        'event_time_span_seconds': np.nan,
        'total_events': len(group)
    }
    
    if len(timestamps) >= 2:
        deltas = timestamps.diff().dropna()
        delta_seconds = deltas.dt.total_seconds()
        
        features['min_inter_event_delta'] = delta_seconds.min()
        features['max_inter_event_delta'] = delta_seconds.max()
        features['mean_inter_event_delta'] = delta_seconds.mean()
        features['event_time_span_seconds'] = (timestamps.max() - timestamps.min()).total_seconds()
        
        # Burstiness: proportion of events within 1 second of each other
        burst_count = (delta_seconds <= 1.0).sum()
        features['burstiness_score'] = burst_count / len(delta_seconds) if len(delta_seconds) > 0 else 0
    
    return pd.Series(features)


# =============================================================================
# MODULE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Direct execution entry point for testing.
    
    Usage:
        python feature_engineering.py <grouped_events_dir> <output_dir>
        
    Example:
        python feature_engineering.py "Grouped Events File" "File Features"
    """
    import sys
    
    # Configure logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) >= 3:
        grouped_events_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
        
        fe = FeatureEngineer()
        results = fe.extract_features(grouped_events_dir, output_dir)
        
        print("\nResults:")
        print("  Success: {}".format(results['success']))
        print("  Message: {}".format(results['message']))
        if results['file_features_csv']:
            print("  Output: {}".format(results['file_features_csv']))
            print("  Files: {}".format(results['file_count']))
        
        sys.exit(0 if results['success'] else 1)
    else:
        print("Feature Engineering Module for NTFS Timestomping Detection")
        print("\nUsage: python feature_engineering.py <grouped_events_dir> <output_dir>")
        sys.exit(0)