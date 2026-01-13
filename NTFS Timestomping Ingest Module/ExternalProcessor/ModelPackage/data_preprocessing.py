# -*- coding: utf-8 -*-

"""
Data Preprocessing Module for NTFS Timestomping Detection

This module provides preprocessing functionality for parsed NTFS system files.
It transforms raw CSV outputs from the Parser stage (MFT_parsed.csv, LogFile_parsed.csv, 
UsnJrnl_parsed.csv) into a unified grouped events CSV for feature engineering.

Coding style and logic adapted from data_preprocessing.txt (Phase 2: Batch Data Preprocessing).
Single-dataset (non-batch) processing model compatible with Autopsy plugin.

Environment: Python 3.x
Dependencies: pandas, numpy, pathlib, logging

Pipeline Flow (from ProjectOverview.txt):
    Stage 2 (Parser) Output -> Stage 3 (This Module) -> Stage 4 (Feature Engineering)
    
    Input:  "Parsed Files" directory containing:
            - MFT_parsed.csv
            - LogFile_parsed.csv  
            - UsnJrnl_parsed.csv
            
    Output: "Grouped Events File" directory containing:
            - grouped_events.csv

Directory Structure (from ntfs_timestomping_detector.py):
    NTFS Timestomping Detector/
    ├── Exported NTFS Files/      # Stage 1: Raw binary NTFS files
    ├── Parsed Files/             # Stage 2: Parser output (INPUT for this module)
    ├── Grouped Events File/      # Stage 3: This module's output
    ├── File Features/            # Stage 4: Feature engineering output
    └── Detection Results/        # Stage 5: ML inference output
"""

import pandas as pd
import numpy as np
import logging
import warnings

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Orchestrator class for data preprocessing.
    
    Coordinates the preprocessing of parsed NTFS files and exports grouped events.
    
    Directory Structure:
        Input:  parsed_dir_path = "Parsed Files" (from ntfs_timestomping_detector.py)
        Output: grouped_events_dir_path = "Grouped Events File"
    
    Usage:
        preprocessor = DataPreprocessor()
        results = preprocessor.preprocess_all(
            parsed_files_dir=Path("path/to/Parsed Files"),
            output_dir=Path("path/to/Grouped Events File")
        )
    """
    
    def __init__(self, logger_obj=None):
        """
        Initialize the preprocessor.
        
        Args:
            logger_obj: Optional logger object. If not provided, uses module logger.
        """
        self.logger = logger_obj if logger_obj else logger
    
    def preprocess_all(self, parsed_files_dir, output_dir):
        """
        Execute the complete preprocessing pipeline.
        
        This method orchestrates all preprocessing steps and outputs a single
        grouped_events.csv file containing all forensic events sorted by file.
        
        Args:
            parsed_files_dir: Input directory containing parsed CSV files from Parser stage
                             (maps to "Parsed Files" directory from ntfs_timestomping_detector.py)
            output_dir: Output directory for grouped_events.csv
                       (maps to "Grouped Events File" directory from ntfs_timestomping_detector.py)
            
        Returns:
            Dict with processing results:
            {
                'success': bool,
                'message': str,
                'grouped_events_csv': Path or None,
                'event_count': int
            }
        """
        parsed_files_dir = Path(parsed_files_dir)
        output_dir = Path(output_dir)
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: DATA PREPROCESSING - Grouped Events Generation")
        self.logger.info("=" * 80)
        self.logger.info("Input directory:  {}".format(parsed_files_dir))
        self.logger.info("Output directory: {}".format(output_dir))
        
        try:
            # Load parsed data
            self.logger.info("\n[Step 1/6] Loading parsed CSV files...")
            data = load_phase1_data(parsed_files_dir)
            if data is None:
                return {
                    'success': False,
                    'message': 'Failed to load parsed data files',
                    'grouped_events_csv': None,
                    'event_count': 0
                }
            
            df_mft = data['mft']
            df_logfile = data['logfile']
            df_usnjrnl = data['usnjrnl']
            
            # Preprocess MFT
            self.logger.info("\n[Step 2/6] Preprocessing MFT...")
            df_mft_ref = preprocess_mft(df_mft)
            
            # Process LogFile
            self.logger.info("\n[Step 3/6] Processing LogFile events...")
            df_logfile_joined = process_logfile_events(df_logfile, df_mft_ref)
            
            # Process UsnJrnl
            self.logger.info("\n[Step 4/6] Processing UsnJrnl events...")
            df_usnjrnl_joined = process_usnjrnl_events(df_usnjrnl, df_mft_ref)
            
            # Normalize schemas
            self.logger.info("\n[Step 5/6] Normalizing event schemas...")
            df_logfile_norm = normalize_logfile_schema(df_logfile_joined)
            df_usnjrnl_norm = normalize_usnjrnl_schema(df_usnjrnl_joined)
            
            # Combine and sort
            self.logger.info("\n[Step 6/6] Combining and sorting events...")
            df_grouped = combine_and_sort_events(df_logfile_norm, df_usnjrnl_norm)
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export grouped events
            output_file = output_dir / 'grouped_events.csv'
            df_grouped.to_csv(output_file, index=False, encoding='utf-8')
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STAGE 2 COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info("✓ Grouped events exported to: {}".format(output_file))
            self.logger.info("  Total events: {:,}".format(len(df_grouped)))
            self.logger.info("  Unique files: {:,}".format(df_grouped['FileFRN'].nunique()))
            self.logger.info("=" * 80)
            
            return {
                'success': True,
                'message': 'Successfully preprocessed {} events'.format(len(df_grouped)),
                'grouped_events_csv': str(output_file),
                'event_count': len(df_grouped)
            }
        
        except Exception as e:
            self.logger.error("Error during preprocessing: {}".format(str(e)), exc_info=True)
            return {
                'success': False,
                'message': 'Preprocessing failed: {}'.format(str(e)),
                'grouped_events_csv': None,
                'event_count': 0
            }


# =============================================================================
# UNIFIED EVENT SCHEMA DEFINITION
# ============================================================================="
# Final column order for grouped_events.csv output
# This schema merges LogFile and UsnJrnl events into a single structure

FINAL_COLUMNS = [
    # Identifiers
    'dataID', 'FileFRN', 'FileName', 'FilePath', 'EventSource', 'EventTimestamp',
    # Sequence numbers
    'LSN', 'USN',    
    # LogFile fields
    'RedoOP', 'UndoOP', 'RedoOPName', 'UndoOPName',
    'RecordOffset', 'AttributeOffset', 'TargetVCN', 'IsTimestampChange',
    # LogFile timestamps
    'Undo_$SI-C', 'Undo_$SI-M', 'Undo_$SI-E', 'Undo_$SI-A',
    'Redo_$SI-C', 'Redo_$SI-M', 'Redo_$SI-E', 'Redo_$SI-A',
    # UsnJrnl fields
    'ReasonCode', 'ReasonFlags',
    'HasBasicInfoChange', 'HasClose', 'HasFileCreate'
]

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_phase1_data(parsed_files_dir):
    """
    Load MFT, LogFile, and UsnJrnl CSVs from Parser stage output.
    
    Parameters:
        parsed_files_dir: Path to directory containing parsed CSV files
                         (MFT_parsed.csv, LogFile_parsed.csv, UsnJrnl_parsed.csv)
    
    Returns:
        dict with keys: mft, logfile, usnjrnl (DataFrames) or None on error
    """
    parsed_files_dir = Path(parsed_files_dir)
    
    logger.info("Loading parsed NTFS files from: {}".format(parsed_files_dir))
    
    mft_csv = parsed_files_dir / "MFT_parsed.csv"
    logfile_csv = parsed_files_dir / "LogFile_parsed.csv"
    usnjrnl_csv = parsed_files_dir / "UsnJrnl_parsed.csv"
    
    # Check if files exist
    missing_files = []
    if not mft_csv.exists():
        missing_files.append("MFT_parsed.csv")
    if not logfile_csv.exists():
        missing_files.append("LogFile_parsed.csv")
    if not usnjrnl_csv.exists():
        missing_files.append("UsnJrnl_parsed.csv")
    
    if missing_files:
        logger.error("Missing parsed files: {}".format(", ".join(missing_files)))
        return None
    
    try:
        logger.info("  Loading MFT...")
        df_mft = pd.read_csv(mft_csv, low_memory=False)
        logger.info("    MFT records: {:,}".format(len(df_mft)))
        
        logger.info("  Loading LogFile...")
        df_logfile = pd.read_csv(logfile_csv, low_memory=False)
        logger.info("    LogFile records: {:,}".format(len(df_logfile)))
        
        logger.info("  Loading UsnJrnl...")
        df_usnjrnl = pd.read_csv(usnjrnl_csv, low_memory=False)
        logger.info("    UsnJrnl records: {:,}".format(len(df_usnjrnl)))
        
        return {
            'mft': df_mft,
            'logfile': df_logfile,
            'usnjrnl': df_usnjrnl
        }
    except Exception as e:
        logger.error("Error loading parsed files: {}".format(str(e)))
        return None


# =============================================================================
# MFT PREPROCESSING FUNCTION
# =============================================================================

def preprocess_mft(df_mft):
    """
    Create MFT reference table for joining with LogFile and UsnJrnl.
    
    Parameters:
        df_mft: MFT DataFrame from Parser stage
    
    Returns:
        df_mft_ref: Preprocessed MFT reference table with FileFRN as key
    """
    logger.info("Preprocessing MFT...")
    
    mft_columns = [
        'EntryNumber', 'FileName', 'FilePath', 'IsActive', 'LSN', 'ParentFRN',
        '$SI-C', '$SI-M', '$SI-E', '$SI-A',
        '$FN-C', '$FN-M', '$FN-E', '$FN-A'
    ]
    
    # Select columns that exist
    available_cols = [c for c in mft_columns if c in df_mft.columns]
    df_mft_ref = df_mft[available_cols].copy()
    
    # Rename EntryNumber to FileFRN
    df_mft_ref = df_mft_ref.rename(columns={'EntryNumber': 'FileFRN'})
    
    # Handle missing values
    if 'FileName' in df_mft_ref.columns:
        df_mft_ref['FileName'] = df_mft_ref['FileName'].fillna('')
    if 'FilePath' in df_mft_ref.columns:
        df_mft_ref['FilePath'] = df_mft_ref['FilePath'].fillna('')
    
    logger.info("  MFT reference table created: {:,} records".format(len(df_mft_ref)))
    
    return df_mft_ref

# =============================================================================
# LOGFILE PROCESSING FUNCTION
# =============================================================================

def process_logfile_events(df_logfile, df_mft_ref):
    """
    Process LogFile events and join with MFT reference.
    
    Parameters:
        df_logfile: LogFile DataFrame from Parser stage
        df_mft_ref: MFT reference table
    
    Returns:
        df_logfile_joined: LogFile events with MFT metadata
    """
    logger.info("Processing LogFile events...")
    
    # Clean TargetFRN
    df_logfile = df_logfile.copy()
    df_logfile['TargetFRN_clean'] = pd.to_numeric(df_logfile['TargetFRN'], errors='coerce')
    df_logfile['TargetFRN_clean'] = df_logfile['TargetFRN_clean'].fillna(-1).astype(int)
    
    # Filter valid records
    df_logfile_valid = df_logfile[df_logfile['TargetFRN_clean'] >= 0].copy()
    
    logger.info("  Valid LogFile records: {:,}".format(len(df_logfile_valid)))
    
    # Join with MFT
    df_logfile_joined = df_logfile_valid.merge(
        df_mft_ref[['FileFRN', 'FileName', 'FilePath', 'IsActive']],
        left_on='TargetFRN_clean',
        right_on='FileFRN',
        how='left'
    )
    
    # Add event source
    df_logfile_joined['EventSource'] = 'LogFile'
    
    # Set EventTimestamp from Redo_$SI-E
    df_logfile_joined['EventTimestamp'] = df_logfile_joined.get('Redo_$SI-E', np.nan)
    
    return df_logfile_joined

# =============================================================================
# USNJRNL PROCESSING FUNCTION
# =============================================================================

def process_usnjrnl_events(df_usnjrnl, df_mft_ref):
    """
    Process UsnJrnl events and join with MFT reference.
    
    Parameters:
        df_usnjrnl: UsnJrnl DataFrame from Parser stage
        df_mft_ref: MFT reference table
    
    Returns:
        df_usnjrnl_joined: UsnJrnl events with MFT metadata
    """
    logger.info("Processing UsnJrnl events...")
    
    # Clean FRN
    df_usnjrnl = df_usnjrnl.copy()
    df_usnjrnl['FRN_clean'] = pd.to_numeric(df_usnjrnl['FRN'], errors='coerce')
    df_usnjrnl['FRN_clean'] = df_usnjrnl['FRN_clean'].fillna(-1).astype(int)
    
    # Filter valid records
    df_usnjrnl_valid = df_usnjrnl[df_usnjrnl['FRN_clean'] >= 0].copy()
    
    logger.info("  Valid UsnJrnl records: {:,}".format(len(df_usnjrnl_valid)))
    
    # Join with MFT
    df_usnjrnl_joined = df_usnjrnl_valid.merge(
        df_mft_ref[['FileFRN', 'FileName', 'FilePath', 'IsActive']],
        left_on='FRN_clean',
        right_on='FileFRN',
        how='left',
        suffixes=('_usn', '_mft')
    )
    
    # Use UsnJrnl filename, fallback to MFT
    if 'FileName_usn' in df_usnjrnl_joined.columns:
        df_usnjrnl_joined['FileName'] = df_usnjrnl_joined['FileName_usn'].fillna(
            df_usnjrnl_joined['FileName_mft'].fillna('')
        )
        df_usnjrnl_joined = df_usnjrnl_joined.drop(
            columns=[c for c in ['FileName_usn', 'FileName_mft'] if c in df_usnjrnl_joined.columns]
        )
    
    # Add event source
    df_usnjrnl_joined['EventSource'] = 'UsnJrnl'
    
    # Set EventTimestamp from Timestamp
    df_usnjrnl_joined['EventTimestamp'] = df_usnjrnl_joined.get('Timestamp', np.nan)
    
    return df_usnjrnl_joined

# =============================================================================
# SCHEMA NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_logfile_schema(df_logfile_joined, data_id=1):
    """
    Normalize LogFile events to unified schema.
    
    Adds placeholder columns for UsnJrnl-specific fields that don't apply
    to LogFile records.
    
    Parameters:
        df_logfile_joined: Processed LogFile DataFrame
        data_id: Dataset identifier
    
    Returns:
        DataFrame with unified schema columns in correct order
    """
    logger.info("Normalizing LogFile schema...")
    
    df = df_logfile_joined.copy()
    
    # Add dataID
    df['dataID'] = data_id
    
    # Add placeholder columns for UsnJrnl-specific fields
    df['USN'] = np.nan
    df['ReasonCode'] = np.nan
    df['ReasonFlags'] = ''
    df['HasBasicInfoChange'] = False
    df['HasClose'] = False
    df['HasFileCreate'] = False
    
    # Ensure all unified schema columns exist
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col not in ['ReasonFlags', 'RedoOPName', 'UndoOPName'] else ''
    
    return df[FINAL_COLUMNS]


def normalize_usnjrnl_schema(df_usnjrnl_joined, data_id=1):
    """
    Normalize UsnJrnl events to unified schema.
    
    Adds placeholder columns for LogFile-specific fields that don't apply
    to UsnJrnl records.
    
    Parameters:
        df_usnjrnl_joined: Processed UsnJrnl DataFrame
        data_id: Dataset identifier
    
    Returns:
        DataFrame with unified schema columns in correct order
    """
    logger.info("Normalizing UsnJrnl schema...")
    
    df = df_usnjrnl_joined.copy()
    
    # Add dataID
    df['dataID'] = data_id
    
    # Add placeholder columns for LogFile-specific fields
    df['LSN'] = np.nan
    df['RedoOP'] = np.nan
    df['UndoOP'] = np.nan
    df['RedoOPName'] = ''
    df['UndoOPName'] = ''
    df['RecordOffset'] = np.nan
    df['AttributeOffset'] = np.nan
    df['TargetVCN'] = np.nan
    df['IsTimestampChange'] = False
    
    # Undo/Redo timestamps not applicable
    for ts_col in ['Undo_$SI-C', 'Undo_$SI-M', 'Undo_$SI-E', 'Undo_$SI-A',
                   'Redo_$SI-C', 'Redo_$SI-M', 'Redo_$SI-E', 'Redo_$SI-A']:
        df[ts_col] = ''
    
    # Ensure all columns exist
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col not in ['ReasonFlags', 'RedoOPName', 'UndoOPName'] else ''
    
    return df[FINAL_COLUMNS]


# =============================================================================
# EVENT COMBINER AND SORTER
# =============================================================================

def combine_and_sort_events(df_logfile_norm, df_usnjrnl_norm):
    """
    Combine LogFile and UsnJrnl events and sort by file and timestamp.
    
    Parameters:
        df_logfile_norm: Normalized LogFile events
        df_usnjrnl_norm: Normalized UsnJrnl events
    
    Returns:
        df_grouped: Combined and sorted DataFrame
    """
    logger.info("Combining and sorting events...")
    
    # Combine event streams
    df_combined = pd.concat([df_logfile_norm, df_usnjrnl_norm], ignore_index=True)
    
    logger.info("  Combined events: {:,}".format(len(df_combined)))
    
    # Convert EventTimestamp
    df_combined['EventTimestamp'] = pd.to_datetime(df_combined['EventTimestamp'], errors='coerce')
    
    # Sort by FileFRN, then EventTimestamp, then LSN/USN
    # LSN and USN determine sequence within same timestamp
    
    # Create sort keys
    sentinel_date = pd.Timestamp('1970-01-01')
    df_combined['_sort_ts'] = df_combined['EventTimestamp'].fillna(sentinel_date)
    df_combined['_sort_seq'] = df_combined['LSN'].fillna(0) + df_combined['USN'].fillna(0)
    
    # Sort by FileFRN, timestamp, sequence
    df_grouped = df_combined.sort_values(
        by=['FileFRN', '_sort_ts', '_sort_seq'],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    
    # Remove temporary columns
    df_grouped = df_grouped.drop(columns=['_sort_ts', '_sort_seq'])
    
    logger.info("  Grouped events: {:,} records".format(len(df_grouped)))
    
    return df_grouped


# =============================================================================
# MODULE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Direct execution entry point for testing.
    
    Usage:
        python data_preprocessing.py <parsed_dir> <output_dir>
        
    Example:
        python data_preprocessing.py "Parsed Files" "Grouped Events File"
    """
    import sys
    
    # Configure logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) >= 3:
        parsed_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
        
        preprocessor = DataPreprocessor()
        results = preprocessor.preprocess_all(parsed_dir, output_dir)
        
        print("\nResults:")
        print("  Success: {}".format(results['success']))
        print("  Message: {}".format(results['message']))
        if results['grouped_events_csv']:
            print("  Output: {}".format(results['grouped_events_csv']))
            print("  Events: {}".format(results['event_count']))
        
        sys.exit(0 if results['success'] else 1)
    else:
        print("Data Preprocessing Module for NTFS Timestomping Detection")
        print("\nUsage: python data_preprocessing.py <parsed_dir> <output_dir>")
        sys.exit(0)