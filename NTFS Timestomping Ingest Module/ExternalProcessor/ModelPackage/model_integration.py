# -*- coding: utf-8 -*-

"""
Model Integration - ML Inference for NTFS Timestomping Detection

This module applies the trained LightGBM model to feature-engineered files
and produces detection outputs suitable for Autopsy plugin integration.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import sys

from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ModelIntegration:
    FEATURE_COLUMNS = [
        'num_timestamp_changes',
        'num_backward_jumps',
        'num_forward_jumps',
        'num_creation_changes',
        'max_backward_jump_seconds',
        'mean_jump_seconds',
        'timestamp_change_density',
        'num_zero_nanosecond_events',
        'only_SI_modified',
        'num_update_resident_value',
        'repeated_update_resident_value',
        'consecutive_timestamp_changes',
        'num_logfile_events',
        'has_logfile_ts_change',
        'has_usn_basic_info',
        'has_usn_close',
        'has_usn_file_create',
        'num_usn_basic_info',
        'num_usn_close',
        'num_usn_file_create',
        'logfile_usn_mismatch',
        'has_usn_basic_pattern',
        'num_usnjrnl_events',
        'min_inter_event_delta',
        'max_inter_event_delta',
        'mean_inter_event_delta',
        'burstiness_score',
        'event_time_span_seconds',
        'total_events'
    ]
    
    EXCLUDE_COLUMNS = ['dataID', 'FileName', 'is_timestomped', 'FilePath']

    # LightGBM optimal threshold from training
    THRESHOLD = 0.02
    
    def __init__(self, model_dir, features_dir, output_dir, logger_obj=None):
        self.logger = logger_obj if logger_obj else logger
        self.model_dir = Path(model_dir)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        
        # Paths to model and scaler
        self.model_path = self.model_dir / "model_lightgbm.joblib"
        self.scaler_path = self.model_dir / "feature_scaler.joblib"
        
        # Path to input CSV
        self.df_input = None
        self._find_df_input()
        
        self.model = None
        self.scaler = None
    
    def log(self, level, msg):
        if self.logger:
            self.logger.log(level, msg)
        else:
            sys.stderr.write(f"[{level}] {msg}\n")
    
    def _find_df_input(self):
        if self.features_dir.exists():
            csv_files = list(self.features_dir.glob("file_features.csv"))
            if csv_files:
                self.df_input = csv_files[0]
                self.log(logging.INFO, f"Found input CSV: {self.df_input}")
    
    def load_models(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
            
            self.model = joblib.load(str(self.model_path))
            self.scaler = joblib.load(str(self.scaler_path))
            
            self.log(logging.INFO, f"Model loaded: {type(self.model).__name__}")
            self.log(logging.INFO, f"Scaler loaded: {type(self.scaler).__name__}")
            self.log(logging.INFO, f"Model expects {self.model.n_features_in_} features")
            
            return True
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            self.log(logging.ERROR, error_msg)
            raise RuntimeError(error_msg)
    
    def load_features(self):
        try:
            if not self.df_input or not self.df_input.exists():
                raise FileNotFoundError(f"Input CSV not found: {self.df_input}")
            
            df_input = pd.read_csv(str(self.df_input))
            self.log(logging.INFO, f"Input data loaded: {len(df_input):,} files")
            self.log(logging.INFO, f"Columns: {len(df_input.columns)}")
            self.log(logging.INFO, f"First few columns: {list(df_input.columns[:10])}")
            
            return df_input
        except Exception as e:
            error_msg = f"Failed to load features: {str(e)}"
            self.log(logging.ERROR, error_msg)
            raise RuntimeError(error_msg)
    
    def prepare_features(self, df_input):
        try:
            missing_cols = [col for col in self.FEATURE_COLUMNS if col not in df_input.columns]
            if missing_cols:
                self.log(logging.WARNING, f"Missing columns: {missing_cols}")
            
            X = df_input[self.FEATURE_COLUMNS].copy()
            
            X = X.fillna(0)
            
            bool_cols = X.select_dtypes(include=['bool']).columns
            X[bool_cols] = X[bool_cols].astype(int)
            
            self.log(logging.INFO, f"Feature matrix shape: {X.shape}")
            self.log(logging.INFO, f"Data types: {X.dtypes.value_counts().to_dict()}")
            
            return X
        except Exception as e:
            error_msg = f"Failed to prepare features: {str(e)}"
            self.log(logging.ERROR, error_msg)
            raise RuntimeError(error_msg)
    
    def predict(self, X, df_input=None):
        try:
            y_prob = self.model.predict_proba(X)[:, 1]
            
            # Apply threshold to get binary predictions
            y_pred = (y_prob >= self.THRESHOLD).astype(int)
            
            if df_input is not None:
                df_results = df_input.copy()
            elif isinstance(self.df_input, pd.DataFrame):
                df_results = self.df_input.copy()
            else:
                raise ValueError("df_input must be provided or self.df_input must be a DataFrame")
            
            df_results['Confidence'] = y_prob
            df_results['Flagged'] = y_pred.astype(bool)
            
            self.log(logging.INFO, "Predictions complete")
            self.log(logging.INFO, "Total files: {:,}".format(len(y_prob)))
            self.log(logging.INFO, "Files flagged: {:,} ({:.2f}%)".format(y_pred.sum(), y_pred.mean()*100))
            self.log(logging.INFO, "Files cleared: {:,}".format((~y_pred.astype(bool)).sum()))
            
            return y_prob, y_pred, df_results
        except Exception as e:
            error_msg = "Failed to generate predictions: {}".format(str(e))
            self.log(logging.ERROR, error_msg)
            raise RuntimeError(error_msg)
    
    def analyze_detection_reasons(self, df_results):
        self.logger.info("Confidence Distribution:")
        self.logger.info("=" * 50)

        # High confidence (>0.5)
        high_conf = (df_results['Confidence'] > 0.5).sum()
        self.logger.info(f"High Confidence (>0.5):      {high_conf:,}")

        # Medium confidence (0.1 - 0.5)
        med_conf = ((df_results['Confidence'] > 0.1) & (df_results['Confidence'] <= 0.5)).sum()
        self.logger.info(f"Medium Confidence (0.1-0.5): {med_conf:,}")

        # Low confidence (threshold - 0.1)
        low_conf = ((df_results['Confidence'] >= self.THRESHOLD) & (df_results['Confidence'] <= 0.1)).sum()
        self.logger.info(f"Low Confidence ({self.THRESHOLD}-0.1):   {low_conf:,}")

        # Below threshold
        below_thresh = (df_results['Confidence'] < self.THRESHOLD).sum()
        self.logger.info(f"Below Threshold (<{self.THRESHOLD}):   {below_thresh:,}")
    
    @staticmethod
    def convert_seconds_to_readable(seconds):
        if seconds >= 86400:
            return f"{seconds/86400:.1f} days"
        elif seconds >= 3600:
            return f"{seconds/3600:.1f} hours"
        elif seconds >= 60:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds:.0f} seconds"
    
    def generate_detection_reasons_enhanced(self, row):
        indicators = []
        
        # Backward timestamp jumps
        if row['num_backward_jumps'] > 0:
            max_jump_sec = row['max_backward_jump_seconds']
            duration = self.convert_seconds_to_readable(max_jump_sec)
            severity = 'HIGH' if max_jump_sec > 86400 else 'MEDIUM'
            
            indicators.append({
                'type': 'BACKWARD_TIMESTAMP',
                'severity': severity,
                'finding': f"{int(row['num_backward_jumps'])} backward timestamp jump(s) detected (max: {duration})",
                'forensic_meaning': "Timestamps moved backwards in time, which cannot occur through normal file "
                                "operations. This indicates deliberate manipulation to make a file appear older "
                                "than it actually is. Attackers use this technique to blend malicious files with "
                                "legitimate system files by matching their timestamps.",
                'artifact_source': '$MFT $STANDARD_INFORMATION timestamps'
            })
        
        # Zero nanosecond timestamps
        if row['num_zero_nanosecond_events'] > 0:
            indicators.append({
                'type': 'ZERO_NANOSECONDS',
                'severity': 'MEDIUM',
                'finding': f"{int(row['num_zero_nanosecond_events'])} timestamp event(s) with zero nanosecond precision",
                'forensic_meaning': "NTFS stores timestamps with 100-nanosecond precision. Legitimate file operations "
                                "produce non-zero nanosecond values due to system clock granularity. Zero nanoseconds "
                                "typically indicates timestamps were set programmatically using APIs like SetFileTime() "
                                "with rounded values, or timestomping tools (Timestomp, SetMACE, NewFileTime) that "
                                "fail to populate sub-second precision.",
                'artifact_source': '$MFT timestamp fields (nanosecond component)'
            })
        
        # SI-only modification
        if row['only_SI_modified'] == 1 or row['only_SI_modified'] == True:
            indicators.append({
                'type': 'SI_ONLY_MODIFICATION',
                'severity': 'HIGH',
                'finding': "Only $STANDARD_INFORMATION timestamps modified, $FILE_NAME unchanged",
                'forensic_meaning': "Normal file operations update both $STANDARD_INFORMATION (SI) and $FILE_NAME (FN) "
                                "attributes. Most timestomping tools only modify SI because FN requires kernel-level "
                                "access. When SI shows different timestamps than FN, the FN attribute reveals the "
                                "true file creation time. This is a strong indicator of timestamp manipulation.",
                'artifact_source': '$MFT $STANDARD_INFORMATION vs $FILE_NAME comparison'
            })
        
        # LogFile/USN mismatch
        if row['logfile_usn_mismatch'] == 1 or row['logfile_usn_mismatch'] == True:
            indicators.append({
                'type': 'ARTIFACT_MISMATCH',
                'severity': 'HIGH',
                'finding': "$LogFile and $UsnJrnl show inconsistent timestamp records",
                'forensic_meaning': "The $LogFile (transaction journal) and $UsnJrnl (change journal) independently "
                                "record file system operations with their own timestamps. When these artifacts show "
                                "different timestamps than the $MFT for the same file operation, it indicates the "
                                "$MFT timestamps were modified AFTER the original operation was journaled. Attackers "
                                "rarely modify journal entries as it requires advanced techniques and risks corruption.",
                'artifact_source': '$LogFile and $UsnJrnl cross-correlation with $MFT'
            })
        
        # High burstiness
        if row['burstiness_score'] > 0.5:
            severity = 'HIGH' if row['burstiness_score'] > 0.8 else 'MEDIUM'
            indicators.append({
                'type': 'HIGH_BURSTINESS',
                'severity': severity,
                'finding': f"Temporal clustering score: {row['burstiness_score']:.3f}",
                'forensic_meaning': "Multiple timestamp modifications occurred in rapid succession (burst pattern). "
                                "Normal file operations produce timestamp changes distributed over time as users "
                                "interact with files naturally. High burstiness suggests automated or scripted "
                                "timestamp manipulation where multiple timestamps were modified in a short time "
                                "window, typical of batch timestomping operations.",
                'artifact_source': '$MFT timestamp change event timing analysis'
            })
        
        # Consecutive timestamp changes
        if row['consecutive_timestamp_changes'] > 2:
            indicators.append({
                'type': 'CONSECUTIVE_CHANGES',
                'severity': 'MEDIUM',
                'finding': f"{int(row['consecutive_timestamp_changes'])} consecutive timestamp modifications without content changes",
                'forensic_meaning': "Multiple sequential modifications to timestamp fields were detected without "
                                "intervening file content changes. Normal file access patterns show content "
                                "modifications (reads, writes) between timestamp updates. Consecutive timestamp-only "
                                "changes indicate deliberate timestamp manipulation rather than normal file usage.",
                'artifact_source': '$MFT and $LogFile event sequence analysis'
            })
        
        # High number of timestamp changes
        if row['num_timestamp_changes'] > 10:
            indicators.append({
                'type': 'EXCESSIVE_CHANGES',
                'severity': 'LOW',
                'finding': f"{int(row['num_timestamp_changes'])} total timestamp change events recorded",
                'forensic_meaning': "An unusually high number of timestamp modifications were recorded for this file. "
                                "While not definitive on its own, excessive timestamp changes combined with other "
                                "indicators may suggest repeated manipulation attempts or automated tools cycling "
                                "through timestamp values.",
                'artifact_source': '$MFT and $LogFile event count'
            })
        
        return indicators
    
    @staticmethod
    def format_short_reason(indicators):
        if not indicators:
            return ""
        parts = []
        for ind in indicators:
            parts.append(f"{ind['type']}: {ind['finding']}")
        return "; ".join(parts)
    
    def get_overall_severity(self, indicators, confidence):
        if not indicators:
            return "LOW"
        
        indicator_severities = [ind['severity'] for ind in indicators]
        if 'HIGH' in indicator_severities:
            max_indicator_severity = 'HIGH'
        elif 'MEDIUM' in indicator_severities:
            max_indicator_severity = 'MEDIUM'
        else:
            max_indicator_severity = 'LOW'
        
        if confidence > 0.5:
            return max_indicator_severity
        elif confidence > 0.1:
            if max_indicator_severity == 'HIGH':
                return 'MEDIUM'
            elif max_indicator_severity == 'MEDIUM':
                return 'LOW'
            else:
                return 'LOW'
        else:
            return 'LOW'
    
    def get_recommended_action(self, indicators, severity):
        if not indicators:
            return "No action required"
        
        types = [ind['type'] for ind in indicators]
        
        if severity == 'HIGH':
            if 'SI_ONLY_MODIFICATION' in types:
                return "VERIFY: Compare $STANDARD_INFORMATION and $FILE_NAME timestamps using MFT parser"
            elif 'ARTIFACT_MISMATCH' in types:
                return "VERIFY: Cross-reference $LogFile and $UsnJrnl entries for this file"
            elif 'BACKWARD_TIMESTAMP' in types:
                return "INVESTIGATE: Examine file metadata and correlate with timeline analysis"
            else:
                return "INVESTIGATE: High confidence detection requires manual verification"
        elif severity == 'MEDIUM':
            return "REVIEW: Manual inspection recommended to confirm timestomping"
        else:
            return "MONITOR: Low confidence detection, consider in context of other findings"
    
    def apply_detection_analysis(self, df_results):
        df_results['Indicators'] = df_results.apply(self.generate_detection_reasons_enhanced, axis=1)

        df_results['Detection_Reasons'] = df_results['Indicators'].apply(self.format_short_reason)

        df_results['Indicator_Count'] = df_results['Indicators'].apply(len)

        df_results['Severity'] = df_results.apply(
            lambda row: self.get_overall_severity(row['Indicators'], row['Confidence']), 
            axis=1
        )

        df_results['Recommended_Action'] = df_results.apply(
            lambda row: self.get_recommended_action(row['Indicators'], row['Severity']),
            axis=1
        )

        self.logger.info("Enhanced detection reasons generated")
        self.logger.info(f"Files with indicators: {(df_results['Indicator_Count'] > 0).sum():,}")
        self.logger.info(f"Severity Distribution (Flagged Files):")
        self.logger.info(str(df_results[df_results['Flagged']]['Severity'].value_counts()))
        
        return df_results
    
    def calculate_forensic_metrics(self, df_results):
        total_files = len(df_results)
        flagged_files = df_results['Flagged'].sum()
        cleared_files = total_files - flagged_files

        # Candidate Reduction Rate (CRR)
        crr = cleared_files / total_files

        # Estimated NNI (Number Needed to Investigate)
        estimated_true_positives = 12  # LoneWolf has 12 known timestomped files
        nni = flagged_files / estimated_true_positives if estimated_true_positives > 0 else 0

        flag_rate = flagged_files / total_files

        self.logger.info("Forensic Metrics:")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Files Analyzed:        {total_files:,}")
        self.logger.info(f"Files Flagged:               {flagged_files:,} ({flag_rate*100:.2f}%)")
        self.logger.info(f"Files Cleared:               {cleared_files:,}")
        self.logger.info(f"Candidate Reduction Rate:    {crr*100:.2f}%")
        self.logger.info(f"Estimated NNI:               {nni:.1f} files/detection")
        
        return {
            'total_files': total_files,
            'flagged_files': flagged_files,
            'flag_rate': flag_rate,
            'crr': crr,
            'nni': nni
        }
    
    def generate_detected_files_output(self, df_results):
        try:
            df_detected = df_results[df_results['Flagged']].copy()
            
            def create_forensic_summary(row):
                indicators = row['Indicators']
                if not indicators:
                    return "Pattern match suggests possible timestomping"
                
                summaries = []
                for ind in indicators:
                    if ind['type'] == 'BACKWARD_TIMESTAMP':
                        summaries.append(f"timestamps manipulated backwards ({ind['finding'].split('max: ')[1].rstrip(')')})")
                    elif ind['type'] == 'ZERO_NANOSECONDS':
                        summaries.append("zero nanosecond precision detected")
                    elif ind['type'] == 'SI_ONLY_MODIFICATION':
                        summaries.append("$SI modified but $FN unchanged")
                    elif ind['type'] == 'ARTIFACT_MISMATCH':
                        summaries.append("journal artifacts inconsistent")
                    elif ind['type'] == 'HIGH_BURSTINESS':
                        summaries.append("burst pattern in modifications")
                
                return "File shows: " + "; ".join(summaries) if summaries else "Multiple indicators suggest timestomping"

            df_detected['Forensic_Summary'] = df_detected.apply(create_forensic_summary, axis=1)

            df_detected['Backward_Jump_Duration'] = df_detected['max_backward_jump_seconds'].apply(
                lambda x: self.convert_seconds_to_readable(x) if x > 0 else "N/A"
            )
            
            if 'only_SI_modified' in df_detected.columns:
                df_detected['only_SI_modified'] = df_detected['only_SI_modified'].astype(bool).map({True: 'true', False: 'false'})

            if 'logfile_usn_mismatch' in df_detected.columns:
                df_detected['logfile_usn_mismatch'] = df_detected['logfile_usn_mismatch'].astype(bool).map({True: 'true', False: 'false'})
                
            if 'consecutive_timestamp_changes' in df_detected.columns:
                df_detected['consecutive_timestamp_changes'] = df_detected['consecutive_timestamp_changes'].astype(bool).map({True: 'true', False: 'false'})
            
            detected_columns = [
                'FileName',
                'Confidence',
                'Severity',
                'Indicator_Count',
                'Forensic_Summary',
                'Detection_Reasons',
                'Recommended_Action',
                'num_backward_jumps',
                'Backward_Jump_Duration',
                'num_zero_nanosecond_events',
                'only_SI_modified',
                'logfile_usn_mismatch',
                'burstiness_score',
                'consecutive_timestamp_changes'
            ]

            if 'FilePath' in df_detected.columns:
                detected_columns = ['FilePath'] + detected_columns
            
            df_detected_output = df_detected[detected_columns].sort_values(
                ['Severity', 'Confidence'], 
                ascending=[True, False],  # HIGH severity first, then by confidence
                key=lambda x: x.map({'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}) if x.name == 'Severity' else x
            )

            severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            df_detected_output['_sort'] = df_detected_output['Severity'].map(severity_order)
            df_detected_output = df_detected_output.sort_values(['_sort', 'Confidence'], ascending=[True, False])
            df_detected_output = df_detected_output.drop('_sort', axis=1)
            
            detected_path = self.output_dir / f"detected_files.csv"
            df_detected_output.to_csv(str(detected_path), index=False)
            
            self.log(logging.INFO, f"Output 1: detected_files.csv")
            self.log(logging.INFO, f"  Location: {detected_path}")
            self.log(logging.INFO, f"  Records: {len(df_detected_output):,}")
            self.log(logging.INFO, f"\nSeverity Breakdown:")
            for severity, count in df_detected_output['Severity'].value_counts().items():
                self.log(logging.INFO, f"    {severity}: {count}")
            self.log(logging.INFO, f"\nTop 5 Detected Files:")
            self.log(logging.INFO, str(df_detected_output.head()[['FileName', 'Confidence', 'Severity', 'Forensic_Summary']]))
            
            return detected_path
        except Exception as e:
            error_msg = f"Failed to generate detected_files: {str(e)}"
            self.log(logging.ERROR, error_msg)
            raise RuntimeError(error_msg)
    
    def generate_full_features_output(self, df_results):
        try:
            df_output = df_results.copy()

            boolean_columns = [
                'only_SI_modified',
                'repeated_update_resident_value',
                'consecutive_timestamp_changes',
                'has_logfile_ts_change',
                'has_usn_basic_info',
                'has_usn_close',
                'has_usn_file_create',
                'logfile_usn_mismatch',
                'has_usn_basic_pattern'
            ]

            for col in boolean_columns:
                if col in df_output.columns:
                    df_output[col] = df_output[col].astype(bool).map({True: 'true', False: 'false'})
                    
            full_output_columns = ['FileName', 'Confidence', 'Flagged'] + self.FEATURE_COLUMNS
            
            if 'FilePath' in df_output.columns:
                full_output_columns = ['FilePath'] + full_output_columns
            
            if 'is_timestomped' in df_output.columns:
                full_output_columns.append('is_timestomped')
            
            df_full_output = df_output[full_output_columns].sort_values('Confidence', ascending=False)
            
            full_path = self.output_dir / f"files_with_features.csv"
            df_full_output.to_csv(str(full_path), index=False)
            
            self.log(logging.INFO, f"Output 2: files_with_features.csv")
            self.log(logging.INFO, f"  Location: {full_path}")
            self.log(logging.INFO, f"  Records: {len(df_full_output):,}")
            self.log(logging.INFO, f"  Columns: {len(full_output_columns)}")
            
            return full_path
        except Exception as e:
            error_msg = f"Failed to generate files_with_features: {str(e)}"
            self.log(logging.ERROR, error_msg)
            raise RuntimeError(error_msg)
    
    def generate_summary_report(self, df_results, metrics=None):
        try:
            if metrics is None:
                metrics = self.calculate_forensic_metrics(df_results)
            
            total_files = metrics['total_files']
            flagged_files = metrics['flagged_files']
            flag_rate = metrics['flag_rate']
            crr = metrics['crr']
            nni = metrics['nni']
            
            severity_order_map = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            all_flagged = df_results[df_results['Flagged']].copy()
            all_flagged['_sort'] = all_flagged['Severity'].map(severity_order_map)
            all_flagged_sorted = all_flagged.sort_values(['_sort', 'Confidence'], ascending=[True, False])[
                ['FileName', 'Confidence', 'Severity', 'Indicator_Count']
            ]

            # Count detection patterns
            backward_jumps = (df_results['num_backward_jumps'] > 0).sum()
            zero_nanoseconds = (df_results['num_zero_nanosecond_events'] > 0).sum()
            si_only = (df_results['only_SI_modified'] == 1).sum() if 'only_SI_modified' in df_results.columns else 0
            logfile_mismatch = (df_results['logfile_usn_mismatch'] == 1).sum() if 'logfile_usn_mismatch' in df_results.columns else 0
            high_burstiness = (df_results['burstiness_score'] > 0.5).sum()

            # Confidence and severity counts
            high_conf = (df_results['Confidence'] > 0.5).sum()
            med_conf = ((df_results['Confidence'] > 0.1) & (df_results['Confidence'] <= 0.5)).sum()
            low_conf = ((df_results['Confidence'] >= self.THRESHOLD) & (df_results['Confidence'] <= 0.1)).sum()

            high_sev = (df_results[df_results['Flagged']]['Severity'] == 'HIGH').sum()
            med_sev = (df_results[df_results['Flagged']]['Severity'] == 'MEDIUM').sum()
            low_sev = (df_results[df_results['Flagged']]['Severity'] == 'LOW').sum()
            
            summary_report = f"""================================================================================
NTFS TIMESTOMPING DETECTION REPORT
================================================================================

Analysis Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Used:         LightGBM (Phase 4 Baseline)
Threshold:          {self.THRESHOLD}

--------------------------------------------------------------------------------
EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Total Files Analyzed:       {total_files:,}
Files Flagged:              {flagged_files:,} ({flag_rate*100:.2f}%)

Confidence Breakdown:
  High Confidence (>0.5):     {high_conf}
  Medium Confidence (0.1-0.5): {med_conf}
  Low Confidence ({self.THRESHOLD}-0.1):  {low_conf}

Severity Breakdown:
  HIGH Severity:   {high_sev} (requires immediate investigation)
  MEDIUM Severity: {med_sev} (manual review recommended)
  LOW Severity:    {low_sev} (consider in context)

Candidate Reduction Rate:   {crr*100:.2f}%
  (Analyst reviews {flagged_files:,} files instead of {total_files:,})

Number Needed to Investigate: ~{nni:.1f} files per expected true positive

--------------------------------------------------------------------------------
ALL FLAGGED FILES ({flagged_files} DETECTED) - SORTED BY SEVERITY THEN CONFIDENCE
--------------------------------------------------------------------------------
Rank  Confidence  Severity  Indicators  FileName
----  ----------  --------  ----------  --------
"""

            for i, (_, row) in enumerate(all_flagged_sorted.iterrows(), 1):
                summary_report += f"{i:<5} {row['Confidence']:.6f}  {row['Severity']:<8}  {int(row['Indicator_Count']):<10}  {row['FileName']}\n"

            summary_report += f"""
--------------------------------------------------------------------------------
DETECTION INDICATOR SUMMARY
--------------------------------------------------------------------------------
The following indicators were detected across all analyzed files:

Indicator Type                    Count    Description
------------------------------    -----    -----------
Backward Timestamp Jumps          {backward_jumps:<5}    Files where timestamps moved backwards
Zero Nanosecond Precision         {zero_nanoseconds:<5}    Files with suspiciously round timestamps
SI-Only Modifications             {si_only:<5}    $STANDARD_INFORMATION modified, $FILE_NAME unchanged
LogFile/USN Mismatch              {logfile_mismatch:<5}    Journal artifacts inconsistent with $MFT
High Burstiness (>0.5)            {high_burstiness:<5}    Rapid successive timestamp modifications

--------------------------------------------------------------------------------
INDICATOR EXPLANATIONS
--------------------------------------------------------------------------------

BACKWARD_TIMESTAMP (HIGH Severity when >1 day)
  Timestamps that move backwards in time cannot occur through normal file
  operations. This definitively indicates deliberate manipulation to make
  files appear older than they actually are.

ZERO_NANOSECONDS (MEDIUM Severity)
  NTFS stores 100-nanosecond precision. Legitimate operations produce non-zero
  values. Zero nanoseconds indicates programmatic timestamp setting, typically
  via timestomping tools that don't populate sub-second precision.

SI_ONLY_MODIFICATION (HIGH Severity)
  Normal operations update both $STANDARD_INFORMATION and $FILE_NAME attributes.
  When only $SI is modified, it indicates tools that bypass the file system.
  The $FILE_NAME attribute retains the original timestamp.

ARTIFACT_MISMATCH (HIGH Severity)
  $LogFile and $UsnJrnl independently record operations. Inconsistency with
  $MFT timestamps indicates post-operation modification. Journal entries are
  difficult to modify without detection.

HIGH_BURSTINESS (MEDIUM-HIGH Severity)
  Multiple timestamp changes in rapid succession suggest automated or scripted
  manipulation. Normal file usage produces distributed timestamp changes.

--------------------------------------------------------------------------------
RECOMMENDED NEXT STEPS
--------------------------------------------------------------------------------
1. Review HIGH severity files immediately - these show strong manipulation signs
2. For BACKWARD_TIMESTAMP findings, compare $SI and $FN timestamps manually
3. Cross-reference flagged files with case timeline
4. Consider context: system restore, VM snapshots may cause false positives

--------------------------------------------------------------------------------
METHODOLOGY
--------------------------------------------------------------------------------
This analysis uses machine learning (LightGBM) trained on 22 datasets with
52 known timestomped files from real APT campaigns. The model achieves 100%
recall on validation datasets, ensuring no known timestomped files are missed.

Features analyzed: {len(self.FEATURE_COLUMNS)} (timestamp patterns, artifact consistency, temporal behavior)
Training data: APT17, APT19, APT21, APT28, APT29, APT30, APT37, APT38, APT40,
               DarkHotel, Kimsuky, Winnti, and controlled timestomping experiments

Based on: Oh et al. (2024) NTFS timestomping detection methodology

--------------------------------------------------------------------------------
OUTPUT FILES GENERATED
--------------------------------------------------------------------------------
1. detected_files.csv - Triage-ready list of all {flagged_files:,} files flagged as potentially timestomped. Includes confidence scores, severity ratings (HIGH/MEDIUM/LOW), forensic summaries, recommended actions, and key feature values per file. Sorted by severity then confidence. Use this as your primary investigative starting point.

2. files_with_features.csv - Complete feature dataset covering all {total_files:,} analyzed files, both flagged and cleared. Contains all 29 model input features alongside confidence scores and flag status. Use for deeper statistical analysis, threshold experimentation, or re-scoring files under alternative detection criteria.

3. summary.txt - This executive report. Provides high-level statistics, the complete ranked list of all {flagged_files:,} flagged files, indicator distribution, and methodology notes. Designed for inclusion in case reports or investigative briefings.

4. autopsy.log.0 - Autopsy's main runtime log file, which records runtime messages, errors, warnings, and status updates generated by Autopsy while working on a case.
   
================================================================================
Generated by Digital Detectives Timestomping Detector v1.0
================================================================================
"""
            
            summary_path = self.output_dir / f"summary.txt"
            with open(str(summary_path), 'w') as f:
                f.write(summary_report)

            self.log(logging.INFO, f"Output 3: summary.txt")
            self.log(logging.INFO, f"  Location: {summary_path}")
            self.log(logging.INFO, f"  Flagged Files Listed: {len(all_flagged_sorted)}")
            
            return summary_path
        except Exception as e:
            error_msg = f"Failed to generate summary: {str(e)}"
            self.log(logging.ERROR, error_msg)
            raise RuntimeError(error_msg)

def run_model_integration(model_dir, features_dir, output_dir, logger_obj=None):
    if logger_obj is None:
        logger_obj = logging.getLogger(__name__)
    
    integrator = ModelIntegration(model_dir, features_dir, output_dir, logger_obj)
    
    logger_obj.info("=" * 80)
    logger_obj.info("STAGE 4: MODEL INTEGRATION - ML INFERENCE")
    logger_obj.info("=" * 80)
    
    total_files = 0
    flagged_files = 0
    flag_rate = 0.0
    
    try:
        integrator.load_models()
        
        df_input = integrator.load_features()
        
        X = integrator.prepare_features(df_input)
        
        y_prob, y_pred, df_results = integrator.predict(X, df_input)
        
        if 'Confidence' not in df_results.columns:
            df_results['Confidence'] = y_prob
        if 'Flagged' not in df_results.columns:
            df_results['Flagged'] = y_pred.astype(bool)
        
        df_results = integrator.apply_detection_analysis(df_results)
        
        logger_obj.info("Generating detection indicators...")
        
        df_results['Indicators'] = df_results.apply(integrator.generate_detection_reasons_enhanced, axis=1)
        df_results['Detection_Reasons'] = df_results['Indicators'].apply(ModelIntegration.format_short_reason)
        df_results['Indicator_Count'] = df_results['Indicators'].apply(len)
        df_results['Severity'] = df_results.apply(
            lambda row: integrator.get_overall_severity(row['Indicators'], row['Confidence']), axis=1
        )
        df_results['Recommended_Action'] = df_results.apply(
            lambda row: integrator.get_recommended_action(row['Indicators'], row['Severity']), axis=1
        )
        
        logger_obj.info("Enhanced detection reasons generated")
        logger_obj.info("Files with indicators: {:,}".format((df_results['Indicator_Count'] > 0).sum()))
        
        total_files = len(df_results)
        flagged_files = df_results['Flagged'].sum()
        flag_rate = flagged_files / total_files
        
        metrics = integrator.calculate_forensic_metrics(df_results)
        
        logger_obj.info("\n" + "=" * 80)
        logger_obj.info("GENERATING OUTPUT FILES")
        logger_obj.info("=" * 80)
        
        integrator.output_dir.mkdir(parents=True, exist_ok=True)
     
        detected_path = integrator.generate_detected_files_output(df_results)
        features_path = integrator.generate_full_features_output(df_results)
        summary_path = integrator.generate_summary_report(df_results, metrics)
        
        logger_obj.info("\n" + "=" * 80)
        logger_obj.info("STAGE 4 COMPLETE")
        logger_obj.info("=" * 80)
        logger_obj.info("Total files analyzed: {:,}".format(total_files))
        logger_obj.info("Files Flagged: {:,} ({:.2f}%)".format(flagged_files, flag_rate*100))
        logger_obj.info("\nOutput files generated:")
        logger_obj.info("  1. {}".format(detected_path))
        logger_obj.info("  2. {}".format(features_path))
        logger_obj.info("  3. {}".format(summary_path))
        logger_obj.info("=" * 80)
            
        return {
            'success': True,
            'message': 'Model integration completed successfully',
            'file_count': total_files,
            'flagged_count': flagged_files,
            'results_df': df_results,
            'output_files': {
                'detected_files': str(detected_path),
                'files_with_features': str(features_path),
                'summary': str(summary_path)
            }
        }
    
    except Exception as e:
        error_msg = "Model integration failed: {}".format(str(e))
        logger_obj.error(error_msg, exc_info=True)
        return {
            'success': False,
            'message': error_msg,
            'file_count': total_files,
            'flagged_count': flagged_files,
            'output_files': None
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) >= 4:
        model_dir = Path(sys.argv[1])
        features_dir = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])
        
        results = run_model_integration(model_dir, features_dir, output_dir)
        
        logging.info("\nResults:")
        logging.info("  Success: {}".format(results['success']))
        logging.info("  Message: {}".format(results['message']))
        logging.info("  Files Analyzed: {}".format(results['file_count']))
        logging.info("  Files Flagged: {}".format(results['flagged_count']))
        if results['output_files']:
            logging.info("  Output Files:")
            for name, path in results['output_files'].items():
                logging.info("    - {}: {}".format(name, path))
        
        sys.exit(0 if results['success'] else 1)
    else:
        logging.info("Model Integration Module for NTFS Timestomping Detection")
        logging.info("\nUsage: python model_integration.py <model_dir> <features_dir> <output_dir>")
        logging.info("\nExample:")
        logging.info("  python model_integration.py \"ModelTraining\" \"File Features\" \"Detection Results\"")
        sys.exit(0)