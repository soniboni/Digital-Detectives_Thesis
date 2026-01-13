# -*- coding: utf-8 -*-

"""
External Processor - NTFS Timestomping Detection Main Entry Point

This module serves as the main entry point for the Python 3 external processing layer.
It orchestrates parsing of NTFS system files and prepares data for feature engineering
and ML inference.

Environment: Python 3.x
Purpose: Main orchestrator for external data processing
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directories to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from Parser.raw_files_parser import RawFilesParser
from ModelPackage.data_preprocessing import DataPreprocessor
from ModelPackage.feature_engineering import FeatureEngineer


def setup_logging():
    """Configure logging for the external processor."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='NTFS Timestomping Detection - External Processor'
    )
    parser.add_argument(
        '--exported-dir',
        required=True,
        type=str,
        help='Path to directory containing exported $MFT, $LogFile, and $UsnJrnl files'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=str,
        help='Path to module output root directory (containing Parsed Files, Grouped Events File, etc.)'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for external processor.
    
    Orchestrates:
    1. Raw file parsing ($MFT, $LogFile, $UsnJrnl)
    2. Data preprocessing (temporal alignment and event grouping)
    3. Feature engineering (forensic feature extraction)
    4. Future: ML model inference
    5. Future: Report generation
    
    Returns parsed results as JSON to stdout.
    """
    
    logger = setup_logging()
    
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        exported_files_dir = Path(args.exported_dir)
        module_output_root = Path(args.output_dir)
        
        # Define stage directories (matching ntfs_timestomping_detector.py)
        parsed_files_dir = module_output_root / "Parsed Files"
        grouped_events_dir = module_output_root / "Grouped Events File"
        features_dir = module_output_root / "File Features"
        results_dir = module_output_root / "Detection Results"
        
        logger.info("=" * 80)
        logger.info("NTFS Timestomping Detection - External Processor")
        logger.info("=" * 80)
        logger.info("Module output root: {}".format(module_output_root))
        logger.info("Exported files directory: {}".format(exported_files_dir))
        logger.info("Parsed files directory: {}".format(parsed_files_dir))
        logger.info("Grouped events directory: {}".format(grouped_events_dir))
        
        # Validate input directory
        if not exported_files_dir.exists():
            raise ValueError("Exported files directory does not exist: {}".format(exported_files_dir))
        
        # Validate and create output directories
        if not parsed_files_dir.exists():
            raise ValueError("Parsed Files directory does not exist: {}".format(parsed_files_dir))
        
        grouped_events_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare results dictionary
        results = {
            'parsing': {},
            'preprocessing': {},
            'feature_engineering': {}
        }
        
        # --- STAGE 1: RAW FILE PARSING ---
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: RAW FILE PARSING")
        logger.info("=" * 80)
        
        parser = RawFilesParser(logger_obj=logger)
        parse_results = parser.parse_all(exported_files_dir, parsed_files_dir)
        
        # Process parsing results
        parsing_success = False
        for file_type, (success, message, dataframe) in parse_results.items():
            results['parsing'][file_type] = {
                'success': success,
                'message': message,
                'records': len(dataframe) if dataframe is not None else 0
            }
            
            if success:
                logger.info("✓ {}: {}".format(file_type, message))
                parsing_success = True
            else:
                logger.warning("✗ {}: {}".format(file_type, message))
        
        # Only proceed to preprocessing if parsing succeeded
        if not parsing_success:
            logger.warning("Parsing completed with errors - skipping preprocessing")
            results['preprocessing']['success'] = False
            results['preprocessing']['message'] = 'Skipped due to parsing errors'
        else:
            # --- STAGE 2: DATA PREPROCESSING ---
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: DATA PREPROCESSING")
            logger.info("=" * 80)
            
            try:
                preprocessor = DataPreprocessor(logger_obj=logger)
                preprocess_results = preprocessor.preprocess_all(parsed_files_dir, grouped_events_dir)
                
                results['preprocessing'] = {
                    'success': preprocess_results['success'],
                    'message': preprocess_results['message'],
                    'grouped_events_csv': preprocess_results['grouped_events_csv'],
                    'event_count': preprocess_results['event_count']
                }
                
                if preprocess_results['success']:
                    logger.info("✓ Preprocessing: {}".format(preprocess_results['message']))
                else:
                    logger.warning("✗ Preprocessing: {}".format(preprocess_results['message']))
            
            except Exception as e:
                error_msg = "Preprocessing error: {}".format(str(e))
                logger.error(error_msg)
                results['preprocessing'] = {
                    'success': False,
                    'message': error_msg,
                    'grouped_events_csv': None,
                    'event_count': 0
                }
            
            # --- STAGE 3: FEATURE ENGINEERING ---
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 3: FEATURE ENGINEERING")
            logger.info("=" * 80)
            
            try:
                fe = FeatureEngineer(logger_obj=logger)
                feature_results = fe.extract_features(grouped_events_dir, features_dir)
                
                results['feature_engineering'] = {
                    'success': feature_results['success'],
                    'message': feature_results['message'],
                    'file_features_csv': feature_results['file_features_csv'],
                    'file_count': feature_results['file_count']
                }
                
                if feature_results['success']:
                    logger.info("✓ Feature Engineering: {}".format(feature_results['message']))
                else:
                    logger.warning("✗ Feature Engineering: {}".format(feature_results['message']))
            
            except Exception as e:
                error_msg = "Feature engineering error: {}".format(str(e))
                logger.error(error_msg)
                results['feature_engineering'] = {
                    'success': False,
                    'message': error_msg,
                    'file_features_csv': None,
                    'file_count': 0
                }
        
        # --- FUTURE STAGES ---
        # STAGE 4: ML Model Inference
        # STAGE 5: Report Generation
        
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        
        # Output results as JSON to stdout (for the Jython invoker to parse)
        print(json.dumps(results, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error("Fatal error in external processor: {}".format(str(e)), exc_info=True)
        
        # Output error as JSON
        error_result = {
            'parsing': {},
            'preprocessing': {},
            'error': str(e)
        }
        print(json.dumps(error_result, indent=2))
        
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)