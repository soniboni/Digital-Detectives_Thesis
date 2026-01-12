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
        help='Path to directory where parsed CSV files will be saved'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for external processor.
    
    Orchestrates:
    1. Raw file parsing ($MFT, $LogFile, $UsnJrnl)
    2. Future: Data preprocessing
    3. Future: Feature engineering
    4. Future: ML model inference
    
    Returns parsed results as JSON to stdout.
    """
    
    logger = setup_logging()
    
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        exported_files_dir = Path(args.exported_dir)
        output_dir = Path(args.output_dir)
        
        logger.info("=" * 80)
        logger.info("NTFS Timestomping Detection - External Processor")
        logger.info("=" * 80)
        logger.info(f"Exported files directory: {exported_files_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Validate input directory
        if not exported_files_dir.exists():
            raise ValueError(f"Exported files directory does not exist: {exported_files_dir}")
        
        # Validate output directory
        if not output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
        
        # --- PHASE 1: RAW FILE PARSING ---
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: RAW FILE PARSING")
        logger.info("=" * 80)
        
        parser = RawFilesParser(logger_obj=logger)
        parse_results = parser.parse_all(exported_files_dir, output_dir)
        
        # Prepare results dictionary
        results = {
            'parsing': {}
        }
        
        # Process parsing results
        for file_type, (success, message, dataframe) in parse_results.items():
            results['parsing'][file_type] = {
                'success': success,
                'message': message,
                'records': len(dataframe) if dataframe is not None else 0
            }
            
            if success:
                logger.info(f"✓ {file_type}: {message}")
            else:
                logger.warning(f"✗ {file_type}: {message}")
        
        # --- FUTURE PHASES ---
        # PHASE 2: Data Preprocessing
        # PHASE 3: Feature Engineering
        # PHASE 4: ML Model Inference
        
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        
        # Output results as JSON to stdout (for the Jython invoker to parse)
        print(json.dumps(results, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in external processor: {str(e)}", exc_info=True)
        
        # Output error as JSON
        error_result = {
            'parsing': {},
            'error': str(e)
        }
        print(json.dumps(error_result, indent=2))
        
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
