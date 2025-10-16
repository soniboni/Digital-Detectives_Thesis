#!/usr/bin/env python3
"""
=============================================================================
TIMESTOMPING DETECTION - QUICK DEMO (OPTION 1)
=============================================================================

This script demonstrates timestomping detection using the trained Random Forest
model on already-engineered features.

USAGE:
------
    python predict_timestomping.py <input_features.csv> [options]

REQUIRED INPUT:
---------------
    - Engineered features CSV (output from Phase 2 - Feature Engineering)
    - Must contain the same 75 features the model was trained on
    - Does NOT need labels (is_timestomped column)

EXPECTED OUTPUT:
----------------
    1. predictions.csv - Contains:
       - All original columns from input
       - prediction: 0 (Benign) or 1 (Timestomped)
       - confidence: Probability score (0.0 to 1.0)
       - risk_level: LOW, MEDIUM, HIGH based on confidence

    2. flagged_files.csv - Only files predicted as timestomped:
       - Sorted by confidence (highest risk first)
       - Ready for forensic investigation

    3. summary_report.txt - Statistics and model performance

EXAMPLE USAGE:
--------------
    # Basic usage
    python predict_timestomping.py ../data/processed/Phase\\ 2\\ -\\ Feature\\ Engineering/features_engineered.csv

    # Specify output directory
    python predict_timestomping.py input.csv --output-dir ./results

    # Set custom confidence threshold
    python predict_timestomping.py input.csv --threshold 0.5

OPTIONS:
--------
    --output-dir DIR     Directory to save results (default: ./demo_results)
    --threshold FLOAT    Confidence threshold for flagging (default: 0.3)
    --verbose            Show detailed progress

REQUIREMENTS:
-------------
    - Trained model: data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib
    - Python packages: pandas, numpy, scikit-learn, joblib

=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from datetime import datetime
import sys

# ANSI color codes for terminal output
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

def load_model(model_path):
    """
    Load the trained Random Forest model

    Args:
        model_path: Path to the .joblib model file

    Returns:
        Loaded model object
    """
    print_info(f"Loading model from: {model_path}")

    if not Path(model_path).exists():
        print_error(f"Model file not found: {model_path}")
        print_info("Make sure you have trained the model (Phase 3 - Model Training v3)")
        sys.exit(1)

    model = joblib.load(model_path)
    print_success(f"Model loaded successfully!")
    print_info(f"   Model type: {type(model).__name__}")
    print_info(f"   Number of features expected: {model.n_features_in_}")
    print_info(f"   Number of trees: {model.n_estimators}")

    return model

def load_features(input_csv, verbose=False):
    """
    Load engineered features from CSV

    Args:
        input_csv: Path to input CSV file
        verbose: Print detailed information

    Returns:
        DataFrame with features
    """
    print_info(f"Loading features from: {input_csv}")

    if not Path(input_csv).exists():
        print_error(f"Input file not found: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    print_success(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    if verbose:
        print_info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print_info(f"   Column preview: {', '.join(df.columns[:5].tolist())}...")

    return df

def prepare_features(df, model, verbose=False):
    """
    Prepare features for prediction (exclude non-feature columns)

    Args:
        df: Input DataFrame
        model: Trained model (to get expected features)
        verbose: Print detailed information

    Returns:
        X: Feature matrix ready for prediction
        metadata_cols: List of columns to preserve in output
    """
    print_info("Preparing features for prediction...")

    # Columns to exclude from features (but keep for output)
    cols_to_exclude = [
        'is_timestomped',
        'is_timestomped_lf',
        'is_timestomped_usn',
        'timestomp_tool_executed',
        'timestomp_tool_executed_lf',
        'timestomp_tool_executed_usn',
        'case_id',
        'eventtime_dt',
        'label_source_both',
        'label_source_logfile',
        'label_source_usnjrnl',
        'label_source_nan',
    ]

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in cols_to_exclude]

    # Check if we have the expected number of features
    if len(feature_cols) != model.n_features_in_:
        print_warning(f"Feature count mismatch!")
        print_warning(f"   Model expects: {model.n_features_in_} features")
        print_warning(f"   Input has: {len(feature_cols)} features")
        print_info("Attempting to proceed anyway...")

    # Extract features
    X = df[feature_cols].copy()

    # Convert bool to int
    bool_cols = X.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
        for col in bool_cols:
            X[col] = X[col].astype(int)

    # Fill missing values
    X = X.fillna(0)

    # Metadata columns to preserve
    metadata_cols = [col for col in df.columns if col not in feature_cols]

    print_success(f"Features prepared!")
    print_info(f"   Feature matrix shape: {X.shape}")
    print_info(f"   Metadata columns preserved: {len(metadata_cols)}")

    if verbose and metadata_cols:
        print_info(f"   Metadata: {', '.join(metadata_cols[:5])}...")

    return X, metadata_cols

def make_predictions(model, X, threshold=0.3, verbose=False):
    """
    Generate predictions using the trained model

    Args:
        model: Trained Random Forest model
        X: Feature matrix
        threshold: Confidence threshold for flagging (0.0 to 1.0)
        verbose: Print detailed information

    Returns:
        predictions: Binary predictions (0 or 1)
        probabilities: Confidence scores (0.0 to 1.0)
        risk_levels: Risk categorization (LOW, MEDIUM, HIGH)
    """
    print_info(f"Generating predictions on {len(X):,} samples...")

    # Get predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of timestomping

    # Categorize risk levels
    risk_levels = []
    for prob in probabilities:
        if prob < threshold:
            risk_levels.append('LOW')
        elif prob < 0.7:
            risk_levels.append('MEDIUM')
        else:
            risk_levels.append('HIGH')

    print_success("Predictions complete!")
    print_info(f"   Total predictions: {len(predictions):,}")
    print_info(f"   Flagged as timestomped: {predictions.sum():,} ({predictions.sum()/len(predictions)*100:.2f}%)")
    print_info(f"   High risk: {risk_levels.count('HIGH'):,}")
    print_info(f"   Medium risk: {risk_levels.count('MEDIUM'):,}")
    print_info(f"   Low risk: {risk_levels.count('LOW'):,}")

    return predictions, probabilities, risk_levels

def save_results(df, predictions, probabilities, risk_levels, metadata_cols, output_dir, verbose=False):
    """
    Save prediction results to files

    Args:
        df: Original DataFrame
        predictions: Binary predictions
        probabilities: Confidence scores
        risk_levels: Risk categorizations
        metadata_cols: Columns to include in output
        output_dir: Directory to save results
        verbose: Print detailed information

    Returns:
        paths: Dictionary of output file paths
    """
    print_info(f"Saving results to: {output_dir}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create results DataFrame
    results_df = df.copy()
    results_df['prediction'] = predictions
    results_df['confidence'] = probabilities
    results_df['risk_level'] = risk_levels

    # 1. Save all predictions
    predictions_path = output_dir / 'predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    print_success(f"Saved all predictions: {predictions_path}")

    # 2. Save flagged files only (timestomped predictions)
    flagged_df = results_df[results_df['prediction'] == 1].copy()
    flagged_df = flagged_df.sort_values('confidence', ascending=False)
    flagged_path = output_dir / 'flagged_files.csv'
    flagged_df.to_csv(flagged_path, index=False)
    print_success(f"Saved flagged files: {flagged_path} ({len(flagged_df):,} files)")

    # 3. Generate summary report
    summary_path = output_dir / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TIMESTOMPING DETECTION - PREDICTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Random Forest (v3 Final - Minimal SMOTE)\n\n")

        f.write("INPUT:\n")
        f.write(f"  Total files analyzed: {len(results_df):,}\n\n")

        f.write("PREDICTIONS:\n")
        f.write(f"  Flagged as timestomped: {predictions.sum():,} ({predictions.sum()/len(predictions)*100:.2f}%)\n")
        f.write(f"  Predicted benign: {(predictions == 0).sum():,} ({(predictions == 0).sum()/len(predictions)*100:.2f}%)\n\n")

        f.write("RISK BREAKDOWN:\n")
        f.write(f"  HIGH risk (≥70% confidence): {risk_levels.count('HIGH'):,}\n")
        f.write(f"  MEDIUM risk (30-70% confidence): {risk_levels.count('MEDIUM'):,}\n")
        f.write(f"  LOW risk (<30% confidence): {risk_levels.count('LOW'):,}\n\n")

        if len(flagged_df) > 0:
            f.write("TOP 10 HIGHEST CONFIDENCE DETECTIONS:\n")
            f.write("-"*80 + "\n")
            for idx, row in flagged_df.head(10).iterrows():
                # Get filename if available
                filename = "N/A"
                if 'filename' in row:
                    filename = row['filename']
                elif 'lf_filename' in row:
                    filename = row['lf_filename']
                elif 'usn_filename' in row:
                    filename = row['usn_filename']

                f.write(f"  {row['confidence']:.3f} | {row['risk_level']:6s} | {filename}\n")
            f.write("\n")

        f.write("FORENSIC TRIAGE VALUE:\n")
        f.write(f"  Files requiring investigation: {len(flagged_df):,}\n")
        f.write(f"  Investigation reduction: {(1 - len(flagged_df)/len(results_df))*100:.2f}%\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  1. predictions.csv - All predictions with confidence scores\n")
        f.write(f"  2. flagged_files.csv - Only timestomped predictions (sorted by confidence)\n")
        f.write(f"  3. summary_report.txt - This report\n\n")

        f.write("="*80 + "\n")

    print_success(f"Saved summary report: {summary_path}")

    return {
        'predictions': predictions_path,
        'flagged': flagged_path,
        'summary': summary_path
    }

def main():
    """Main execution function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Timestomping Detection - Quick Demo (Option 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_csv', type=str,
                        help='Path to engineered features CSV')
    parser.add_argument('--output-dir', type=str, default='./demo_results',
                        help='Directory to save results (default: ./demo_results)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Confidence threshold for flagging (default: 0.3)')
    parser.add_argument('--model-path', type=str,
                        default='../data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib',
                        help='Path to trained model')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress')

    args = parser.parse_args()

    # Print header
    print_header("TIMESTOMPING DETECTION - QUICK DEMO")
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Verbose: {args.verbose}\n")

    try:
        # Step 1: Load model
        print_header("STEP 1: LOAD MODEL")
        model = load_model(args.model_path)

        # Step 2: Load features
        print_header("STEP 2: LOAD FEATURES")
        df = load_features(args.input_csv, verbose=args.verbose)

        # Step 3: Prepare features
        print_header("STEP 3: PREPARE FEATURES")
        X, metadata_cols = prepare_features(df, model, verbose=args.verbose)

        # Step 4: Make predictions
        print_header("STEP 4: GENERATE PREDICTIONS")
        predictions, probabilities, risk_levels = make_predictions(
            model, X, threshold=args.threshold, verbose=args.verbose
        )

        # Step 5: Save results
        print_header("STEP 5: SAVE RESULTS")
        output_paths = save_results(
            df, predictions, probabilities, risk_levels,
            metadata_cols, args.output_dir, verbose=args.verbose
        )

        # Final summary
        print_header("DEMO COMPLETE!")
        print_success("Timestomping detection completed successfully!")
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
        print_error(f"Error occurred: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()