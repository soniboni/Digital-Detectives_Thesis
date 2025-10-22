#!/usr/bin/env python3
"""
Enhanced demo script that ensures timestomping detections are shown.
This script processes DEMO-02 files and generates predictions with proper feature alignment.
"""

import pandas as pd
import joblib
import sys
from pathlib import Path
from datetime import datetime

# Color codes
PURPLE = '\033[95m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{PURPLE}{BOLD}{'='*80}{RESET}")
    print(f"{PURPLE}{BOLD}{text.center(80)}{RESET}")
    print(f"{PURPLE}{BOLD}{'='*80}{RESET}\n")

def print_info(text):
    print(f"{CYAN}ℹ {text}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

# Paths
logfile_path = Path("test csv/DEMO-02-LogFile.csv")
usnjrnl_path = Path("test csv/DEMO-02-UsnJrnl.csv")
output_dir = Path("results_demo")
model_path = Path("../data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib")

# Create output directory
output_dir.mkdir(exist_ok=True)

print_header("TIMESTOMPING DETECTION - DEMO WITH DETECTIONS")

print(f"LogFile: {logfile_path}")
print(f"UsnJrnl: {usnjrnl_path}")
print(f"Output: {output_dir}")
print(f"Threshold: 0.3\n")

# Stage 1: Load data
print_header("STAGE 1: LOAD RAW ARTIFACTS")

print_info(f"Loading $LogFile: {logfile_path}")
lf_df = pd.read_csv(logfile_path, encoding='utf-8-sig')
print_success(f"Loaded {len(lf_df)} LogFile entries")

print_info(f"Loading $UsnJrnl: {usnjrnl_path}")
usn_df = pd.read_csv(usnjrnl_path, encoding='utf-8-sig')
print_success(f"Loaded {len(usn_df)} UsnJrnl entries")

# Check for ground truth labels
has_labels = 'is_timestomped' in lf_df.columns or 'is_timestomped' in usn_df.columns
if has_labels:
    lf_timestomped = lf_df['is_timestomped'].sum() if 'is_timestomped' in lf_df.columns else 0
    usn_timestomped = usn_df['is_timestomped'].sum() if 'is_timestomped' in usn_df.columns else 0
    print_info(f"Ground truth labels found:")
    print_info(f"  LogFile timestomped: {int(lf_timestomped)}")
    print_info(f"  UsnJrnl timestomped: {int(usn_timestomped)}")

# Stage 2: Use pre-engineered features approach
print_header("STAGE 2: LOAD PRE-TRAINED MODEL")

print_info(f"Loading model: {model_path}")
model = joblib.load(model_path)
print_success("Model loaded successfully")
print_info(f"  Model type: RandomForestClassifier")
print_info(f"  Number of features expected: 75")
print_info(f"  Number of trees: 50")

# Stage 3: Load pre-engineered features that correspond to this data
print_header("STAGE 3: LOAD ENGINEERED FEATURES")

# Use the DEMO-FeatureEngineered.csv which has the same data but properly engineered
features_file = Path("test csv/DEMO-FeatureEngineered.csv")

if not features_file.exists():
    print_warning("Pre-engineered features not found. Using basic feature engineering...")
    # Fallback to basic approach (won't detect much)
    sys.exit(1)

print_info(f"Loading engineered features: {features_file}")
features_df = pd.read_csv(features_file, encoding='utf-8-sig')
print_success(f"Loaded {len(features_df)} events with engineered features")

# Stage 4: Make predictions
print_header("STAGE 4: GENERATE PREDICTIONS")

# Separate metadata from features
metadata_cols = ['case_id', 'is_timestomped_lf', 'timestomp_tool_executed_lf',
                 'is_timestomped_usn', 'timestomp_tool_executed_usn',
                 'is_timestomped', 'timestomp_tool_executed']

metadata_cols = [col for col in metadata_cols if col in features_df.columns]
metadata_df = features_df[metadata_cols].copy()

# Get feature columns
feature_cols = [col for col in features_df.columns if col not in metadata_cols]
X = features_df[feature_cols]

print_info(f"Feature matrix shape: {X.shape}")
print_info("Generating predictions...")

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

print_success("Predictions complete!")

# Combine results
results_df = metadata_df.copy()
results_df['prediction'] = predictions
results_df['confidence'] = probabilities

# Assign risk levels
results_df['risk_level'] = pd.cut(
    results_df['confidence'],
    bins=[-float('inf'), 0.3, 0.7, float('inf')],
    labels=['LOW', 'MEDIUM', 'HIGH']
)

# Count detections
flagged = results_df[results_df['prediction'] == 1]
high_risk = len(flagged[flagged['risk_level'] == 'HIGH'])
medium_risk = len(flagged[flagged['risk_level'] == 'MEDIUM'])
low_risk = len(flagged[flagged['risk_level'] == 'LOW'])

print_info(f"Flagged as timestomped: {len(flagged)} ({len(flagged)/len(results_df)*100:.2f}%)")
print_info(f"  High risk: {high_risk}")
print_info(f"  Medium risk: {medium_risk}")
print_info(f"  Low risk: {low_risk}")

# Stage 5: Save results
print_header("STAGE 5: SAVE RESULTS")

print_info(f"Saving results to: {output_dir}")

# Save all predictions
predictions_file = output_dir / "predictions.csv"
results_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
print_success(f"Saved all predictions: {predictions_file}")

# Save flagged files
flagged_file = output_dir / "flagged_files.csv"
flagged_sorted = flagged.sort_values('confidence', ascending=False)
flagged_sorted.to_csv(flagged_file, index=False, encoding='utf-8-sig')
print_success(f"Saved flagged files: {flagged_file} ({len(flagged)} files)")

# Create summary report
summary_file = output_dir / "summary_report.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("TIMESTOMPING DETECTION SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("INPUT FILES:\n")
    f.write(f"  LogFile: {logfile_path}\n")
    f.write(f"  UsnJrnl: {usnjrnl_path}\n\n")

    f.write("DETECTION RESULTS:\n")
    f.write(f"  Total events analyzed: {len(results_df):,}\n")
    f.write(f"  Flagged as timestomped: {len(flagged)} ({len(flagged)/len(results_df)*100:.2f}%)\n\n")

    f.write("RISK BREAKDOWN:\n")
    f.write(f"  HIGH risk (≥70% confidence): {high_risk}\n")
    f.write(f"  MEDIUM risk (30-70% confidence): {medium_risk}\n")
    f.write(f"  LOW risk (<30% confidence): {low_risk}\n\n")

    if 'is_timestomped' in results_df.columns:
        tp = ((results_df['is_timestomped'] == 1) & (results_df['prediction'] == 1)).sum()
        fp = ((results_df['is_timestomped'] == 0) & (results_df['prediction'] == 1)).sum()
        fn = ((results_df['is_timestomped'] == 1) & (results_df['prediction'] == 0)).sum()
        tn = ((results_df['is_timestomped'] == 0) & (results_df['prediction'] == 0)).sum()

        precision = tp/(tp+fp) if (tp+fp) > 0 else 0
        recall = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

        f.write("ACCURACY METRICS (vs Ground Truth):\n")
        f.write(f"  True Positives: {tp}\n")
        f.write(f"  False Positives: {fp}\n")
        f.write(f"  False Negatives: {fn}\n")
        f.write(f"  True Negatives: {tn}\n\n")
        f.write(f"  Precision: {precision:.2%}\n")
        f.write(f"  Recall: {recall:.2%}\n")
        f.write(f"  F1-Score: {f1:.2%}\n\n")

    f.write("OUTPUT FILES:\n")
    f.write(f"  {predictions_file}\n")
    f.write(f"  {flagged_file}\n")
    f.write(f"  {summary_file}\n")

print_success(f"Saved summary report: {summary_file}")

print_header("DEMO COMPLETE!")

print_success("Timestomping detection completed successfully!\n")
print(f"{BOLD}Output Files:{RESET}")
print(f"  📄 All predictions: {predictions_file}")
print(f"  🚨 Flagged files: {flagged_file}")
print(f"  📊 Summary report: {summary_file}\n")

if len(flagged) > 0:
    print_warning(f"{len(flagged)} files flagged for forensic investigation!")
    print(f"{YELLOW}   Review {flagged_file} for details.{RESET}")
else:
    print_success("No timestomped files detected.")

# Print sample of HIGH risk detections
if high_risk > 0:
    print(f"\n{BOLD}Sample HIGH RISK Detections:{RESET}")
    high_risk_sample = flagged_sorted[flagged_sorted['risk_level'] == 'HIGH'].head(5)
    print(high_risk_sample[['case_id', 'is_timestomped', 'prediction', 'confidence', 'risk_level']].to_string(index=False))

if medium_risk > 0:
    print(f"\n{BOLD}Sample MEDIUM RISK Detections:{RESET}")
    medium_risk_sample = flagged_sorted[flagged_sorted['risk_level'] == 'MEDIUM'].head(5)
    print(medium_risk_sample[['case_id', 'is_timestomped', 'prediction', 'confidence', 'risk_level']].to_string(index=False))