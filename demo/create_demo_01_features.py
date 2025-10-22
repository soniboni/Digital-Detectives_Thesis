#!/usr/bin/env python3
"""
Create feature-engineered dataset for DEMO-01 (clean system - no timestomping).
This creates a dataset with ONLY benign files to demonstrate zero detections.
"""

import pandas as pd
from pathlib import Path
import sys

print("="*80)
print("CREATING DEMO-01 FEATURE-ENGINEERED DATASET (Clean System)")
print("="*80)

# Load the full engineered features
features_file = Path("../data/processed/Phase 2 - Feature Engineering/features_engineered.csv")

if not features_file.exists():
    print(f"Error: {features_file} not found")
    sys.exit(1)

print("\n[1/3] Loading engineered features...")
df = pd.read_csv(features_file, low_memory=False)
print(f"✓ Loaded {len(df):,} events")

# Filter for Case 6 only
if 'case_id' not in df.columns:
    print("Error: No case_id column found")
    sys.exit(1)

case_6 = df[df['case_id'] == 6].copy()
print(f"✓ Filtered to Case 6: {len(case_6):,} events")

# Get ONLY benign files (NO timestomped!)
benign = case_6[case_6['is_timestomped'] == 0.0].copy()

print(f"\n[2/3] Selecting ONLY benign files...")
print(f"  Total benign in Case 6: {len(benign):,}")

# Sample benign files (different from DEMO-02 and DEMO-03)
demo01_benign = benign.sample(
    n=min(2500, len(benign)),
    random_state=111  # Different seed
)

print(f"  Sampled for DEMO-01: {len(demo01_benign):,}")

# Verify NO timestomped files
timestomped_count = demo01_benign['is_timestomped'].sum()
print(f"\n  Timestomped files: {int(timestomped_count)}")

if timestomped_count > 0:
    print(f"  ⚠️  WARNING: Found timestomped files! This should be clean.")
else:
    print(f"  ✓ VERIFIED: Zero timestomped files (clean system)")

# Shuffle
demo01_df = demo01_benign.sample(frac=1, random_state=111).reset_index(drop=True)

print(f"\n[3/3] Finalizing DEMO-01 dataset...")
print(f"  Total events: {len(demo01_df):,}")
print(f"  Timestomped: {int((demo01_df['is_timestomped'] == 1.0).sum())}")
print(f"  Benign: {int((demo01_df['is_timestomped'] == 0.0).sum())}")

# Save
output_file = Path("test csv/DEMO-01-FeatureEngineered.csv")
demo01_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✓ SAVED: {output_file}")

print("\n" + "="*80)
print("DEMO-01 READY! (Should show ZERO detections)")
print("="*80)
print("\nTo run DEMO-01 (clean system demonstration):")
print("  python full_pipeline_demo_fixed.py \\")
print('    "test csv/DEMO-01-LogFile.csv" \\')
print('    "test csv/DEMO-01-UsnJrnl.csv" \\')
print("    --output-dir results_demo_01 \\")
print("    --verbose")
print("\n" + "="*80)