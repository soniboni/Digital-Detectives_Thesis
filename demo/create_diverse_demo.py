#!/usr/bin/env python3
"""
Create a diverse demonstration dataset with varied confidence levels.
This extracts timestomped files from MULTIPLE cases with different file types
to show realistic confidence score variation.
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np

print("Creating diverse demo dataset with varied confidence levels...\n")

# Load the full engineered features dataset
features_file = Path("../data/processed/Phase 2 - Feature Engineering/features_engineered.csv")

if not features_file.exists():
    print(f"Error: {features_file} not found")
    sys.exit(1)

print("Loading full engineered features...")
df = pd.read_csv(features_file, low_memory=False)
print(f"Loaded {len(df):,} events")

# Check for timestomping
if 'is_timestomped' not in df.columns:
    print("Error: No 'is_timestomped' column found")
    sys.exit(1)

timestomped_count = df['is_timestomped'].sum()
print(f"Found {int(timestomped_count)} timestomped events\n")

# Get timestomped and benign files
timestomped = df[df['is_timestomped'] == 1.0].copy()
benign = df[df['is_timestomped'] == 0.0].copy()

print("Analyzing timestomped files by case:")
if 'case_id' in timestomped.columns:
    case_breakdown = timestomped['case_id'].value_counts()
    print(case_breakdown)
    print()

# Strategy: Sample from multiple cases to get diverse file types and patterns
# This will create natural variation in confidence scores

# Sample timestomped files from different cases
sampled_timestomped = []

# If we have case_id, sample from multiple cases
if 'case_id' in timestomped.columns:
    cases = timestomped['case_id'].unique()
    print(f"Sampling from {len(cases)} different cases for diversity...")

    # Sample proportionally from each case
    for case in cases:
        case_data = timestomped[timestomped['case_id'] == case]
        # Take up to 30% from each case, minimum 5 if available
        sample_size = max(5, int(len(case_data) * 0.3))
        sample_size = min(sample_size, len(case_data))

        case_sample = case_data.sample(n=sample_size, random_state=42)
        sampled_timestomped.append(case_sample)
        print(f"  Case {case}: sampled {len(case_sample)} / {len(case_data)} timestomped files")

    demo_timestomped = pd.concat(sampled_timestomped, ignore_index=True)
else:
    # No case_id, just take a diverse random sample
    print("No case_id found, sampling randomly...")
    demo_timestomped = timestomped.sample(n=min(100, len(timestomped)), random_state=42)

# Sample benign files (larger sample for realistic class distribution)
demo_benign = benign.sample(n=min(2000, len(benign)), random_state=42)

# Combine
demo_df = pd.concat([demo_timestomped, demo_benign], ignore_index=True)

# Shuffle to mix timestomped and benign
demo_df = demo_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✓ Created diverse demo dataset:")
print(f"  Timestomped: {len(demo_timestomped)}")
print(f"  Benign: {len(demo_benign)}")
print(f"  Total: {len(demo_df)}")

# Save to test csv folder
output_file = Path("test csv/DEMO-FeatureEngineered-Diverse.csv")
demo_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: {output_file}")

print("\n" + "="*80)
print("To use this diverse dataset, run:")
print(f'  python full_pipeline_demo_fixed.py \\')
print(f'    "test csv/DEMO-02-LogFile.csv" \\')
print(f'    "test csv/DEMO-02-UsnJrnl.csv" \\')
print(f'    --verbose')
print()
print("Then update full_pipeline_demo_fixed.py line 141 to use:")
print(f'  features_file = Path("test csv/DEMO-FeatureEngineered-Diverse.csv")')
print("="*80)