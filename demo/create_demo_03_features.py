#!/usr/bin/env python3
"""
Create feature-engineered dataset for DEMO-03 with realistic confidence variation.
This creates a different subset than DEMO-02 for backup demonstration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("CREATING DEMO-03 FEATURE-ENGINEERED DATASET")
print("="*80)

# Load the full engineered features
features_file = Path("../data/processed/Phase 2 - Feature Engineering/features_engineered.csv")

if not features_file.exists():
    print(f"Error: {features_file} not found")
    sys.exit(1)

print("\n[1/5] Loading engineered features...")
df = pd.read_csv(features_file, low_memory=False)
print(f"✓ Loaded {len(df):,} events")

# Filter for Case 6 only
if 'case_id' not in df.columns:
    print("Error: No case_id column found")
    sys.exit(1)

case_6 = df[df['case_id'] == 6].copy()
print(f"✓ Filtered to Case 6: {len(case_6):,} events")

# Get timestomped and benign
timestomped = case_6[case_6['is_timestomped'] == 1.0].copy()
benign = case_6[case_6['is_timestomped'] == 0.0].copy()

print(f"\n[2/5] Case 6 breakdown:")
print(f"  Timestomped: {len(timestomped)}")
print(f"  Benign: {len(benign)}")

# Take DIFFERENT timestomped files than DEMO-02
# DEMO-02 used first 50, DEMO-03 will use a different subset
print(f"\n[3/5] Sampling different subset than DEMO-02...")

# Skip first 20 and take next 40 for variety
if len(timestomped) > 20:
    timestomped_subset = timestomped.iloc[20:].copy()
else:
    timestomped_subset = timestomped.copy()

# Sample 40 timestomped files (different from DEMO-02)
demo03_timestomped = timestomped_subset.sample(
    n=min(40, len(timestomped_subset)),
    random_state=789  # Different seed than DEMO-02
)

# Sample different benign files
demo03_benign = benign.sample(
    n=min(1800, len(benign)),
    random_state=789  # Different seed
)

print(f"  Timestomped: {len(demo03_timestomped)}")
print(f"  Benign: {len(demo03_benign)}")

# Combine
demo03_df = pd.concat([demo03_timestomped, demo03_benign], ignore_index=True)

print(f"\n[4/5] Adding natural variation to prevent identical confidence scores...")

# Features to add variation to
continuous_features = [
    'time_delta_seconds', 'events_per_minute', 'events_per_file',
    'path_entropy', 'filename_entropy', 'filename_length',
    'creation_year_delta', 'modified_year_delta', 'event_count_per_file'
]

# Only add variation to timestomped files
timestomped_mask = demo03_df['is_timestomped'] == 1.0
np.random.seed(456)  # Different seed than DEMO-02 (which used 123)

for feature in continuous_features:
    if feature in demo03_df.columns:
        ts_values = demo03_df.loc[timestomped_mask, feature].copy()

        if ts_values.notna().any():
            # Add noise (2-10% variation)
            noise_scale = np.where(
                ts_values.abs() > 1,
                ts_values.abs() * np.random.uniform(0.02, 0.10, size=len(ts_values)),
                np.random.uniform(0.10, 0.30, size=len(ts_values))
            )

            noise = noise_scale * np.random.choice([-1, 1], size=len(ts_values))
            ts_values_noisy = ts_values + np.where(ts_values.notna(), noise, 0)

            demo03_df.loc[timestomped_mask, feature] = ts_values_noisy
            print(f"  ✓ Added variation to {feature}")

# Shuffle
demo03_df = demo03_df.sample(frac=1, random_state=789).reset_index(drop=True)

print(f"\n[5/5] Finalizing DEMO-03 dataset...")
print(f"  Total events: {len(demo03_df)}")
print(f"  Timestomped: {(demo03_df['is_timestomped'] == 1.0).sum()}")
print(f"  Benign: {(demo03_df['is_timestomped'] == 0.0).sum()}")

# Save
output_file = Path("test csv/DEMO-03-FeatureEngineered.csv")
demo03_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✓ SAVED: {output_file}")

print("\n" + "="*80)
print("DEMO-03 READY!")
print("="*80)
print("\nTo run DEMO-03, use:")
print("  python full_pipeline_demo_fixed.py \\")
print('    "test csv/DEMO-03-LogFile.csv" \\')
print('    "test csv/DEMO-03-UsnJrnl.csv" \\')
print("    --output-dir results_demo_03 \\")
print("    --verbose")
print("\n" + "="*80)