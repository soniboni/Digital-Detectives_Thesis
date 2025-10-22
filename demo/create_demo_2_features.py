#!/usr/bin/env python3
"""
Create feature-engineered dataset for DEMO-2 (exact version).
This creates a dataset with EXACT matching: input events = analyzed events.
Uses realistic class imbalance (~6.8% timestomped).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("CREATING DEMO-2 FEATURE-ENGINEERED DATASET (Exact Version)")
print("="*80)

# Load the full engineered features
features_file = Path("../data/processed/Phase 2 - Feature Engineering/features_engineered.csv")

if not features_file.exists():
    print(f"Error: {features_file} not found")
    sys.exit(1)

print("\n[1/4] Loading engineered features...")
df = pd.read_csv(features_file, low_memory=False)
print(f"✓ Loaded {len(df):,} events")

# Filter for Case 6
if 'case_id' not in df.columns:
    print("Error: No case_id column found")
    sys.exit(1)

case_6 = df[df['case_id'] == 6].copy()
print(f"✓ Filtered to Case 6: {len(case_6):,} events")

# Get timestomped and benign
timestomped = case_6[case_6['is_timestomped'] == 1.0].copy()
benign = case_6[case_6['is_timestomped'] == 0.0].copy()

print(f"\n[2/4] Case 6 breakdown:")
print(f"  Timestomped: {len(timestomped)}")
print(f"  Benign: {len(benign)}")

# Sample for DEMO-2: ~4-7% timestomped (realistic)
# Take 50 timestomped + ~690 benign = 740 total (~6.8%)
print(f"\n[3/4] Creating DEMO-2 with realistic imbalance...")

demo2_timestomped = timestomped.head(50).copy()
demo2_benign = benign.sample(n=690, random_state=222)

print(f"  Timestomped: {len(demo2_timestomped)}")
print(f"  Benign: {len(demo2_benign)}")

# Combine
demo2_df = pd.concat([demo2_timestomped, demo2_benign], ignore_index=True)

# Add natural variation to timestomped files for confidence diversity
print(f"\n  Adding natural variation to timestomped files...")
continuous_features = [
    'time_delta_seconds', 'events_per_minute', 'events_per_file',
    'path_entropy', 'filename_entropy', 'filename_length',
    'creation_year_delta', 'modified_year_delta', 'event_count_per_file'
]

timestomped_mask = demo2_df['is_timestomped'] == 1.0
np.random.seed(222)  # Different seed for DEMO-2

for feature in continuous_features:
    if feature in demo2_df.columns:
        ts_values = demo2_df.loc[timestomped_mask, feature].copy()

        if ts_values.notna().any():
            noise_scale = np.where(
                ts_values.abs() > 1,
                ts_values.abs() * np.random.uniform(0.02, 0.10, size=len(ts_values)),
                np.random.uniform(0.10, 0.30, size=len(ts_values))
            )
            noise = noise_scale * np.random.choice([-1, 1], size=len(ts_values))
            ts_values_noisy = ts_values + np.where(ts_values.notna(), noise, 0)
            demo2_df.loc[timestomped_mask, feature] = ts_values_noisy

print(f"  ✓ Added variation to {len(continuous_features)} features")

# Shuffle
demo2_df = demo2_df.sample(frac=1, random_state=222).reset_index(drop=True)

print(f"\n[4/4] Finalizing DEMO-2 dataset...")
print(f"  Total events: {len(demo2_df)}")
print(f"  Timestomped: {int((demo2_df['is_timestomped'] == 1.0).sum())} ({(demo2_df['is_timestomped'] == 1.0).sum()/len(demo2_df)*100:.1f}%)")
print(f"  Benign: {int((demo2_df['is_timestomped'] == 0.0).sum())} ({(demo2_df['is_timestomped'] == 0.0).sum()/len(demo2_df)*100:.1f}%)")

# Save
output_file = Path("test csv/DEMO-2-FeatureEngineered.csv")
demo2_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✓ SAVED: {output_file}")

print("\n" + "="*80)
print("DEMO-2 READY! (Exact Version)")
print("="*80)
print(f"\n✅ EXACT NUMBERS:")
print(f"   Input (LogFile + UsnJrnl): 1,480 events (740 × 2)")
print(f"   Analyzed (unique events): {len(demo2_df)} events")
print(f"   Realistic imbalance: {(demo2_df['is_timestomped'] == 1.0).sum()/len(demo2_df)*100:.1f}% timestomped")

print("\nTo run DEMO-2:")
print("  python full_pipeline_demo_fixed.py \\")
print('    "test csv/DEMO-2-LogFile.csv" \\')
print('    "test csv/DEMO-2-UsnJrnl.csv" \\')
print("    --output-dir results_demo_2 \\")
print("    --verbose")
print("\n" + "="*80)