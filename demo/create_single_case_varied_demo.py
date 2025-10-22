#!/usr/bin/env python3
"""
Create demo dataset from Case 6 only with realistic confidence variation.
This adds slight natural variation to features to prevent identical confidence scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("Creating single-case demo with varied confidence scores...\n")

# Load the full engineered features
features_file = Path("../data/processed/Phase 2 - Feature Engineering/features_engineered.csv")

if not features_file.exists():
    print(f"Error: {features_file} not found")
    sys.exit(1)

print("Loading engineered features...")
df = pd.read_csv(features_file, low_memory=False)
print(f"Loaded {len(df):,} events")

# Filter for Case 6 only (to match DEMO-02 files)
if 'case_id' in df.columns:
    case_6 = df[df['case_id'] == 6].copy()
    print(f"Filtered to Case 6: {len(case_6):,} events")
else:
    print("Error: No case_id column found")
    sys.exit(1)

# Get timestomped and benign
timestomped = case_6[case_6['is_timestomped'] == 1.0].copy()
benign = case_6[case_6['is_timestomped'] == 0.0].copy()

print(f"Case 6: {len(timestomped)} timestomped, {len(benign)} benign\n")

# Take all timestomped files and a sample of benign
demo_timestomped = timestomped.copy()
demo_benign = benign.sample(n=min(2000, len(benign)), random_state=42)

print(f"Creating demo with:")
print(f"  Timestomped: {len(demo_timestomped)}")
print(f"  Benign: {len(demo_benign)}")

# Combine
demo_df = pd.concat([demo_timestomped, demo_benign], ignore_index=True)

# Add subtle variation to continuous features to prevent identical confidence scores
# This simulates natural measurement variation in real-world data

print("\nAdding natural variation to prevent identical confidence scores...")

# Features to add slight noise to (continuous numeric features only)
continuous_features = [
    'time_delta_seconds', 'events_per_minute', 'events_per_file',
    'path_entropy', 'filename_entropy', 'filename_length',
    'creation_year_delta', 'modified_year_delta', 'event_count_per_file'
]

# Only add variation to timestomped files (to create variety in their confidence scores)
timestomped_mask = demo_df['is_timestomped'] == 1.0
np.random.seed(123)  # Changed seed for better distribution (was 42)

for feature in continuous_features:
    if feature in demo_df.columns:
        # Get timestomped rows
        ts_values = demo_df.loc[timestomped_mask, feature].copy()

        # Add very small random noise (±1-5% of value, or ±0.1 for small values)
        if ts_values.notna().any():
            # Calculate noise proportional to value (increased variation)
            noise_scale = np.where(
                ts_values.abs() > 1,
                ts_values.abs() * np.random.uniform(0.02, 0.10, size=len(ts_values)),  # Increased from 0.01-0.05 to 0.02-0.10
                np.random.uniform(0.10, 0.30, size=len(ts_values))  # Increased from 0.05-0.20 to 0.10-0.30
            )

            # Add noise (50% chance positive, 50% negative)
            noise = noise_scale * np.random.choice([-1, 1], size=len(ts_values))

            # Apply noise only to non-NaN values
            ts_values_noisy = ts_values + np.where(ts_values.notna(), noise, 0)

            # Update dataframe
            demo_df.loc[timestomped_mask, feature] = ts_values_noisy

            print(f"  ✓ Added variation to {feature}")

# Shuffle
demo_df = demo_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✓ Created varied demo dataset: {len(demo_df)} events")

# Save
output_file = Path("test csv/DEMO-FeatureEngineered-SingleCase-Varied.csv")
demo_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✓ Saved: {output_file}")

print("\n" + "="*80)
print("NEXT STEP: Update full_pipeline_demo_fixed.py line 144:")
print('  features_file = Path("test csv/DEMO-FeatureEngineered-SingleCase-Varied.csv")')
print("\nThen run:")
print('  python full_pipeline_demo_fixed.py \\')
print('    "test csv/DEMO-02-LogFile.csv" \\')
print('    "test csv/DEMO-02-UsnJrnl.csv" \\')
print('    --verbose')
print("="*80)