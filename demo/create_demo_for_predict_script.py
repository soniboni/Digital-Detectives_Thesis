#!/usr/bin/env python3
"""
Create demonstration data from engineered features (not raw CSVs).
This will work with predict_timestomping.py and show actual detections.
"""

import pandas as pd
import sys
from pathlib import Path

# Load the engineered features (which includes model-ready data with labels)
print("Loading engineered features...")
features_file = Path("../data/processed/Phase 2 - Feature Engineering/features_engineered.csv")

if not features_file.exists():
    print(f"Error: {features_file} not found")
    sys.exit(1)

df = pd.read_csv(features_file, low_memory=False)
print(f"Loaded {len(df):,} events")

# Check for timestomping
if 'is_timestomped' in df.columns:
    timestomped_count = df['is_timestomped'].sum()
    print(f"Found {int(timestomped_count)} timestomped events")

    # Get timestomped and benign files
    timestomped = df[df['is_timestomped'] == 1.0]
    benign = df[df['is_timestomped'] == 0.0]

    # Sample: 100 timestomped + 2000 benign for demo
    sample_size = min(100, len(timestomped))
    demo_timestomped = timestomped.head(sample_size)
    demo_benign = benign.sample(n=min(2000, len(benign)), random_state=42)

    demo_df = pd.concat([demo_timestomped, demo_benign])

    print(f"\nCreating demo dataset:")
    print(f"  Timestomped: {len(demo_timestomped)}")
    print(f"  Benign: {len(demo_benign)}")
    print(f"  Total: {len(demo_df)}")

    # Save to test csv folder
    output_file = Path("test csv/DEMO-FeatureEngineered.csv")
    demo_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved demo file: {output_file}")

    # Show sample of timestomped files
    print("\nSample timestomped files in demo:")
    print(demo_timestomped[['filename', 'filepath', 'is_timestomped']].head(10).to_string())

    print("\n\nTo run the demo with this file, use:")
    print(f'  python predict_timestomping.py "{output_file}"')

else:
    print("No 'is_timestomped' column found in engineered features")
    print(f"Columns available: {list(df.columns)}")