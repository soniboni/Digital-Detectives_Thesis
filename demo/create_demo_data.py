#!/usr/bin/env python3
"""
Create demonstration CSV files with confirmed timestomping for testing.
This script extracts a subset from the merged data that contains timestomped files.
"""

import pandas as pd
import sys
from pathlib import Path

# Load the merged case data (which has both artifacts merged)
print("Loading merged case 6 data...")
merged_file = Path("../data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/06-PE-Merged.csv")

if not merged_file.exists():
    print(f"Error: {merged_file} not found")
    sys.exit(1)

df = pd.read_csv(merged_file, low_memory=False)
print(f"Loaded {len(df):,} events")

# Check for timestomping
if 'is_timestomped' in df.columns:
    timestomped_count = df['is_timestomped'].sum()
    print(f"Found {int(timestomped_count)} timestomped events")

    # Get sample of timestomped and benign files
    timestomped = df[df['is_timestomped'] == 1.0]
    benign = df[df['is_timestomped'] == 0.0]

    # Sample: 50 timestomped + 1000 benign for demo
    sample_size = min(50, len(timestomped))
    demo_timestomped = timestomped.head(sample_size)
    demo_benign = benign.sample(n=min(1000, len(benign)), random_state=42)

    demo_df = pd.concat([demo_timestomped, demo_benign]).sort_values('eventtime')

    print(f"\nCreating demo dataset:")
    print(f"  Timestomped: {len(demo_timestomped)}")
    print(f"  Benign: {len(demo_benign)}")
    print(f"  Total: {len(demo_df)}")

    # Save to test csv folder
    output_file = Path("test csv/DEMO-Case6-Merged.csv")
    demo_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved demo file: {output_file}")

    # Show sample of timestomped files
    print("\nSample timestomped files in demo:")
    print(demo_timestomped[['filename', 'filepath', 'is_timestomped']].head(10))

else:
    print("No 'is_timestomped' column found in merged data")
    print(f"Columns available: {list(df.columns)}")