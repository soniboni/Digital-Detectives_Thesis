#!/usr/bin/env python3
"""
Create separate demonstration LogFile and UsnJrnl CSV files with timestomping.
This extracts timestomped files from Case 6 and creates separate artifact files
for full_pipeline_demo.py testing.
"""

import pandas as pd
import sys
from pathlib import Path

# Load the merged case data
print("Loading merged case 6 data...")
merged_file = Path("../data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/06-PE-Merged.csv")

if not merged_file.exists():
    print(f"Error: {merged_file} not found")
    sys.exit(1)

df = pd.read_csv(merged_file, low_memory=False)
print(f"Loaded {len(df):,} events")

# Check merge type to understand data structure
if 'merge_type' in df.columns:
    print(f"\nMerge type breakdown:")
    print(df['merge_type'].value_counts())

# Get timestomped events
if 'is_timestomped' in df.columns:
    timestomped_count = df['is_timestomped'].sum()
    print(f"\nFound {int(timestomped_count)} timestomped events")

    # Separate by artifact type
    lf_only = df[df['merge_type'] == 'logfile_only']
    usn_only = df[df['merge_type'] == 'usnjrnl_only']
    both = df[df['merge_type'] == 'matched']

    print(f"\nArtifact breakdown:")
    print(f"  LogFile only: {len(lf_only)}")
    print(f"  UsnJrnl only: {len(usn_only)}")
    print(f"  Both artifacts: {len(both)}")

    # Get timestomped files
    timestomped = df[df['is_timestomped'] == 1.0]
    benign = df[df['is_timestomped'] == 0.0]

    print(f"\nTimestomped by artifact:")
    print(f"  LogFile only: {len(timestomped[timestomped['merge_type'] == 'logfile_only'])}")
    print(f"  UsnJrnl only: {len(timestomped[timestomped['merge_type'] == 'usnjrnl_only'])}")
    print(f"  Both artifacts: {len(timestomped[timestomped['merge_type'] == 'matched'])}")

    # Sample: Get timestomped + benign for each artifact
    # For LogFile: include lf_only + both
    # For UsnJrnl: include usn_only + both

    # Sample sizes
    n_timestomped = min(50, len(timestomped))
    n_benign = 1000

    # Get timestomped sample
    demo_timestomped = timestomped.head(n_timestomped)
    demo_benign = benign.sample(n=min(n_benign, len(benign)), random_state=42)

    # Combine
    demo_df = pd.concat([demo_timestomped, demo_benign])

    # Create LogFile CSV (logfile_only + matched)
    lf_data = demo_df[demo_df['merge_type'].isin(['logfile_only', 'matched'])].copy()

    # Get LogFile columns (lf_* prefix)
    lf_columns = ['case_id', 'eventtime', 'filename', 'filepath']
    lf_columns += [col for col in df.columns if col.startswith('lf_')]
    lf_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']

    # Keep only columns that exist
    lf_columns = [col for col in lf_columns if col in lf_data.columns]
    lf_df = lf_data[lf_columns].copy()

    # Create UsnJrnl CSV (usnjrnl_only + matched)
    usn_data = demo_df[demo_df['merge_type'].isin(['usnjrnl_only', 'matched'])].copy()

    # Get UsnJrnl columns (usn_* prefix)
    usn_columns = ['case_id', 'eventtime', 'filename', 'filepath']
    usn_columns += [col for col in df.columns if col.startswith('usn_')]
    usn_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']

    # Keep only columns that exist
    usn_columns = [col for col in usn_columns if col in usn_data.columns]
    usn_df = usn_data[usn_columns].copy()

    print(f"\n✓ Created demo datasets:")
    print(f"  LogFile events: {len(lf_df)}")
    print(f"    - Timestomped: {int(lf_df['is_timestomped'].sum())}")
    print(f"  UsnJrnl events: {len(usn_df)}")
    print(f"    - Timestomped: {int(usn_df['is_timestomped'].sum())}")

    # Save to test csv folder
    lf_output = Path("test csv/DEMO-Case6-LogFile.csv")
    usn_output = Path("test csv/DEMO-Case6-UsnJrnl.csv")

    lf_df.to_csv(lf_output, index=False, encoding='utf-8-sig')
    usn_df.to_csv(usn_output, index=False, encoding='utf-8-sig')

    print(f"\n✓ Saved demo files:")
    print(f"  {lf_output}")
    print(f"  {usn_output}")

    # Show sample of timestomped files
    print("\nSample timestomped files in demo (LogFile):")
    lf_timestomped = lf_df[lf_df['is_timestomped'] == 1.0]
    if len(lf_timestomped) > 0:
        print(lf_timestomped[['filename', 'filepath', 'is_timestomped']].head(5))
    else:
        print("  (No timestomped files in LogFile artifact)")

    print("\nSample timestomped files in demo (UsnJrnl):")
    usn_timestomped = usn_df[usn_df['is_timestomped'] == 1.0]
    if len(usn_timestomped) > 0:
        print(usn_timestomped[['filename', 'filepath', 'is_timestomped']].head(5))
    else:
        print("  (No timestomped files in UsnJrnl artifact)")

else:
    print("No 'is_timestomped' column found in merged data")
    print(f"Columns available: {list(df.columns)}")