#!/usr/bin/env python3
"""
Create DEMO-03 LogFile and UsnJrnl CSV files with different timestomped files.
This creates a backup demo set for thesis defense panel questions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("CREATING DEMO-03 FILES (Alternative Demo Set)")
print("="*80)

# Load the merged case 6 data
merged_file = Path("../data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/06-PE-Merged.csv")

if not merged_file.exists():
    print(f"Error: {merged_file} not found")
    sys.exit(1)

print("\n[1/5] Loading Case 6 merged data...")
df = pd.read_csv(merged_file, low_memory=False)
print(f"✓ Loaded {len(df):,} events")

# Separate by artifact type
lf_data = df[df['merge_type'] == 'logfile_only'].copy()
usn_data = df[df['merge_type'] == 'usnjrnl_only'].copy()
matched_data = df[df['merge_type'] == 'matched'].copy()

print(f"\n[2/5] Artifact breakdown:")
print(f"  LogFile only: {len(lf_data):,}")
print(f"  UsnJrnl only: {len(usn_data):,}")
print(f"  Matched (both): {len(matched_data):,}")

# Get timestomped files
lf_timestomped = lf_data[lf_data['is_timestomped'] == 1.0].copy()
usn_timestomped = usn_data[usn_data['is_timestomped'] == 1.0].copy()
matched_timestomped = matched_data[matched_data['is_timestomped'] == 1.0].copy()

print(f"\n[3/5] Timestomped files:")
print(f"  LogFile: {len(lf_timestomped)}")
print(f"  UsnJrnl: {len(usn_timestomped)}")
print(f"  Matched: {len(matched_timestomped)}")

# Strategy: Sample DIFFERENT timestomped files than DEMO-02
# DEMO-02 used the first 50 timestomped files, DEMO-03 will use a different subset

# For DEMO-03, take timestomped files from indices 20-70 (different from DEMO-02's 0-50)
# Also include more matched files for variety

# Sample timestomped files (skip first 20 to avoid overlap with DEMO-02)
if len(matched_timestomped) > 20:
    demo03_timestomped = matched_timestomped.iloc[20:].copy()
else:
    demo03_timestomped = matched_timestomped.copy()

# Take a subset for demo (around 40-50 files)
demo03_sample_size = min(40, len(demo03_timestomped))
demo03_timestomped_sample = demo03_timestomped.sample(n=demo03_sample_size, random_state=789)

print(f"\n[4/5] Creating DEMO-03 sample:")
print(f"  Timestomped files: {len(demo03_timestomped_sample)}")

# Add benign files (different sample than DEMO-02)
lf_benign = lf_data[lf_data['is_timestomped'] == 0.0].copy()
usn_benign = usn_data[usn_data['is_timestomped'] == 0.0].copy()

# Sample different benign files using different random seed
lf_benign_sample = lf_benign.sample(n=min(150, len(lf_benign)), random_state=789)
usn_benign_sample = usn_benign.sample(n=min(800, len(usn_benign)), random_state=789)

print(f"  Benign LogFile: {len(lf_benign_sample)}")
print(f"  Benign UsnJrnl: {len(usn_benign_sample)}")

# Split timestomped files into LogFile and UsnJrnl portions
# For matched files, we need to create both LogFile and UsnJrnl versions
demo03_lf = demo03_timestomped_sample.copy()
demo03_usn = demo03_timestomped_sample.copy()

# Add benign files
demo03_lf = pd.concat([demo03_lf, lf_benign_sample], ignore_index=True)
demo03_usn = pd.concat([demo03_usn, usn_benign_sample], ignore_index=True)

# Get LogFile columns
lf_columns = ['case_id', 'eventtime', 'filename', 'filepath']
lf_columns += [col for col in df.columns if col.startswith('lf_')]
lf_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
lf_columns = [col for col in lf_columns if col in demo03_lf.columns]

# Get UsnJrnl columns
usn_columns = ['case_id', 'eventtime', 'filename', 'filepath']
usn_columns += [col for col in df.columns if col.startswith('usn_')]
usn_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
usn_columns = [col for col in usn_columns if col in demo03_usn.columns]

# Create final DataFrames
demo03_logfile = demo03_lf[lf_columns].copy()
demo03_usnjrnl = demo03_usn[usn_columns].copy()

# Sort by eventtime
demo03_logfile = demo03_logfile.sort_values('eventtime').reset_index(drop=True)
demo03_usnjrnl = demo03_usnjrnl.sort_values('eventtime').reset_index(drop=True)

print(f"\n[5/5] Final DEMO-03 datasets:")
print(f"  LogFile events: {len(demo03_logfile)} ({demo03_logfile['is_timestomped'].sum():.0f} timestomped)")
print(f"  UsnJrnl events: {len(demo03_usnjrnl)} ({demo03_usnjrnl['is_timestomped'].sum():.0f} timestomped)")

# Save files
lf_output = Path("test csv/DEMO-03-LogFile.csv")
usn_output = Path("test csv/DEMO-03-UsnJrnl.csv")

demo03_logfile.to_csv(lf_output, index=False, encoding='utf-8-sig')
demo03_usnjrnl.to_csv(usn_output, index=False, encoding='utf-8-sig')

print(f"\n✓ SAVED DEMO-03 FILES:")
print(f"  {lf_output}")
print(f"  {usn_output}")

# Show sample of timestomped files
print(f"\n📋 Sample timestomped files in DEMO-03:")
lf_ts = demo03_logfile[demo03_logfile['is_timestomped'] == 1.0]
if len(lf_ts) > 0:
    print("\nLogFile timestomped samples:")
    print(lf_ts[['filename']].head(5).to_string(index=False))

usn_ts = demo03_usnjrnl[demo03_usnjrnl['is_timestomped'] == 1.0]
if len(usn_ts) > 0:
    print("\nUsnJrnl timestomped samples:")
    print(usn_ts[['filename']].head(5).to_string(index=False))

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("\n1. Create varied feature-engineered dataset for DEMO-03:")
print("   python create_demo_03_features.py")
print("\n2. Run the demo:")
print("   python full_pipeline_demo_fixed.py \\")
print('     "test csv/DEMO-03-LogFile.csv" \\')
print('     "test csv/DEMO-03-UsnJrnl.csv" \\')
print("     --output-dir results_demo_03 \\")
print("     --verbose")
print("\n" + "="*80)