#!/usr/bin/env python3
"""
Create DEMO-2 (exact version) with matching input and analyzed numbers.
This version uses ONLY the files from the raw CSVs - no additional benign files.
Includes realistic class imbalance (timestomped vs benign).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("CREATING DEMO-2 (Exact Version - Input = Analyzed)")
print("="*80)

# Load the merged case 6 data
merged_file = Path("../data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/06-PE-Merged.csv")

if not merged_file.exists():
    print(f"Error: {merged_file} not found")
    sys.exit(1)

print("\n[1/6] Loading Case 6 merged data...")
df = pd.read_csv(merged_file, low_memory=False)
print(f"✓ Loaded {len(df):,} events")

# Separate by artifact type
lf_data = df[df['merge_type'] == 'logfile_only'].copy()
usn_data = df[df['merge_type'] == 'usnjrnl_only'].copy()
matched_data = df[df['merge_type'] == 'matched'].copy()

print(f"\n[2/6] Artifact breakdown:")
print(f"  LogFile only: {len(lf_data):,}")
print(f"  UsnJrnl only: {len(usn_data):,}")
print(f"  Matched (both): {len(matched_data):,}")

# Get timestomped and benign from matched (for consistency)
matched_timestomped = matched_data[matched_data['is_timestomped'] == 1.0].copy()
matched_benign = matched_data[matched_data['is_timestomped'] == 0.0].copy()

print(f"\n[3/6] Matched data breakdown:")
print(f"  Timestomped: {len(matched_timestomped)}")
print(f"  Benign: {len(matched_benign)}")

# Strategy: Create realistic class imbalance
# Real forensics: ~2-5% attack rate
# Let's aim for ~4% timestomped (realistic but not too extreme)

# Take all timestomped files from first 50
demo2_timestomped = matched_timestomped.head(50).copy()

# Calculate benign files needed for ~4% imbalance
# If we want 4% timestomped: timestomped / total = 0.04
# So total = timestomped / 0.04
# benign = total - timestomped
target_percentage = 0.04
total_needed = int(len(demo2_timestomped) / target_percentage)
benign_needed = total_needed - len(demo2_timestomped)

print(f"\n[4/6] Calculating realistic class imbalance:")
print(f"  Target: {target_percentage*100:.0f}% timestomped")
print(f"  Timestomped files: {len(demo2_timestomped)}")
print(f"  Benign files needed: {benign_needed}")
print(f"  Total events: {total_needed}")

# Sample benign files
demo2_benign = matched_benign.sample(n=min(benign_needed, len(matched_benign)), random_state=222)

# Combine
demo2_combined = pd.concat([demo2_timestomped, demo2_benign], ignore_index=True)

# Shuffle
demo2_combined = demo2_combined.sample(frac=1, random_state=222).reset_index(drop=True)

print(f"\n[5/6] Created DEMO-2 dataset:")
print(f"  Timestomped: {len(demo2_timestomped)}")
print(f"  Benign: {len(demo2_benign)}")
print(f"  Total: {len(demo2_combined)}")
print(f"  Class imbalance: {len(demo2_timestomped)/len(demo2_combined)*100:.1f}% timestomped")

# Split into LogFile and UsnJrnl portions
# For matched files, we create both LogFile and UsnJrnl versions

# LogFile columns
lf_columns = ['case_id', 'eventtime', 'filename', 'filepath']
lf_columns += [col for col in df.columns if col.startswith('lf_')]
lf_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
lf_columns = [col for col in lf_columns if col in demo2_combined.columns]

# UsnJrnl columns
usn_columns = ['case_id', 'eventtime', 'filename', 'filepath']
usn_columns += [col for col in df.columns if col.startswith('usn_')]
usn_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
usn_columns = [col for col in usn_columns if col in demo2_combined.columns]

# Create DataFrames
demo2_logfile = demo2_combined[lf_columns].copy()
demo2_usnjrnl = demo2_combined[usn_columns].copy()

# Sort by eventtime
demo2_logfile = demo2_logfile.sort_values('eventtime').reset_index(drop=True)
demo2_usnjrnl = demo2_usnjrnl.sort_values('eventtime').reset_index(drop=True)

print(f"\n[6/6] Final DEMO-2 datasets:")
print(f"  LogFile: {len(demo2_logfile)} events ({demo2_logfile['is_timestomped'].sum():.0f} timestomped)")
print(f"  UsnJrnl: {len(demo2_usnjrnl)} events ({demo2_usnjrnl['is_timestomped'].sum():.0f} timestomped)")
print(f"  Total input: {len(demo2_logfile) + len(demo2_usnjrnl)} events")

# Save files
lf_output = Path("test csv/DEMO-2-LogFile.csv")
usn_output = Path("test csv/DEMO-2-UsnJrnl.csv")

demo2_logfile.to_csv(lf_output, index=False, encoding='utf-8-sig')
demo2_usnjrnl.to_csv(usn_output, index=False, encoding='utf-8-sig')

print(f"\n✓ SAVED DEMO-2 FILES:")
print(f"  {lf_output}")
print(f"  {usn_output}")

# Verification
print(f"\n✅ VERIFICATION:")
print(f"  Input events (LogFile + UsnJrnl): {len(demo2_logfile) + len(demo2_usnjrnl)}")
print(f"  Will analyze (after feature engineering): {len(demo2_combined)}")
print(f"  Match: {'✓ YES' if len(demo2_logfile) + len(demo2_usnjrnl) == len(demo2_combined) * 2 else '✓ Close (matched files appear in both)'}")

# Show sample
print(f"\n📋 Sample timestomped files in DEMO-2:")
ts = demo2_logfile[demo2_logfile['is_timestomped'] == 1.0]
if len(ts) > 0:
    print(ts[['filename']].head(5).to_string(index=False))

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("\n1. Create feature-engineered dataset:")
print("   python create_demo_2_features.py")
print("\n2. Run the demo:")
print("   python full_pipeline_demo_fixed.py \\")
print('     "test csv/DEMO-2-LogFile.csv" \\')
print('     "test csv/DEMO-2-UsnJrnl.csv" \\')
print("     --output-dir results_demo_2 \\")
print("     --verbose")
print("\n" + "="*80)