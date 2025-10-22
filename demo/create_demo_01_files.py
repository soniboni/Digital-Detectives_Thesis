#!/usr/bin/env python3
"""
Create DEMO-01 LogFile and UsnJrnl CSV files with NO timestomped files.
This demonstrates what the system looks like when NO timestomping is detected.
"""

import pandas as pd
from pathlib import Path
import sys

print("="*80)
print("CREATING DEMO-01 FILES (Clean System - No Timestomping)")
print("="*80)

# Load the merged case 6 data
merged_file = Path("../data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/06-PE-Merged.csv")

if not merged_file.exists():
    print(f"Error: {merged_file} not found")
    sys.exit(1)

print("\n[1/4] Loading Case 6 merged data...")
df = pd.read_csv(merged_file, low_memory=False)
print(f"✓ Loaded {len(df):,} events")

# Separate by artifact type
lf_data = df[df['merge_type'] == 'logfile_only'].copy()
usn_data = df[df['merge_type'] == 'usnjrnl_only'].copy()

print(f"\n[2/4] Artifact breakdown:")
print(f"  LogFile only: {len(lf_data):,}")
print(f"  UsnJrnl only: {len(usn_data):,}")

# Get ONLY benign files (NO timestomped files!)
lf_benign = lf_data[lf_data['is_timestomped'] == 0.0].copy()
usn_benign = usn_data[usn_data['is_timestomped'] == 0.0].copy()

print(f"\n[3/4] Filtering for benign files only...")
print(f"  Benign LogFile: {len(lf_benign):,}")
print(f"  Benign UsnJrnl: {len(usn_benign):,}")

# Sample benign files (different from DEMO-02 and DEMO-03)
# Use a small sample to keep demo fast
lf_sample = lf_benign.sample(n=min(300, len(lf_benign)), random_state=111)
usn_sample = usn_benign.sample(n=min(1200, len(usn_benign)), random_state=111)

print(f"\n  Sampled for DEMO-01:")
print(f"    LogFile: {len(lf_sample)}")
print(f"    UsnJrnl: {len(usn_sample)}")

# Get LogFile columns
lf_columns = ['case_id', 'eventtime', 'filename', 'filepath']
lf_columns += [col for col in df.columns if col.startswith('lf_')]
lf_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
lf_columns = [col for col in lf_columns if col in lf_sample.columns]

# Get UsnJrnl columns
usn_columns = ['case_id', 'eventtime', 'filename', 'filepath']
usn_columns += [col for col in df.columns if col.startswith('usn_')]
usn_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
usn_columns = [col for col in usn_columns if col in usn_sample.columns]

# Create final DataFrames
demo01_logfile = lf_sample[lf_columns].copy()
demo01_usnjrnl = usn_sample[usn_columns].copy()

# Sort by eventtime
demo01_logfile = demo01_logfile.sort_values('eventtime').reset_index(drop=True)
demo01_usnjrnl = demo01_usnjrnl.sort_values('eventtime').reset_index(drop=True)

# Verify NO timestomped files
lf_ts_count = demo01_logfile['is_timestomped'].sum()
usn_ts_count = demo01_usnjrnl['is_timestomped'].sum()

print(f"\n[4/4] Final DEMO-01 datasets (CLEAN SYSTEM):")
print(f"  LogFile events: {len(demo01_logfile)} (timestomped: {int(lf_ts_count)})")
print(f"  UsnJrnl events: {len(demo01_usnjrnl)} (timestomped: {int(usn_ts_count)})")

if lf_ts_count > 0 or usn_ts_count > 0:
    print(f"\n⚠️  WARNING: Found timestomped files! This should be a clean dataset.")
else:
    print(f"\n✓ VERIFIED: Zero timestomped files (clean system)")

# Save files
lf_output = Path("test csv/DEMO-01-LogFile.csv")
usn_output = Path("test csv/DEMO-01-UsnJrnl.csv")

demo01_logfile.to_csv(lf_output, index=False, encoding='utf-8-sig')
demo01_usnjrnl.to_csv(usn_output, index=False, encoding='utf-8-sig')

print(f"\n✓ SAVED DEMO-01 FILES:")
print(f"  {lf_output}")
print(f"  {usn_output}")

# Show sample of benign files
print(f"\n📋 Sample benign files in DEMO-01:")
print(demo01_logfile[['filename']].head(5).to_string(index=False))

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("\n1. Create feature-engineered dataset for DEMO-01:")
print("   python create_demo_01_features.py")
print("\n2. Run the demo (should show 0 detections):")
print("   python full_pipeline_demo_fixed.py \\")
print('     "test csv/DEMO-01-LogFile.csv" \\')
print('     "test csv/DEMO-01-UsnJrnl.csv" \\')
print("     --output-dir results_demo_01 \\")
print("     --verbose")
print("\n" + "="*80)