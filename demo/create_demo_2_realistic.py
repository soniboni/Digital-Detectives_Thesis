#!/usr/bin/env python3
"""
Create DEMO-2 with REALISTIC LogFile and UsnJrnl distributions.
This version creates authentic artifact distributions where:
- Some files only appear in LogFile (system-level operations)
- Some files only appear in UsnJrnl (user-level changes)
- Some files appear in BOTH (high-activity files, especially timestomped ones)

This addresses the "too perfect" problem of having identical event counts.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

print("="*70)
print("CREATING REALISTIC DEMO-2 WITH VARIED ARTIFACT DISTRIBUTIONS")
print("="*70)

# Load the merged case data
print("\n[1/7] Loading merged case 6 data...")
merged_file = Path("../data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging/06-PE-Merged.csv")

if not merged_file.exists():
    print(f"Error: {merged_file} not found")
    sys.exit(1)

df = pd.read_csv(merged_file, low_memory=False)
print(f"  Loaded {len(df):,} total events")

# Check merge type distribution
if 'merge_type' in df.columns:
    print(f"\n[2/7] Original merge type breakdown:")
    merge_counts = df['merge_type'].value_counts()
    for merge_type, count in merge_counts.items():
        pct = count / len(df) * 100
        print(f"  {merge_type:15s}: {count:6,} ({pct:5.1f}%)")

# Get timestomped and benign files
timestomped = df[df['is_timestomped'] == 1.0].copy()
benign = df[df['is_timestomped'] == 0.0].copy()

print(f"\n[3/7] Selecting files for DEMO-2:")
print(f"  Available timestomped: {len(timestomped):,}")
print(f"  Available benign: {len(benign):,}")

# Strategy for realistic demo:
# - Use indices 10-59 for timestomped (50 files, different from DEMO-02's 0-49)
# - Target ~6-7% timestomped for realistic compromised system
# - Create varied artifact distribution

np.random.seed(333)  # Different seed from other demos

# Select 50 timestomped files (indices 10-59)
demo2_timestomped = timestomped.iloc[10:60].copy()

# Calculate benign needed for ~6.5% timestomped rate
target_percentage = 0.065
total_unique_needed = int(len(demo2_timestomped) / target_percentage)
benign_needed = total_unique_needed - len(demo2_timestomped)

print(f"\n[4/7] Target composition:")
print(f"  Timestomped files: {len(demo2_timestomped)}")
print(f"  Benign files needed: {benign_needed}")
print(f"  Total unique files: {total_unique_needed}")
print(f"  Target timestomped %: {target_percentage*100:.1f}%")

# Select benign files with realistic merge_type distribution
# Real NTFS: More UsnJrnl events than LogFile
demo2_benign = benign.sample(n=min(benign_needed, len(benign)), random_state=333)

# Combine all selected files
demo2_combined = pd.concat([demo2_timestomped, demo2_benign]).reset_index(drop=True)

actual_percentage = len(demo2_timestomped) / len(demo2_combined) * 100
print(f"\n[5/7] Created DEMO-2 master dataset:")
print(f"  Timestomped: {len(demo2_timestomped)}")
print(f"  Benign: {len(demo2_benign)}")
print(f"  Total unique: {len(demo2_combined)}")
print(f"  Actual timestomped: {actual_percentage:.2f}%")

# NOW CREATE REALISTIC ARTIFACT SPLITS
print(f"\n[6/7] Creating realistic artifact distributions...")

# Analyze merge types in our selection
print(f"\n  Merge type breakdown in DEMO-2:")
merge_breakdown = demo2_combined['merge_type'].value_counts()
for merge_type, count in merge_breakdown.items():
    pct = count / len(demo2_combined) * 100
    print(f"    {merge_type:15s}: {count:4} ({pct:5.1f}%)")

# Create LogFile CSV: include 'logfile_only' + 'matched' entries
lf_data = demo2_combined[demo2_combined['merge_type'].isin(['logfile_only', 'matched'])].copy()

# Create UsnJrnl CSV: include 'usnjrnl_only' + 'matched' entries
usn_data = demo2_combined[demo2_combined['merge_type'].isin(['usnjrnl_only', 'matched'])].copy()

print(f"\n  Artifact file breakdown:")
print(f"    LogFile events: {len(lf_data)} (logfile_only + matched)")
print(f"    UsnJrnl events: {len(usn_data)} (usnjrnl_only + matched)")
print(f"    Ratio: 1:{len(usn_data)/len(lf_data):.2f} (UsnJrnl typically has MORE)")

# Extract LogFile columns
lf_columns = ['case_id', 'eventtime', 'filename', 'filepath']
lf_columns += [col for col in df.columns if col.startswith('lf_')]
lf_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
lf_columns = [col for col in lf_columns if col in lf_data.columns]

demo2_logfile = lf_data[lf_columns].copy()

# Extract UsnJrnl columns
usn_columns = ['case_id', 'eventtime', 'filename', 'filepath']
usn_columns += [col for col in df.columns if col.startswith('usn_')]
usn_columns += ['is_timestomped', 'timestomp_tool_executed', 'suspicious_tool_name', 'label_source']
usn_columns = [col for col in usn_columns if col in usn_data.columns]

demo2_usnjrnl = usn_data[usn_columns].copy()

# Sort by eventtime for realistic timeline
demo2_logfile = demo2_logfile.sort_values('eventtime').reset_index(drop=True)
demo2_usnjrnl = demo2_usnjrnl.sort_values('eventtime').reset_index(drop=True)

# Save to CSV
lf_output = Path("test csv/DEMO-2-LogFile.csv")
usn_output = Path("test csv/DEMO-2-UsnJrnl.csv")

demo2_logfile.to_csv(lf_output, index=False, encoding='utf-8-sig')
demo2_usnjrnl.to_csv(usn_output, index=False, encoding='utf-8-sig')

print(f"\n[7/7] ✓ Saved DEMO-2 artifact files:")
print(f"  {lf_output}")
print(f"    - Total events: {len(demo2_logfile)}")
print(f"    - Timestomped: {int(demo2_logfile['is_timestomped'].sum())}")
print(f"  {usn_output}")
print(f"    - Total events: {len(demo2_usnjrnl)}")
print(f"    - Timestomped: {int(demo2_usnjrnl['is_timestomped'].sum())}")

print(f"\n" + "="*70)
print("REALISTIC INPUT SUMMARY FOR DEMO-2:")
print("="*70)
print(f"LogFile events:  {len(demo2_logfile):4} events")
print(f"UsnJrnl events:  {len(demo2_usnjrnl):4} events")
print(f"Total input:     {len(demo2_logfile) + len(demo2_usnjrnl):4} events")
print(f"")
print(f"Unique files that will be analyzed: {len(demo2_combined)}")
print(f"  (Deduplication merges 'matched' files that appear in both)")
print("="*70)

print("\n✓ DEMO-2 created with realistic artifact distributions!")
print("  Next step: Run create_demo_2_features.py to generate feature-engineered dataset")