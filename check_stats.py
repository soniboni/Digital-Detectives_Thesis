import pandas as pd
from pathlib import Path

# Path to merged files
MERGED_DIR = Path("data/processed/Phase 1 - Data Collection & Preprocessing/B. Data Case Merging")

print("="*80)
print("CHECKING SUSPICIOUS COUNTS IN MERGED FILES")
print("="*80)

total_susp_lf = 0
total_susp_usn = 0
total_susp_merged = 0
total_susp_matched = 0
total_susp_lf_only = 0
total_susp_usn_only = 0

for i in range(1, 13):
    case_id = f"{i:02d}-PE"
    filepath = MERGED_DIR / f"{case_id}-Merged.csv"

    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Count suspicious flags
    susp_lf = df['is_suspicious_lf'].fillna(0).sum()
    susp_usn = df['is_suspicious_usn'].fillna(0).sum()
    susp_merged = df['is_suspicious'].fillna(0).sum()

    # Count by merge type
    susp_df = df[df['is_suspicious'] == 1]
    matched = len(susp_df[susp_df['merge_type'] == 'matched'])
    lf_only = len(susp_df[susp_df['merge_type'] == 'logfile_only'])
    usn_only = len(susp_df[susp_df['merge_type'] == 'usnjrnl_only'])

    total_susp_lf += susp_lf
    total_susp_usn += susp_usn
    total_susp_merged += susp_merged
    total_susp_matched += matched
    total_susp_lf_only += lf_only
    total_susp_usn_only += usn_only

    print(f"\n{case_id}:")
    print(f"  is_suspicious_lf sum:    {int(susp_lf)}")
    print(f"  is_suspicious_usn sum:   {int(susp_usn)}")
    print(f"  is_suspicious sum:       {int(susp_merged)}")
    print(f"  Breakdown: Matched={matched}, LF-only={lf_only}, USN-only={usn_only}")

print("\n" + "="*80)
print("TOTALS:")
print("="*80)
print(f"Total is_suspicious_lf flags:    {int(total_susp_lf)}")
print(f"Total is_suspicious_usn flags:   {int(total_susp_usn)}")
print(f"Total is_suspicious events:      {int(total_susp_merged)}")
print(f"\nBreakdown:")
print(f"  Matched (both):    {total_susp_matched}")
print(f"  LF-only:           {total_susp_lf_only}")
print(f"  USN-only:          {total_susp_usn_only}")
print(f"\nCalculations:")
print(f"  Sum of LF + USN flags:           {int(total_susp_lf + total_susp_usn)}")
print(f"  Verification (LF + USN + Both×2): {total_susp_lf_only} + {total_susp_usn_only} + {total_susp_matched}×2 = {total_susp_lf_only + total_susp_usn_only + total_susp_matched*2}")
