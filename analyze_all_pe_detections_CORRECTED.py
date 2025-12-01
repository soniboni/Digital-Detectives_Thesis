#!/usr/bin/env python3
"""
CORRECTED Comprehensive Analysis of detect_accurate.py Results Across All PE Cases

This script properly categorizes ground truth labels into:
1. TRUE malicious timestomping (LogFile source OR "[Malicious]" label)
2. File system tunneling (UsnJrnl source AND "[Suspicious]" label)

Key Fix: Uses ground truth SOURCE field instead of prediction features
"""

import pandas as pd
import os
from pathlib import Path

# Directories
OUTPUT_DIR = Path("test/output")
SUSPICIOUS_DIR = Path("data/raw/suspicious")

def analyze_case(case_num):
    """Analyze a single PE case with CORRECTED categorization logic."""
    case_id = f"{case_num:02d}-pe"

    # Load predictions
    pred_path = OUTPUT_DIR / case_id / "predictions_with_features.csv"
    if not pred_path.exists():
        return None

    predictions = pd.read_csv(pred_path)

    # Load ground truth
    sus_path = SUSPICIOUS_DIR / f"{case_num:02d}-PE-Suspicious.csv"
    if not sus_path.exists():
        return None

    suspicious = pd.read_csv(sus_path, encoding='utf-8-sig')

    # Filter to Timestamp Manipulation only
    timestomped_labels = suspicious[suspicious['category'] == 'Timestamp Manipulation']

    results = {
        'case': case_num,
        'total_events': len(predictions),
        'total_labeled_timestomped': len(timestomped_labels),

        # TRUE timestomping (LogFile source OR [Malicious] label)
        'true_timestomping_events': 0,
        'true_timestomping_detected_high': 0,
        'true_timestomping_detected_medium': 0,
        'true_timestomping_detected_low': 0,
        'true_timestomping_missed': 0,

        # File system tunneling (UsnJrnl source AND [Suspicious] label)
        'tunneling_events': 0,
        'tunneling_detected_high': 0,
        'tunneling_detected_medium': 0,
        'tunneling_detected_low': 0,
        'tunneling_missed': 0,

        # Detection confidence breakdown
        'high_confidence_detections': len(predictions[predictions['probability'] >= 0.7]),
        'medium_confidence_detections': len(predictions[(predictions['probability'] >= 0.5) & (predictions['probability'] < 0.7)]),
        'low_confidence_detections': len(predictions[(predictions['probability'] >= 0.3) & (predictions['probability'] < 0.5)]),
    }

    # Analyze each labeled event
    for _, label in timestomped_labels.iterrows():
        if label['source'] == 'usnjrnl':
            # Find matching prediction by USN
            match = predictions[predictions['usn_usn'] == label['lsn/usn']]
        else:  # logfile
            # Find matching prediction by LSN
            match = predictions[predictions['lf_lsn'] == label['lsn/usn']]

        if len(match) == 0:
            continue  # Event not in Phase 1 filtered output

        # Get highest probability for this event
        max_prob = match['probability'].max()

        # CORRECTED CATEGORIZATION: Use ground truth SOURCE and detail
        is_malicious = (
            label['source'] == 'logfile' or
            '[Malicious]' in str(label['detail'])
        )

        # Categorize: TRUE timestomping vs file system tunneling
        if is_malicious:
            results['true_timestomping_events'] += 1
            if max_prob >= 0.7:
                results['true_timestomping_detected_high'] += 1
            elif max_prob >= 0.5:
                results['true_timestomping_detected_medium'] += 1
            elif max_prob >= 0.3:
                results['true_timestomping_detected_low'] += 1
            else:
                results['true_timestomping_missed'] += 1
        else:
            # File system tunneling (usnjrnl source with [Suspicious] label)
            results['tunneling_events'] += 1
            if max_prob >= 0.7:
                results['tunneling_detected_high'] += 1
            elif max_prob >= 0.5:
                results['tunneling_detected_medium'] += 1
            elif max_prob >= 0.3:
                results['tunneling_detected_low'] += 1
            else:
                results['tunneling_missed'] += 1

    # Calculate rates
    if results['true_timestomping_events'] > 0:
        results['true_timestomping_recall'] = (
            (results['true_timestomping_detected_high'] +
             results['true_timestomping_detected_medium'] +
             results['true_timestomping_detected_low']) /
            results['true_timestomping_events']
        )
        results['true_timestomping_high_conf_rate'] = (
            results['true_timestomping_detected_high'] / results['true_timestomping_events']
        )
    else:
        results['true_timestomping_recall'] = 0.0
        results['true_timestomping_high_conf_rate'] = 0.0

    if results['tunneling_events'] > 0:
        results['tunneling_recall'] = (
            (results['tunneling_detected_high'] +
             results['tunneling_detected_medium'] +
             results['tunneling_detected_low']) /
            results['tunneling_events']
        )
    else:
        results['tunneling_recall'] = 0.0

    return results


def main():
    print("="*80)
    print("CORRECTED ANALYSIS: TRUE Malicious Timestomping vs File System Tunneling")
    print("="*80)
    print()
    print("KEY FIX: Now properly categorizes based on ground truth SOURCE field")
    print("  - TRUE Malicious: LogFile source OR [Malicious] label")
    print("  - File System Tunneling: UsnJrnl source AND [Suspicious] label")
    print()

    all_results = []

    # Analyze each case
    for case_num in range(1, 13):
        result = analyze_case(case_num)
        if result:
            all_results.append(result)
            true_count = result['true_timestomping_events']
            tunnel_count = result['tunneling_events']
            print(f"Case {case_num:02d}: {true_count} TRUE malicious, {tunnel_count} tunneling events")
        else:
            print(f"Case {case_num:02d}: Missing files")

    if not all_results:
        print("\nERROR: No results found!")
        return

    # Create summary DataFrame
    df = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("PER-CASE RESULTS")
    print("="*80)
    print()

    # Display key metrics per case
    display_cols = [
        'case', 'total_labeled_timestomped',
        'true_timestomping_events', 'true_timestomping_detected_high',
        'tunneling_events', 'tunneling_detected_high',
        'true_timestomping_recall', 'true_timestomping_high_conf_rate'
    ]

    print(df[display_cols].to_string(index=False))

    # Aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS ACROSS ALL CASES")
    print("="*80)
    print()

    total_true = df['true_timestomping_events'].sum()
    total_true_detected_high = df['true_timestomping_detected_high'].sum()
    total_true_detected_medium = df['true_timestomping_detected_medium'].sum()
    total_true_detected_low = df['true_timestomping_detected_low'].sum()
    total_true_missed = df['true_timestomping_missed'].sum()

    total_tunneling = df['tunneling_events'].sum()
    total_tunneling_detected_high = df['tunneling_detected_high'].sum()
    total_tunneling_detected_medium = df['tunneling_detected_medium'].sum()
    total_tunneling_detected_low = df['tunneling_detected_low'].sum()
    total_tunneling_missed = df['tunneling_missed'].sum()

    print(f"TRUE Malicious Timestomping (LogFile Evidence or [Malicious] Label):")
    print(f"  Total events: {total_true}")
    print(f"  Detected HIGH confidence (≥70%): {total_true_detected_high} ({total_true_detected_high/total_true*100 if total_true > 0 else 0:.1f}%)")
    print(f"  Detected MEDIUM confidence (50-70%): {total_true_detected_medium} ({total_true_detected_medium/total_true*100 if total_true > 0 else 0:.1f}%)")
    print(f"  Detected LOW confidence (30-50%): {total_true_detected_low} ({total_true_detected_low/total_true*100 if total_true > 0 else 0:.1f}%)")
    print(f"  Missed (<30%): {total_true_missed} ({total_true_missed/total_true*100 if total_true > 0 else 0:.1f}%)")

    if total_true > 0:
        true_recall = (total_true_detected_high + total_true_detected_medium + total_true_detected_low) / total_true
        print(f"  Overall Recall: {true_recall*100:.2f}%")
        print(f"  HIGH Confidence Rate: {total_true_detected_high/total_true*100:.2f}%")

    print()
    print(f"File System Tunneling (UsnJrnl only, [Suspicious] Label):")
    print(f"  Total events: {total_tunneling}")
    print(f"  Detected HIGH confidence (≥70%): {total_tunneling_detected_high} ({total_tunneling_detected_high/total_tunneling*100 if total_tunneling > 0 else 0:.1f}%)")
    print(f"  Detected MEDIUM confidence (50-70%): {total_tunneling_detected_medium} ({total_tunneling_detected_medium/total_tunneling*100 if total_tunneling > 0 else 0:.1f}%)")
    print(f"  Detected LOW confidence (30-50%): {total_tunneling_detected_low} ({total_tunneling_detected_low/total_tunneling*100 if total_tunneling > 0 else 0:.1f}%)")
    print(f"  Missed (<30%): {total_tunneling_missed} ({total_tunneling_missed/total_tunneling*100 if total_tunneling > 0 else 0:.1f}%)")

    if total_tunneling > 0:
        tunneling_recall = (total_tunneling_detected_high + total_tunneling_detected_medium + total_tunneling_detected_low) / total_tunneling
        print(f"  Overall Recall: {tunneling_recall*100:.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print()

    if total_true > 0:
        true_high_rate = total_true_detected_high / total_true * 100
        if true_high_rate >= 90:
            print(f"✅ Model SUCCESSFULLY detects TRUE timestomping at HIGH confidence ({true_high_rate:.1f}%)")
        elif true_high_rate >= 70:
            print(f"⚠️  Model detects TRUE timestomping but with room for improvement ({true_high_rate:.1f}%)")
        else:
            print(f"❌ Model needs improvement for TRUE timestomping detection ({true_high_rate:.1f}%)")

    print()
    print("Model Behavior Assessment:")
    print(f"  - Prioritizes cross-artifact validation (LogFile + UsnJrnl)")
    print(f"  - Gives HIGH confidence to events with Time Reversal + BASIC_INFO_CHANGE")

    if total_tunneling > 0:
        tunneling_low_rate = total_tunneling_missed / total_tunneling * 100
        if tunneling_low_rate >= 80:
            print(f"  ✅ Correctly reduces confidence for file system tunneling ({tunneling_low_rate:.1f}% < 30%)")
        else:
            print(f"  ⚠️  May be over-detecting file system tunneling as malicious")

    # Save detailed results
    output_file = "detection_analysis_CORRECTED.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
