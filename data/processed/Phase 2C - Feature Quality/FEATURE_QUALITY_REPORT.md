# Phase 2C Feature Quality Analysis Report

**Date**: 2025-11-27
**Input**: Phase 2B output (63 columns, 154,550 records)
**Output**: Final curated dataset (16 features)

---

## Summary

- **Total features analyzed**: 25
- **Features selected**: 16
- **Features removed**: 9
- **Selection criteria**: Importance >= 0.5%, correlation < 0.95
- **Cumulative importance**: 97.92%

---

## Selected Features (16)

| Rank | Feature | Importance (%) | Group |
|----|---------|----------------|-------|
| 1 | `in_temp_dir` | 30.02 | Location |\n| 2 | `event_frequency_per_file` | 20.53 | Temporal |\n| 3 | `events_in_5min_window` | 11.30 | Temporal |\n| 4 | `events_in_1min_window` | 6.20 | Temporal |\n| 5 | `filename_length` | 5.65 | File Type |\n| 6 | `path_depth` | 5.44 | Location |\n| 7 | `is_archive` | 5.38 | File Type |\n| 8 | `in_program_files` | 3.16 | Location |\n| 9 | `event_frequency_per_case` | 2.54 | Temporal |\n| 10 | `in_windows_dir` | 2.50 | Location |\n| 11 | `usn_complete_manipulation_pattern` | 1.31 | Pattern |\n| 12 | `time_until_next_event_seconds` | 1.11 | Temporal |\n| 13 | `is_executable` | 0.95 | File Type |\n| 14 | `has_logfile_evidence` | 0.77 | Cross-Artifact |\n| 15 | `time_since_previous_event_seconds` | 0.53 | Temporal |\n| 16 | `in_users_dir` | 0.52 | Location |\n\n---

## Removed Features (8)

| Feature | Importance (%) | Reason |
|---------|----------------|--------|
| `in_system32` | 0.061 | Low importance (<0.5%) |\n| `is_system_file` | 0.002 | Low importance (<0.5%) |\n| `is_hidden_file` | 0.002 | Low importance (<0.5%) |\n| `has_suspicious_extension` | 0.028 | Low importance (<0.5%) |\n| `source_confidence_score` | 0.167 | Low importance (<0.5%) |\n| `has_usnjrnl_evidence` | 0.139 | Low importance (<0.5%) |\n| `usn_basic_info_change` | 0.285 | Low importance (<0.5%) |\n| `event_vs_modified_after_days` | 0.177 | Low importance (<0.5%) |\n\n## Highly Correlated Pairs\n\n- `source_confidence_score` <-> `has_logfile_evidence` (r=0.960)\n- `has_usnjrnl_evidence` <-> `usn_basic_info_change` (r=1.000)\n- `usn_file_closed` <-> `usn_complete_manipulation_pattern` (r=1.000)\n\n---

## Data Integrity

- ✓ Total records: 154,550 (unchanged)
- ✓ Timestomped events: 252 (unchanged)
- ✓ Benign events: 154,298 (unchanged)

---

## Files Generated

1. **Final Dataset**: `all_cases_combined_final_features.csv` (16 features + 10 essential columns)
2. **Feature Importance**: `feature_importance_rankings.csv`
3. **Visualizations**:
   - `boolean_features_distribution.png`
   - `continuous_features_distribution.png`
   - `correlation_heatmap.png`
   - `feature_importance_chart.png`

---

## Next Steps

**Phase 3: Baseline Model Training**
- Train 5 ML algorithms (Random Forest, Logistic Regression, XGBoost, LightGBM, Neural Network)
- Evaluate performance with selected features
- Compare model performance
- Select best model for production

---

## Conclusion

Feature selection complete. Selected 16 high-quality features representing 97.92% of total feature importance. Dataset is ML-ready for Phase 3 baseline model training.
