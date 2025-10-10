# Timestomping Detection Demo Scripts

This directory contains two demonstration scripts for the timestomping detection system developed in this thesis.

---

## 📋 Overview

**What is Timestomping?**
Timestomping is an anti-forensic technique where attackers manipulate file timestamps (MAC times) to hide malicious activity or evade detection.

**What These Scripts Do:**
- Analyze NTFS forensic artifacts ($LogFile + $UsnJrnl) to detect timestomping
- Use machine learning (Random Forest) to identify suspicious timestamp manipulation
- Provide forensic triage by flagging high-risk files for investigation

---

## 🎯 Two Demo Options

### **Option 1: Quick Demo** (`predict_timestomping.py`)
- **Input:** Pre-engineered features CSV (from Phase 2)
- **Best for:** Testing the trained model quickly
- **Use case:** You already have feature-engineered data

### **Option 2: Full Pipeline Demo** (`full_pipeline_demo.py`)
- **Input:** Raw $LogFile CSV + $UsnJrnl CSV (unlabeled)
- **Best for:** Complete end-to-end demonstration
- **Use case:** You have raw NTFS artifacts from a forensic case

---

## 📦 Requirements

### Python Packages
```bash
pip install pandas numpy scikit-learn joblib imbalanced-learn
```

### Trained Model
Both scripts require the trained model:
```
../data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib
```

Make sure you've completed Phase 3 (Model Training) before running these demos.

---

## 🚀 Option 1: Quick Demo

### Usage

```bash
python predict_timestomping.py <input_features.csv> [options]
```

### Example

```bash
# Basic usage (using test data from Phase 3)
python predict_timestomping.py ../data/processed/Phase\ 2\ -\ Feature\ Engineering/features_engineered.csv

# With custom output directory
python predict_timestomping.py input.csv --output-dir ./case_001_results

# With custom confidence threshold (default: 0.3)
python predict_timestomping.py input.csv --threshold 0.5

# Verbose mode
python predict_timestomping.py input.csv --verbose
```

### Input Requirements

**Input CSV must contain these 75 engineered features:**
- Temporal features: `hour_of_day`, `day_of_week`, `is_weekend`, `is_off_hours`, etc.
- Timestamp anomalies: `creation_after_modification`, `accessed_before_creation`, etc.
- Path features: `path_depth`, `is_system_path`, `is_temp_path`, etc.
- Event patterns: `lf_event_encoded`, `usn_event_encoded`, etc.
- Cross-artifact features: `has_both_artifacts`, `merge_matched`, etc.

**Does NOT need:**
- Labels (`is_timestomped` column) - the model will predict this
- Raw text columns (these are dropped during feature preparation)

### Output Files

1. **`predictions.csv`** - All records with predictions
   - Columns: `prediction` (0=Benign, 1=Timestomped), `confidence` (0.0-1.0), `risk_level` (LOW/MEDIUM/HIGH)

2. **`flagged_files.csv`** - Only timestomped predictions (sorted by confidence)
   - Ready for forensic investigation

3. **`summary_report.txt`** - Detection statistics and top findings

### Expected Output Structure

```
demo_results/
├── predictions.csv         # All predictions with confidence scores
├── flagged_files.csv       # High-priority files for investigation
└── summary_report.txt      # Summary statistics
```

---

## 🔬 Option 2: Full Pipeline Demo

### Usage

```bash
python full_pipeline_demo.py <logfile.csv> <usnjrnl.csv> [options]
```

### Example

```bash
# Basic usage
python full_pipeline_demo.py LogFile.csv UsnJrnl.csv

# With options
python full_pipeline_demo.py LogFile.csv UsnJrnl.csv \
  --output-dir ./case_001_results \
  --case-id 1 \
  --threshold 0.4 \
  --save-intermediates \
  --verbose
```

### Input Requirements

**$LogFile CSV must contain:**
- `eventtime` - Timestamp in format `MM/DD/YY HH:MM:SS`
- `filename` - File name
- `filepath` - Full file path
- `lf_lsn` - Log Sequence Number
- `lf_event` - Event type (e.g., "File Creation", "File Deletion")
- `lf_creation_time`, `lf_modified_time`, `lf_mft_modified_time`, `lf_accessed_time` - MAC timestamps
- Other $LogFile-specific columns

**$UsnJrnl CSV must contain:**
- `eventtime` - Timestamp in format `MM/DD/YY HH:MM:SS`
- `filename` - File name
- `filepath` - Full file path
- `usn_usn` - Update Sequence Number
- `usn_event_info` - Event information
- `usn_file_attribute` - File attributes
- Other $UsnJrnl-specific columns

**Important:**
- Input CSVs do **NOT** need labels - the model will predict them
- Must follow the format from Phase 1 parsers
- Both files should be from the same forensic case

### Output Files

1. **`predictions.csv`** - All events with predictions
2. **`flagged_files.csv`** - Timestomped events (sorted by confidence)
3. **`summary_report.txt`** - Detection summary
4. **`master_timeline.csv`** - Merged timeline (if `--save-intermediates` used)
5. **`features_engineered.csv`** - ML features (if `--save-intermediates` used)

### Expected Output Structure

```
full_pipeline_results/
├── predictions.csv         # All predictions
├── flagged_files.csv       # Flagged for investigation
├── summary_report.txt      # Summary report
├── master_timeline.csv     # (Optional) Merged timeline
└── features_engineered.csv # (Optional) Engineered features
```

---

## 📊 Understanding the Output

### Prediction Columns

| Column | Type | Description |
|--------|------|-------------|
| `prediction` | 0 or 1 | 0 = Benign, 1 = Timestomped |
| `confidence` | 0.0 - 1.0 | Model's confidence score (probability) |
| `risk_level` | LOW/MEDIUM/HIGH | Risk categorization based on confidence |

### Risk Levels

- **HIGH** (≥70% confidence) - Strong indication of timestomping, investigate immediately
- **MEDIUM** (30-70% confidence) - Moderate suspicion, worth reviewing
- **LOW** (<30% confidence) - Likely benign, flagged due to conservative threshold

### Model Performance (from Phase 3 v3)

- **Precision:** 42.7% - Out of flagged files, ~43% are actually timestomped
- **Recall:** 65.7% - Catches ~66% of all timestomped files
- **F1-Score:** 0.517 - Balanced performance for imbalanced data
- **False Positive Rate:** 0.030% - Very low false alarm rate
- **Investigation Reduction:** 99.95% - Only need to review ~0.05% of files

**What this means:**
- The model is an effective **triage tool**
- Dramatically reduces investigation scope (from 300K files to ~150 files)
- Some false positives expected (conservative approach favors catching more threats)
- Manual verification still needed for flagged files

---

## 🛠️ Command-Line Options

### Option 1 (`predict_timestomping.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `./demo_results` | Directory to save results |
| `--threshold` | `0.3` | Confidence threshold for flagging (0.0-1.0) |
| `--model-path` | `../data/processed/Phase 3.../random_forest_model_final.joblib` | Path to trained model |
| `--verbose` | `False` | Show detailed progress |

### Option 2 (`full_pipeline_demo.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `./full_pipeline_results` | Directory to save results |
| `--threshold` | `0.3` | Confidence threshold for flagging (0.0-1.0) |
| `--case-id` | `1` | Case identifier |
| `--model-path` | `../data/processed/Phase 3.../random_forest_model_final.joblib` | Path to trained model |
| `--save-intermediates` | `False` | Save master timeline and features |
| `--verbose` | `False` | Show detailed progress |

---

## 💡 Tips & Best Practices

### Adjusting the Threshold

The `--threshold` parameter controls sensitivity:

- **Lower threshold (e.g., 0.2):** More files flagged (higher recall, more false positives)
- **Default threshold (0.3):** Balanced approach (recommended)
- **Higher threshold (e.g., 0.5):** Fewer files flagged (lower recall, fewer false positives)

**Recommendation:** Start with default (0.3), adjust based on your investigation priorities.

### Handling Large Datasets

For datasets with >1M events:
- Use `--verbose` to monitor progress
- Expect 5-15 minutes processing time
- Ensure ~2GB available memory
- Consider using `--save-intermediates` for debugging

### Interpreting Results

1. **Start with HIGH risk files** - These are most likely timestomped
2. **Review MEDIUM risk files** - Context-dependent (check file type, location)
3. **LOW risk files** - Generally safe to ignore unless specific concerns

### Common Issues

**"Model file not found"**
- Ensure Phase 3 (Model Training) is complete
- Check model path with `--model-path`

**"Feature count mismatch"**
- Input CSV missing required features (Option 1)
- Check that CSV is from Phase 2 Feature Engineering

**"Column not found in input CSV"**
- Raw CSV format doesn't match expected structure (Option 2)
- Verify CSV follows Phase 1 parser output format

---

## 📈 Example Workflow

### Forensic Investigation Scenario

1. **Extract NTFS artifacts** from suspect drive
   ```bash
   # Use MFTECmd, $LogFileParser, $UsnJrnlParser
   ```

2. **Run full pipeline demo**
   ```bash
   python full_pipeline_demo.py LogFile.csv UsnJrnl.csv \
     --output-dir ./suspect_case_001 \
     --case-id 1 \
     --save-intermediates \
     --verbose
   ```

3. **Review flagged files**
   ```bash
   # Open flagged_files.csv
   # Sort by confidence (highest first)
   # Investigate files with HIGH risk
   ```

4. **Manual verification**
   - Check file context (system vs. user files)
   - Verify MAC timestamp relationships
   - Cross-reference with other forensic evidence

---

## 🎓 Thesis Context

These demo scripts implement the timestomping detection system developed for:

**Thesis:** "Automated Detection of Timestomping in NTFS Forensic Artifacts using Machine Learning"

**Key Contributions:**
- Cross-artifact analysis ($LogFile + $UsnJrnl)
- Feature engineering for timestamp anomaly detection
- Practical triage tool reducing investigation scope by 99.95%

**Limitations:**
- Trained on specific timestomping tools (NTimeStomp, SetMACE)
- Limited dataset diversity (12 cases, controlled environment)
- Requires manual verification of flagged files

**Future Work:**
- Expand training data with more diverse timestomping techniques
- Develop Autopsy plugin for automated integration
- Improve detection of sophisticated timestamp manipulation

---

## 📞 Support

For questions or issues:
1. Check the usage instructions in script headers (`python script.py --help`)
2. Review the thesis documentation (notebooks/Phase 3 - Model Training/)
3. Verify all prerequisites are installed

---

## 📄 License

Part of thesis research project. See main repository for license details.

---

**Last Updated:** October 2025
**Model Version:** v3 Final (Minimal SMOTE 1:1000)
**Python Version:** 3.8+