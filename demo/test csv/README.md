# Demo Test CSV Files

This folder contains demonstration CSV files for testing the timestomping detection pipeline.

## Available Demo Files

### 1. DEMO-FeatureEngineered.csv ✅ **RECOMMENDED FOR DEMO**

**Purpose**: Demonstrate timestomping detection with high accuracy

**Content**:
- 2,100 events (100 timestomped + 2,000 benign)
- Pre-engineered features ready for model inference
- Contains ground truth labels for validation

**Usage**:
```bash
cd demo
source venv/bin/activate
python predict_timestomping.py "test csv/DEMO-FeatureEngineered.csv" --verbose
```

**Expected Results**:
- **71 files flagged** for investigation
- **46 HIGH risk** detections (≥70% confidence)
- **25 MEDIUM risk** detections (30-70% confidence)
- **97.18% Precision** (69 true positives, 2 false positives)
- **69.00% Recall** (69/100 timestomped files detected)

**Output Files** (in `demo_results/`):
- `predictions.csv` - All 2,100 predictions with confidence scores
- `flagged_files.csv` - 71 timestomped files sorted by confidence
- `summary_report.txt` - Detection statistics

---

### 2. DEMO-Case6-LogFile.csv + DEMO-Case6-UsnJrnl.csv

**Purpose**: Demonstrate full pipeline from raw NTFS artifacts

**Content**:
- LogFile: 204 events (50 timestomped)
- UsnJrnl: 905 events (50 timestomped)
- Extracted from Case 6 (PE dataset)
- Includes WindowsUpdate.*.etl files with confirmed timestomping

**Usage**:
```bash
cd demo
source venv/bin/activate
python full_pipeline_demo.py \
  "test csv/DEMO-Case6-LogFile.csv" \
  "test csv/DEMO-Case6-UsnJrnl.csv" \
  --verbose
```

**Note**: These files demonstrate the complete pipeline (merge → feature engineering → prediction), but may produce lower detection rates due to feature distribution differences between raw and engineered data.

**Output Files** (in `full_pipeline_results/`):
- `master_timeline.csv` - Merged timeline from both artifacts
- `features_engineered.csv` - ML-ready features
- `predictions.csv` - All predictions with confidence scores
- `flagged_files.csv` - Timestomped predictions
- `summary_report.txt` - Detection summary

---

### 3. DEMO-Case6-Merged.csv

**Purpose**: Single merged timeline file (for reference)

**Content**:
- 1,050 events (50 timestomped + 1,000 benign)
- Merged LogFile + UsnJrnl data
- Includes ground truth labels

**Note**: This is a reference file. Use the separate LogFile/UsnJrnl files with `full_pipeline_demo.py` or the feature-engineered file with `predict_timestomping.py`.

---

## Original Case 6 Files

### 06-PE-LogFile.csv + 06-PE-UsnJrnl.csv

**Purpose**: Full Case 6 dataset for complete analysis

**Content**: Complete NTFS artifacts from Case 6

**Usage**: Same as DEMO files but with full dataset

---

### 06-APT-LogFile.csv + 06-APT-UsnJrnl.csv + 06-APT-SuspiciousBehavior.csv

**Purpose**: Case 6 APT dataset with suspicious behavior indicators

**Content**:
- Full APT attack scenario
- SuspiciousBehavior.csv shows IEXPLORE.EXE was timestomped

---

## Quick Start

For the best demonstration experience:

```bash
# 1. Navigate to demo directory
cd demo

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run the demo with pre-engineered features (RECOMMENDED)
python predict_timestomping.py "test csv/DEMO-FeatureEngineered.csv" --verbose

# 4. Check results
cat demo_results/summary_report.txt
head -20 demo_results/flagged_files.csv
```

This will demonstrate:
- **HIGH confidence detections** (≥70% probability)
- **MEDIUM confidence detections** (30-70% probability)
- **Precision and recall** comparison with ground truth
- **Risk-based triage** for forensic investigation

---

## Creating Your Own Demo Files

Use the provided scripts to create custom demo datasets:

```bash
# Create demo from merged case data
python create_demo_data.py

# Create separate LogFile/UsnJrnl demo files
python create_demo_separate_files.py

# Create demo from engineered features (recommended)
python create_demo_for_predict_script.py
```

---

## Troubleshooting

**Q: Why does full_pipeline_demo.py show 0 detections?**

A: The raw CSV → feature engineering pipeline may produce different feature distributions than the training data. Use `DEMO-FeatureEngineered.csv` with `predict_timestomping.py` for reliable demo results.

**Q: Can I use my own forensic data?**

A: Yes! Export $LogFile and $UsnJrnl using [NTFS Log Tracker](http://www.forensic-proof.com/archives/3244) and run:
```bash
python full_pipeline_demo.py <your-logfile.csv> <your-usnjrnl.csv>
```

**Q: How do I interpret confidence scores?**

A:
- **HIGH (≥70%)**: Strong indicators of timestomping, high priority investigation
- **MEDIUM (30-70%)**: Mixed signals, contextual analysis recommended
- **LOW (<30%)**: Mostly false positives, lowest priority