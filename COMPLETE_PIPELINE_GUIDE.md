# Complete End-to-End Pipeline - Usage Guide

## Overview

`detect_complete.py` is a production-ready script that processes raw LogFile and UsnJrnl CSV files through the complete Digital Detectives pipeline.

**Pipeline Stages**:
1. **Phase 1**: Data cleaning and smart union merging
2. **Phase 2**: Feature engineering (26 features)
3. **Phase 3**: Model inference and prediction

---

## Quick Start

```bash
source .venv/bin/activate
python detect_complete.py
```

Follow the interactive prompts to provide:
- LogFile CSV path
- UsnJrnl CSV path
- Detection threshold (default: 0.5)
- Output directory (default: ./results)

---

## Input Requirements

### LogFile CSV Format

**Required columns**:
- `LSN` - Log Sequence Number
- `EventTime(UTC+8)` or `EventTime` - Timestamp
- `Full Path` or `FullPath` - File path
- `File/Directory Name` - Filename

**Optional columns** (recommended):
- `Event` - Event type
- `Detail` - Event details
- `CreationTime`, `ModifiedTime`, `MFTModifiedTime`, `AccessedTime`
- `Redo`, `Target VCN`, `Cluster Index`

### UsnJrnl CSV Format

**Required columns**:
- `USN` - Update Sequence Number
- `TimeStamp(UTC+8)` or `TimeStamp` - Timestamp
- `FullPath` or `Full Path` - File path
- `File/Directory Name` - Filename

**Optional columns** (recommended):
- `EventInfo` - USN event information
- `SourceInfo`, `FileAttribute`, `Carving Flag`
- `FileReferenceNumber`, `ParentFileReferenceNumber`

---

## Features Engineered

The pipeline creates **26 features** required by the model:

### File-Level Features (4)
1. `filename_length` - Length of filename
2. `path_depth` - Number of directory levels
3. `is_executable` - Boolean: .exe, .dll, .sys, etc.
4. `is_archive` - Boolean: .zip, .rar, .7z, etc.

### Path-Based Features (4)
5. `in_temp_dir` - Boolean: in \temp\ or \tmp\
6. `in_windows_dir` - Boolean: in \windows\
7. `in_program_files` - Boolean: in \program files\
8. `in_users_dir` - Boolean: in \users\

### Frequency Features (2)
9. `event_frequency_per_file` - Events per file
10. `event_frequency_per_case` - Total events in dataset

### Temporal Features (4)
11. `time_since_previous_event_seconds` - Time since last event
12. `time_until_next_event_seconds` - Time until next event
13. `events_in_1min_window` - Events within ±1 minute
14. `events_in_5min_window` - Events within ±5 minutes

### Cross-Artifact Features (2)
15. `has_logfile_evidence` - Boolean: file appears in LogFile
16. `usn_complete_manipulation_pattern` - Boolean: manipulation keywords

### Preserved Columns
- `lf_lsn` - LogFile LSN (for forensic tracking)
- `usn_usn` - UsnJrnl USN (for forensic tracking)
- `lf_event` - LogFile event description
- `usn_event_info` - USN event details
- `source` - Data source (logfile_only/usnjrnl_only/both)

---

## Output Files

### 1. predictions.csv
All analyzed events with predictions.

**Columns**:
```
filepath, filename, eventtime, source, lf_lsn, usn_usn,
lf_event, usn_event_info, probability, prediction, risk_level
```

### 2. flagged_files.csv
Only events flagged as suspicious (prediction=1), sorted by probability (highest first).

**Same columns as predictions.csv**

### 3. summary_report.txt
Detailed analysis report including:
- Summary statistics
- Risk level breakdown (CRITICAL/HIGH/MEDIUM/LOW)
- Top 10 flagged files with details
- LSN/USN information for forensic tracking

---

## Risk Classification

| Risk Level | Probability Range | Meaning |
|------------|------------------|---------|
| **CRITICAL** | ≥ 0.7 | Very high confidence - almost certainly timestomped |
| **HIGH** | 0.5 - 0.7 | High confidence - likely timestomped |
| **MEDIUM** | 0.3 - 0.5 | Moderate concern - suspicious activity |
| **LOW** | < 0.3 | Low risk - unlikely timestomped |

---

## Threshold Selection

| Threshold | Files Flagged | Use Case |
|-----------|---------------|----------|
| **0.3** | MEDIUM and above | High sensitivity - investigative phase |
| **0.5** | HIGH and above | **Recommended** - balanced approach |
| **0.7** | CRITICAL only | High confidence - conclusive evidence |

---

## Example Usage

### Basic Usage
```bash
source .venv/bin/activate
python detect_complete.py
```

When prompted:
```
LogFile CSV Path: data/raw/logfile/01-PE-LogFile.csv
UsnJrnl CSV Path: data/raw/usnjrnl/01-PE-UsnJrnl.csv
Detection Threshold: 0.5
Output Directory: results/case_01
Proceed? y
```

### Command-Line Usage (Non-Interactive)

You can run the test script:
```bash
bash test_complete_pipeline.sh
```

Or modify the test script for your own data.

---

##Processing Time

**Expected processing time**:
- Small dataset (<10K events): 30-60 seconds
- Medium dataset (10K-100K events): 1-5 minutes
- Large dataset (100K-500K events): 5-15 minutes

**Note**: Windowed event features (events_in_1min_window, events_in_5min_window) are the most computationally expensive, requiring O(n²) time complexity.

---

## Troubleshooting

### Error: "Model not found"
```
ERROR: Model not found at models/xgboost_model.pkl
```
**Solution**: Ensure you've completed Phase 3 and the model is saved in `models/xgboost_model.pkl`

### Error: Missing columns
```
KeyError: 'Full Path'
```
**Solution**: Check your CSV column names match the expected format. The script supports common variations:
- `Full Path` or `FullPath`
- `EventTime(UTC+8)` or `EventTime`
- `TimeStamp(UTC+8)` or `TimeStamp`

### Slow processing on large datasets
**Solution**: The windowed event features are computationally expensive. For very large datasets (>100K events), consider:
- Processing in batches
- Running on a more powerful machine
- Optimizing the windowed feature calculation

### No detections (0 files flagged)
**Possible causes**:
1. Dataset is genuinely clean (no timestomping)
2. Feature engineering differs from training data
3. Threshold is too high

**Solution**: Try lowering the threshold to 0.3 and check `predictions.csv` for probability distribution.

---

## Validation

### Testing on Case 12 (Known Data)

To verify the pipeline works correctly, test on Case 12:

```bash
python detect_complete.py
# LogFile: data/raw/logfile/12-PE-LogFile.csv
# UsnJrnl: data/raw/usnjrnl/12-PE-UsnJrnl.csv
# Threshold: 0.5
# Output: test_outputs/validation
```

**Expected results**:
- Should detect ~70-80 suspicious files
- Top detections should be in `\Windows\Temp\` directory
- High CRITICAL risk files (probability > 0.95)

### Testing on New Data

When testing on completely new forensic data:

1. **Run the pipeline**:
```bash
python detect_complete.py
```

2. **Check output**:
```bash
cat results/summary_report.txt
head results/flagged_files.csv
```

3. **Validate results**:
- Review flagged files manually
- Check if known timestomped files are detected
- Investigate false positives/negatives

4. **Adjust threshold if needed**:
- Too many false positives? Increase threshold to 0.6-0.7
- Missing known timestomps? Decrease threshold to 0.3-0.4

---

## Integration with Forensic Workflow

### Step 1: Extract NTFS Artifacts
- Use Autopsy, FTK Imager, or Eric Zimmerman's tools
- Export $LogFile and $UsnJrnl as CSV

### Step 2: Run Detection
```bash
python detect_complete.py \
  --logfile case_001/LogFile.csv \
  --usnjrnl case_001/UsnJrnl.csv \
  --output results/case_001
```

### Step 3: Investigate Flagged Files
- Review `flagged_files.csv`
- Start with CRITICAL risk files (probability ≥ 0.7)
- Use LSN/USN to locate events in original CSV files
- Cross-reference with timeline analysis

### Step 4: Document Findings
- Use `summary_report.txt` as basis for forensic report
- Include probability scores and risk levels
- Document investigation methodology

---

## Comparison: Complete Pipeline vs. Processed Data Tool

| Feature | `detect_complete.py` | `detect_processed.py` |
|---------|----------------------|----------------------|
| **Input** | Raw LogFile + UsnJrnl CSV | Phase 2C processed CSV |
| **Processing** | Full Phase 1 + 2 + 3 | Phase 3 only |
| **Use Case** | New forensic cases | Testing on existing data |
| **Speed** | Slower (feature engineering) | Faster (pre-processed) |
| **Ground Truth** | No (unknown data) | Yes (if available) |
| **Production Ready** | ✅ Yes | ⚠️ Test only |

---

## Next Steps

### For Testing
1. Run on Case 12 to verify pipeline works
2. Compare results with `detect_processed.py` output
3. Should get similar detection rates

### For Production Use
1. Test on completely new forensic data (NOT Cases 1-12)
2. Validate detections manually
3. Adjust threshold based on false positive rate
4. Document performance metrics

### For Autopsy Integration
1. Validate pipeline on external data first
2. Confirm performance metrics are acceptable
3. Then proceed with Autopsy module development

---

## Performance Expectations

**Model Performance** (based on Case 12):
- **Recall**: 98.57% (catches 98.57% of timestomped files)
- **Precision**: 84.15% (84.15% of flagged files are truly timestomped)
- **F1-Score**: 0.9079
- **False Negative Rate**: 1.43% (misses ~1-2% of timestomps)
- **False Positive Rate**: 0.25% (flags ~0.25% of clean files)

---

## Support

For issues or questions:
1. Check this guide first
2. Review `MODEL_EVALUATION_REPORT.md`
3. Check output files for error messages
4. Verify input CSV format matches requirements

---

## License

Part of the Digital Detectives thesis project.
For research and educational use.
