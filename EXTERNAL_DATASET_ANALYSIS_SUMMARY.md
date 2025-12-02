# External Dataset Analysis Summary

## Overview

Analyzed 14 external APT datasets to identify overfitting patterns in the current model (v1.0).

**Date**: 2025-12-01
**Model**: XGBoost baseline (trained on Cases 01-12 PE)
**External Datasets**: 14 APT cases (APT17, APT19, APT21, APT28, APT29, APT30, APT37, APT38, APT40, DarkHotel x2, Kimsuky, Winnti x2)

---

## Critical Findings

### 🚨 **COMPLETE MODEL FAILURE ON EXTERNAL DATA**

**Detection Performance**:
- **Total timestomped events**: 42 across 14 datasets
- **Detected HIGH confidence (≥70%)**: 0 (0.0%)
- **Detected ANY confidence (≥30%)**: 0 (0.0%)
- **Detection rate**: **0%** on ALL external APT datasets

**Model Behavior**:
- Model detects `\Windows\Temp\` log files instead of actual timestomped files
- Most common false positives:
  - `\Windows\Temp\DESKTOP-JK343U8-20221222-####.log` (71.5-87.5% confidence)
  - `\Program Files\Microsoft OneDrive\*.exe` and `*.xml` files (94-99% confidence)

---

## Feature Distribution Analysis

### Training Data (PE Cases 01-12) vs External Data (APT Cases)

| Feature | PE Training (Timestomped) | APT External (All Events) | Difference |
|---------|---------------------------|---------------------------|------------|
| **in_temp_dir** | **~40-50%** | **3.1%** | **-40%** |
| **in_program_files** | ~15-20% | 3.4% | -15% |
| **in_windows_dir** | ~30% | 9.0% | -21% |
| **has_logfile_evidence** | ~7% | 19.2% | +12% |

**Key Insight**: APT datasets have VERY FEW events in `\Windows\Temp\` (3.1%) compared to PE training data (40-50%). The model learned "temp_dir = timestomped" instead of learning actual timestamp manipulation patterns.

---

## Root Cause: Location-Based Overfitting

### What the Model Learned (WRONG):
1. **Files in `\Windows\Temp\`** → HIGH confidence timestomped
2. **Files in `\Program Files\`** → Maybe timestomped
3. **Files with LogFile evidence** → Somewhat suspicious

### What the Model SHOULD Learn (CORRECT):
1. **Time Reversal Event** (LogFile) + **BASIC_INFO_CHANGE** (UsnJrnl) → HIGH confidence
2. **Impossible timestamp sequences** → HIGH confidence
3. **Zero nanoseconds** in timestamps → Suspicious
4. **Cross-artifact validation** → Increase confidence

---

## Per-Dataset Results

| Dataset | Events | Ground Truth Timestomped | HIGH Detections | Detection Rate | Top False Positive |
|---------|--------|--------------------------|-----------------|----------------|-------------------|
| 01-apt17 | 23,135 | 2 | 16 (0.07%) | 0% | `\Windows\Temp\...-1344.log` (71.5%) |
| 02-apt19 | 23,526 | 2 | 16 (0.07%) | 0% | `\Windows\Temp\...-2224.log` (87.5%) |
| 03-apt21 | 22,317 | 2 | 16 (0.07%) | 0% | `\Windows\Temp\...-2224.log` (71.5%) |
| 04-apt28 | 23,200 | 2 | 16 (0.07%) | 0% | `\Windows\Temp\...-0707.log` (71.5%) |
| 05-apt29 | 23,843 | **12** | 16 (0.07%) | 0% | `\Windows\Temp\...-1824.log` (71.5%) |
| 06-apt30 | 23,836 | 2 | 17 (0.07%) | 0% | `\Windows\Temp\...-2224a.log` (84.7%) |
| 07-apt37 | 23,172 | 2 | 16 (0.07%) | 0% | `\Windows\Temp\...-0844.log` (71.5%) |
| 08-apt38 | 22,265 | 2 | 16 (0.07%) | 0% | `\Windows\Temp\...-0344.log` (71.5%) |
| 09-apt40 | 23,273 | 2 | 16 (0.07%) | 0% | `\Windows\Temp\...-1344.log` (71.5%) |
| 10-darkhotel663 | 17,446 | 4 | 3 (0.02%) | 0% | `OneDrive.VisualElementsManifest.xml` (98.5%) |
| 11-darkhotelbbd | 17,418 | 6 | 3 (0.02%) | 0% | `OneDrive.VisualElementsManifest.xml` (98.5%) |
| 12-kimusky | 17,418 | 0 (no GT) | 3 (0.02%) | N/A | `OneDrive.VisualElementsManifest.xml` (98.5%) |
| 13-winnti731 | 14,082 | 2 | 2 (0.01%) | 0% | `OneDriveStandaloneUpdater.exe` (94.3%) |
| 14-winnti43b | 15,963 | 2 | 3 (0.02%) | 0% | `OneDrive.VisualElementsManifest.xml` (99.0%) |
| **TOTAL** | **276,894** | **42** | **147 (0.05%)** | **0%** | - |

---

## Observations

### Pattern 1: WindowsTemp Log File False Positives
- **APT17-APT40** datasets (9 cases): All flag `\Windows\Temp\DESKTOP-JK343U8-***.log` files
- Probability: 71.5% - 87.5% (HIGH confidence)
- **Reality**: These are likely benign system log files
- **Root Cause**: Model learned temp_dir = suspicious from PE training data

### Pattern 2: OneDrive File False Positives
- **DarkHotel, Kimsuky, Winnti** datasets (5 cases): All flag OneDrive files
- Probability: 94.3% - 99.0% (VERY HIGH confidence)
- Files: `OneDrive.exe`, `OneDriveStandaloneUpdater.exe`, `OneDrive.VisualElementsManifest.xml`
- **Reality**: Legitimate Microsoft files
- **Root Cause**: Model learned program_files = suspicious

### Pattern 3: Complete Failure to Detect Actual Timestomping
- **0/42 timestomped files detected** across all external datasets
- Even with 19.2% LogFile evidence availability (higher than training!)
- Actual timestomped files are in different locations (e.g., `\Windows\SysWOW64\`, `\Windows\System32\`)

---

## Recommendations for v2.0

### 1. Ground Truth Cleaning
- Remove file system tunneling events from positive class (233 events)
- Keep only TRUE malicious timestomping (19 events with LogFile evidence)
- Add file system tunneling as explicit negative examples

###  2. Feature Engineering Improvements
**Remove/Reduce**:
- `in_temp_dir` (30% importance → causes massive overfitting)
- `in_program_files` (11.5% importance → false positives)
- Location-based features should be < 10% total importance

**Add/Enhance**:
- **Time Reversal Event detection** (LogFile signature)
- **BASIC_INFO_CHANGE pattern** (UsnJrnl signature)
- **Cross-artifact validation score** (both artifacts agree = HIGH confidence)
- **Timestamp anomaly features**:
  - Zero nanoseconds in FILETIME
  - Copied timestamps (multiple files with identical timestamps)
  - Impossible sequences (created > modified)
- **File system tunneling detection** (15-second window, rename patterns)

### 3. Training Data Diversification
**Add to Training Set** (select 4-6 datasets):
- **05-APT29**: 12 timestomped events (good diversity)
- **11-DarkHotelbbd**: 6 timestomped events
- **10-DarkHotel663**: 4 timestomped events
- **01-APT17, 02-APT19, 03-APT21**: 2 events each

**Reserve for Testing**:
- **13-Winnti731, 14-Winnti43b**: Winnti-specific patterns
- **07-APT37, 08-APT38, 09-APT40**: Different APT groups

### 4. Evaluation Strategy
**Stratified Metrics**:
- TRUE malicious timestomping: Target >90% HIGH confidence detection
- File system tunneling: Target <10% HIGH confidence (correctly LOW)
- External APT test set: Target >70% HIGH confidence detection

**Cross-Validation**:
- Train on PE (01-12) + APT (05, 10, 11) + 3 others
- Test on held-out APT datasets (Winnti, APT37/38/40)
- Ensure model generalizes across different attack tools and locations

---

## Next Steps

1. ✅ **Analysis Complete**: Identified location-based overfitting as root cause
2. 📋 **Select Training Datasets**: Choose 4-6 APT datasets to add to training
3. 📋 **Re-engineer Features**: Implement timestamp anomaly and cross-artifact validation features
4. 📋 **Clean Ground Truth**: Separate malicious from tunneling in PE datasets
5. 📋 **Retrain Model**: XGBoost with diverse data and improved features
6. 📋 **Validate on External Test Set**: Held-out APT datasets

---

## Conclusion

The current model (v1.0) **completely fails** on external APT datasets due to severe location-based overfitting. The model learned shortcuts like "files in temp directory are timestomped" instead of learning actual forensic indicators of timestamp manipulation.

**Good News**: We now have:
- 14 diverse APT datasets for training/testing
- Clear understanding of overfitting root cause
- Concrete roadmap for v2.0 improvements

**Target for v2.0**:
- >90% detection on TRUE malicious timestomping (PE + APT)
- <10% false positives on file system tunneling
- >70% detection on external test sets (generalization proof)
