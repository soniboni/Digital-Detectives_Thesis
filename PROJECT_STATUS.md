# Project Status: Fresh Setup Complete

## Branch: model-training-soni-1
## Date: 2025-11-24

---

## ✅ CLEANUP SUMMARY

### Deleted:
- ❌ Phase 2 features_engineered.csv (324 MB) - HAD BROKEN FEATURES
- ❌ Phase 3 Model Training folder (35 MB) - OLD MODELS
- ❌ demo/venv/ (388 MB)
- ❌ model package/venv/ (462 MB)

### Space Reclaimed: ~1.2 GB
### New Project Size: 3.3 GB (down from ~4.5 GB)

---

## 📁 CURRENT STRUCTURE

```
Digital-Detectives_Thesis/  (3.3 GB)
│
├── data/  (2.1 GB)
│   ├── raw/  (~2.0 GB)
│   │   ├── logfile/          # 12 $LogFile CSV files
│   │   ├── usnjrnl/          # 12 $UsnJrnl CSV files
│   │   └── suspicious/       # Ground truth labels
│   │
│   └── processed/  (~100 MB)
│       ├── Phase 0 - Data Cleaning/  (NEW - EMPTY)
│       │   └── (ready for cleaning output)
│       │
│       ├── Phase 1 - Data Collection & Preprocessing/
│       │   ├── A. Data Labelled/      # 24 labeled CSV files
│       │   ├── B. Data Case Merging/  # 12 merged case files
│       │   └── C. Master Timeline/
│       │       └── master_timeline.csv  ← START HERE (78 MB)
│       │
│       └── Phase 2 - Feature Engineering/
│           └── (empty - old features deleted)
│
├── notebooks/  (1.6 MB)
│   ├── Phase 0 - Data Cleaning/  (NEW - EMPTY)
│   ├── Phase 1 - Data Collection & Preprocessing/
│   ├── Phase 2 - Feature Engineering/
│   └── Phase 3 - Model Training/
│
├── demo/  (117 MB - no venv)
├── model package/  (190 MB - no venv)
├── Autopsy File Ingest Module/  (8 KB)
│
├── README.md
├── requirements.txt
├── CLEANUP_PLAN.md
└── PROJECT_STATUS.md
```

---

## 🎯 NEXT STEPS

### Phase 0: Data Cleaning (Week 1)
**Input:** `data/processed/Phase 1 - Data Collection & Preprocessing/C. Master Timeline/master_timeline.csv`

**Tasks:**
1. ✅ Fix anomaly features (MAC timestamp calculations)
2. ✅ Handle missing values (81% in timestamp deltas)
3. ✅ Remove duplicates (38,564 rows / 4.95%)
4. ✅ Resolve label conflicts (226 cases)
5. ✅ Cap outliers (events_per_minute)
6. ✅ Remove invalid rows (8 rows missing eventtime)

**Output:** `data/processed/Phase 0 - Data Cleaning/cleaned_data.csv`

---

### Phase 1: Feature Engineering (Week 2)
**Input:** Cleaned data from Phase 0

**Tasks:**
1. Remove 38 zero-importance features
2. Create continuous anomaly features
3. Add cross-file temporal patterns
4. Simplify cross-artifact features

**Output:** `data/processed/Phase 2 - Feature Engineering/features_engineered_v2.csv`

---

### Phase 2: Model Training (Week 3)
**Input:** Engineered features from Phase 1

**Tasks:**
1. Train Random Forest baseline
2. Try XGBoost/LightGBM
3. Ensemble best models

**Output:** `data/processed/Phase 3 - Model Training/final_model/`

---

## 📊 CRITICAL DATA ISSUES IDENTIFIED

### Issue #1: Broken Anomaly Features (CRITICAL)
```
creation_after_modification:  0 occurrences in timestomped, 24,703 in benign (BACKWARDS!)
has_future_timestamp:         0 occurrences in timestomped, 102,576 in benign (BACKWARDS!)
accessed_before_creation:     0 occurrences in timestomped, 29 in benign (BACKWARDS!)
```
**Root cause:** Calculated on `eventtime` instead of MAC timestamps

### Issue #2: Missing Values (81%)
```
creation_year_delta:  635,582 missing (81.62%)
modified_year_delta:  635,582 missing (81.62%)
```

### Issue #3: Duplicates (4.95%)
```
Total duplicate rows: 38,564 (4.95% of 778,692)
```

### Issue #4: Label Conflicts (226)
```
Files with conflicting LF vs USN labels: 226
```

### Issue #5: Dataset Info
```
Total records: 778,692
Timestomped files: 247 (0.032%)
Imbalance ratio: 1:3,151
```

---

## 🚀 READY TO START

**Current Status:** ✅ Clean slate, ready for Phase 0 (Data Cleaning)

**Starting Point:** `data/processed/Phase 1 - Data Collection & Preprocessing/C. Master Timeline/master_timeline.csv`

**Next Action:** Create `notebooks/Phase 0 - Data Cleaning/Data Cleaning.ipynb`
