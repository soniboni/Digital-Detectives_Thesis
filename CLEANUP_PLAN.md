# Cleanup Plan for Fresh Model Training Setup

## Current Branch: model-training-soni-1

## ✅ KEEP (Essential Data & Code)
```
data/
├── raw/                           # Original forensic artifacts (KEEP)
│   ├── logfile/                   # Raw $LogFile exports
│   ├── usnjrnl/                   # Raw $UsnJrnl exports
│   └── suspicious/                # Ground truth labels
│
└── processed/
    ├── Phase 1 - Data Collection & Preprocessing/
    │   ├── A. Data Labelled/      # Labeled CSVs (KEEP)
    │   ├── B. Data Case Merging/  # Merged cases (KEEP)
    │   └── C. Master Timeline/    # master_timeline.csv (KEEP - will use as input)
    │
    ├── Phase 2 - Feature Engineering/
    │   └── features_engineered.csv  # OLD - reference only, will regenerate
    │
    └── Phase 3 - Model Training/   # DELETE ENTIRE FOLDER - will regenerate

notebooks/
├── Phase 1/                       # KEEP - for reference
├── Phase 2/                       # KEEP - for reference (but won't use)
└── Phase 3/                       # KEEP - for reference

README.md                          # KEEP
requirements.txt                   # KEEP - may update
```

## 🗑️ DELETE (To Clean Up)
```
data/processed/Phase 2 - Feature Engineering/
└── features_engineered.csv        # 324 MB - OLD, has broken features

data/processed/Phase 3 - Model Training/  # DELETE ENTIRE FOLDER (35 MB)
├── *.png                          # Old visualizations
├── *.csv                          # Old metrics
├── *.joblib                       # Old models (7.8 MB)
├── isolation_forest/              # Failed approach (9.2 MB)
├── v2_experiments/                # Old experiments (396 KB)
└── v3_final/                      # Old final model (17 MB)

demo/                              # 388 MB - Demo folder (optional)
└── venv/                          # Python virtual env - large

model package/                     # 462 MB - Old deployment package
└── venv/                          # Python virtual env - large

models/                            # Empty folder (can recreate)
outputs/                           # Empty folder (can recreate)
src/                               # Empty folder (can recreate)

Autopsy File Ingest Module/       # WIP - not needed for training (OPTIONAL)
```

## 📦 TOTAL SPACE TO RECLAIM
- Phase 2 features_engineered.csv: ~324 MB
- Phase 3 Model Training folder: ~35 MB
- demo/venv/: ~388 MB
- model package/venv/: ~462 MB
**Total: ~1.2 GB**

## 🎯 NEW STRUCTURE (After Cleanup)
```
Digital-Detectives_Thesis/
├── data/
│   ├── raw/                       # Source data (unchanged)
│   │   ├── logfile/
│   │   ├── usnjrnl/
│   │   └── suspicious/
│   │
│   └── processed/
│       ├── Phase 1 - Data Collection & Preprocessing/
│       │   ├── A. Data Labelled/
│       │   ├── B. Data Case Merging/
│       │   └── C. Master Timeline/
│       │       └── master_timeline.csv  ← START HERE
│       │
│       ├── Phase 0 - Data Cleaning/  (NEW - will create)
│       │   ├── cleaned_data.csv
│       │   └── data_quality_report.txt
│       │
│       ├── Phase 2 - Feature Engineering/  (REGENERATE)
│       │   └── features_engineered_v2.csv
│       │
│       └── Phase 3 - Model Training/  (REGENERATE)
│           └── final_model/
│
├── notebooks/
│   ├── Phase 0 - Data Cleaning/  (NEW - will create)
│   │   └── Data Cleaning.ipynb
│   ├── Phase 1/  (keep for reference)
│   ├── Phase 2/  (keep for reference)
│   └── Phase 3/  (keep for reference)
│
├── README.md
├── requirements.txt
└── .gitignore
