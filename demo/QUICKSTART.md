# Quick Start Guide

## ⚡ 5-Minute Setup

### 1. Install Dependencies (One-Time Setup)

```bash
cd demo
./setup.sh
```

This creates a virtual environment and installs all required packages.

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

**Important:** You must activate the virtual environment every time you want to use the demo scripts!

---

## 🚀 Running the Demos

### Option 1: Quick Demo (Test with Existing Data)

```bash
python predict_timestomping.py \
  ../data/processed/Phase\ 2\ -\ Feature\ Engineering/features_engineered.csv \
  --verbose
```

**Expected:** Runs in ~30 seconds, flags ~157 files

### Option 2: Full Pipeline (New Unlabeled Data)

```bash
python full_pipeline_demo.py \
  path/to/LogFile.csv \
  path/to/UsnJrnl.csv \
  --verbose
```

**Expected:** Runs in 5-10 minutes for ~1M events

---

## 📊 Check Results

```bash
# View summary
cat demo_results/summary_report.txt

# Check flagged files
head demo_results/flagged_files.csv
```

---

## 🛑 When Done

```bash
deactivate
```

---

## ❓ Troubleshooting

### "No module named 'pandas'"
- Make sure you activated the virtual environment: `source venv/bin/activate`
- See the `(venv)` prefix in your terminal prompt

### "Model file not found"
- Ensure Phase 3 (Model Training) is complete
- Model should exist at: `../data/processed/Phase 3 - Model Training/v3_final/random_forest_model_final.joblib`

### Need more help?
- Read full [README.md](README.md)
- Run `python predict_timestomping.py --help`

---

## 📋 Cheat Sheet

```bash
# Setup (once)
./setup.sh

# Activate (every session)
source venv/bin/activate

# Run Option 1
python predict_timestomping.py <features.csv>

# Run Option 2
python full_pipeline_demo.py <logfile.csv> <usnjrnl.csv>

# Deactivate (when done)
deactivate
```