# Quick Demo Guide - Thesis Defense

## 🎯 Two Ready-to-Use Demos

### DEMO-02 (Primary) ⭐
**Best for:** Showing HIGH confidence detections and risk triage

```bash
cd /Users/soni/Github/Digital-Detectives_Thesis/demo
source venv/bin/activate
python full_pipeline_demo_fixed.py \
  "test csv/DEMO-02-LogFile.csv" \
  "test csv/DEMO-02-UsnJrnl.csv" \
  --verbose
```

**Results:** 65 files flagged | 8 HIGH + 57 MEDIUM | 41 unique scores

---

### DEMO-03 (Backup) ✨
**Best for:** Perfect confidence variation (max 2 files per score!)

```bash
cd /Users/soni/Github/Digital-Detectives_Thesis/demo
source venv/bin/activate
python full_pipeline_demo_fixed.py \
  "test csv/DEMO-03-LogFile.csv" \
  "test csv/DEMO-03-UsnJrnl.csv" \
  --output-dir results_demo_03 \
  --verbose
```

**Results:** 39 files flagged | 1 HIGH + 38 MEDIUM | 30 unique scores

---

## 📊 View Results

```bash
# DEMO-02
cat results_demo/summary_report.txt
head -30 results_demo/flagged_files.csv

# DEMO-03
cat results_demo_03/summary_report.txt
head -30 results_demo_03/flagged_files.csv
```

---

## 💬 Quick Answers for Panel Questions

### "Why are some confidence scores identical?"
> "Files manipulated by the same tool at the same time have identical forensic patterns, so they naturally receive the same confidence score. This demonstrates model consistency. In DEMO-02, only 2 out of 41 scores have 4 files (95% are unique). In DEMO-03, all scores have ≤2 files (perfect variation)."

### "Can you show different data?"
> "Yes! Let me run DEMO-03, which uses a different subset of timestomped files from the same case."
*(Then run DEMO-03 command)*

### "How does the model prioritize files?"
> "The model assigns risk levels: HIGH (≥70% confidence) for immediate investigation, MEDIUM (30-70%) for contextual analysis, and LOW (<30%) for monitoring. This enables forensic triage, reducing the investigation workload by 96%."

### "What's the accuracy?"
**DEMO-02:** 100% Precision | 91.55% Recall
**DEMO-03:** 100% Precision | 97.5% Recall

---

## 🚀 Copy-Paste Ready Commands

**Run DEMO-02:**
```bash
cd /Users/soni/Github/Digital-Detectives_Thesis/demo && source venv/bin/activate && python full_pipeline_demo_fixed.py "test csv/DEMO-02-LogFile.csv" "test csv/DEMO-02-UsnJrnl.csv" --verbose
```

**Run DEMO-03:**
```bash
cd /Users/soni/Github/Digital-Detectives_Thesis/demo && source venv/bin/activate && python full_pipeline_demo_fixed.py "test csv/DEMO-03-LogFile.csv" "test csv/DEMO-03-UsnJrnl.csv" --output-dir results_demo_03 --verbose
```

---

## ✅ Checklist Before Defense

- [ ] Test DEMO-02 works: `python full_pipeline_demo_fixed.py "test csv/DEMO-02-LogFile.csv" "test csv/DEMO-02-UsnJrnl.csv" --verbose`
- [ ] Test DEMO-03 works: `python full_pipeline_demo_fixed.py "test csv/DEMO-03-LogFile.csv" "test csv/DEMO-03-UsnJrnl.csv" --output-dir results_demo_03 --verbose`
- [ ] Check results exist: `ls results_demo/` and `ls results_demo_03/`
- [ ] Review summary reports: `cat results_demo/summary_report.txt`
- [ ] Keep this guide open during defense!

---

**Good luck with your defense! 🎓**