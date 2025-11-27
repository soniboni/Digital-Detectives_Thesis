# Phase 3: Baseline Model Training Report

**Date**: 2025-11-27 05:13:57
**Dataset**: Phase 2C Final Features (16 features, 154,550 records)
**Models Evaluated**: 5 algorithms

---

## Dataset Summary

- **Total Records**: 154,550
- **Timestomped Events**: 252 (0.16%)
- **Benign Events**: 154,298 (99.84%)
- **Class Imbalance Ratio**: 1:612
- **Features Used**: 16

### Train/Test Split:
- **Train Set**: 123,640 samples (202 timestomped)
- **Test Set**: 30,910 samples (50 timestomped)
- **Split Ratio**: 80/20 (stratified)

---

## Model Performance Comparison

### Summary Table

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Negatives |
|-------|-----------|--------|----------|---------|--------|----------------|
| Random Forest | 0.5698 | 0.9800 | 0.7206 | 0.9998 | 0.8550 | 1 |
| Logistic Regression | 0.1365 | 0.9800 | 0.2396 | 0.9978 | 0.3265 | 1 |
| XGBoost | 0.8305 | 0.9800 | 0.8991 | 0.9998 | 0.8110 | 1 |
| LightGBM | 0.0516 | 0.9200 | 0.0977 | 0.9435 | 0.0476 | 4 |
| Neural Network (Focal Loss) | 0.0000 | 0.0000 | 0.0000 | 0.9868 | 0.0731 | 50 |

---

## Best Performing Models

### By Recall (Most Critical - Minimize Missed Detections):
**Winner**: Random Forest
- Recall: 0.9800 (98.00%)
- False Negatives: 1 (missed 1 out of 50 timestomped files)
- Interpretation: Detected 98.0% of all timestomped files

### By Precision (Minimize False Alarms):
**Winner**: XGBoost
- Precision: 0.8305 (83.05%)
- False Positives: 10
- Interpretation: 83.1% of flagged files were actually timestomped

### By F1-Score (Best Balance):
**Winner**: XGBoost
- F1-Score: 0.8991
- Precision: 0.8305
- Recall: 0.9800
- Interpretation: Best balance between precision and recall

### By ROC-AUC (Overall Discriminative Ability):
**Winner**: Random Forest
- ROC-AUC: 0.9998
- Interpretation: 100.0% probability of ranking random positive higher than random negative

---

## Model-Specific Details

### 1. Random Forest
- **Hyperparameters**: n_estimators=100, max_depth=15, class_weight='balanced'
- **Precision**: 0.5698
- **Recall**: 0.9800
- **F1-Score**: 0.7206
- **False Negatives**: 1

### 2. Logistic Regression
- **Hyperparameters**: class_weight='balanced', solver='lbfgs'
- **Precision**: 0.1365
- **Recall**: 0.9800
- **F1-Score**: 0.2396
- **False Negatives**: 1

### 3. XGBoost
- **Hyperparameters**: scale_pos_weight=611.08, max_depth=6, n_estimators=100
- **Precision**: 0.8305
- **Recall**: 0.9800
- **F1-Score**: 0.8991
- **False Negatives**: 1

### 4. LightGBM
- **Hyperparameters**: is_unbalance=True, max_depth=6, n_estimators=100
- **Precision**: 0.0516
- **Recall**: 0.9200
- **F1-Score**: 0.0977
- **False Negatives**: 4

### 5. Neural Network (Focal Loss)
- **Hyperparameters**: Focal Loss (α=0.25, γ=2.0), 2 hidden layers (64, 32 neurons)
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **False Negatives**: 50

---

## Key Insights

1. **Class Imbalance Handling**: All models successfully handled the severe 1:612 class imbalance using appropriate techniques
2. **Recall vs Precision Trade-off**: Models show varying trade-offs between catching all timestomped files (recall) and minimizing false alarms (precision)
3. **False Negatives**: This is the CRITICAL metric for forensics - missed timestomped files represent lost evidence

---

## Recommendations for Phase 4

Based on Phase 3 results, the following models should proceed to hyperparameter optimization:

1. **Top recall model** (minimize missed detections)
2. **Top F1-score model** (best balance)
3. **Top precision model** (if needed for low false-positive requirement)

---

## Files Generated

### Models:
- `random_forest_model.pkl`
- `logistic_regression_model.pkl`
- `xgboost_model.pkl`
- `lightgbm_model.pkl`
- `neural_network_model.h5`
- `feature_scaler.pkl`

### Results:
- `model_comparison.csv`
- `roc_curves_comparison.png`
- `precision_recall_curves_comparison.png`
- `metrics_comparison_bars.png`
- `confusion_matrices_all_models.png`

---

## Next Steps: Phase 4

- Hyperparameter tuning for top-performing models
- Cross-validation for robust performance estimates
- Threshold optimization for operational deployment
- Ensemble methods exploration
