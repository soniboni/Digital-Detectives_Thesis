"""
Quick Test: Verify the trained model works on known timestomped data
This tests the model directly without the prototype notebooks
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

print("=" * 80)
print("TESTING MODEL ON TRAINING DATA")
print("=" * 80)

# Load the trained model
print("\n1. Loading trained model...")
model = joblib.load('data/processed/Phase 5 - V2 Hyperparameter Tuning/random_forest_tuned_v2.pkl')
print(f"   ✓ Model loaded: {type(model).__name__}")
print(f"   ✓ Expects {len(model.feature_names_in_)} features")

# Load Phase 3 final training data (already preprocessed)
print("\n2. Loading Phase 3 final training data...")
df = pd.read_csv('data/processed/Phase 3 - V2 Feature Selection/all_cases_combined_v2_phase3_final.csv',
                 low_memory=False)
print(f"   ✓ Loaded {len(df):,} records")
print(f"   ✓ Timestomped: {(df['timestomped'] == 1).sum():,}")

# Filter to case 11 (our test case)
print("\n3. Filtering to Case 11 (prototype test case)...")
case11 = df[df['case_id'] == 11].copy()
print(f"   ✓ Case 11: {len(case11):,} records")
print(f"   ✓ Timestomped in case 11: {(case11['timestomped'] == 1).sum()}")

# Get SetMACE rows specifically
setmace = case11[case11['filename'].str.contains('SetMACE_SI_MACE_Copy_Manipulation', na=False)]
print(f"\n4. SetMACE rows in case 11: {len(setmace)}")
if len(setmace) > 0:
    print(f"   Row indices: {setmace.index.tolist()}")
    print(f"   Timestomped labels: {setmace['timestomped'].tolist()}")

# Prepare features (same as training)
print("\n5. Preparing features...")
identifier_cols = ['case_id', 'eventtime', 'eventtime_dt', 'filename', 'filepath', 'merge_key']
target_col = 'timestomped'
feature_cols = [col for col in df.columns if col not in identifier_cols + [target_col]]

X_case11 = case11[feature_cols].copy()
y_case11 = case11[target_col].copy()

print(f"   ✓ Feature matrix: {X_case11.shape}")

# Verify feature alignment
model_features = set(model.feature_names_in_)
data_features = set(X_case11.columns)

missing = model_features - data_features
extra = data_features - model_features

if missing:
    print(f"   ⚠️  Missing {len(missing)} features")
if extra:
    print(f"   ⚠️  Extra {len(extra)} features")

# Align features with model
X_case11 = X_case11[model.feature_names_in_]
print(f"   ✓ Aligned features: {X_case11.shape}")

# Make predictions
print("\n6. Running predictions...")
predictions = model.predict(X_case11)
probabilities = model.predict_proba(X_case11)[:, 1]

# Add to dataframe
case11['predicted'] = predictions
case11['confidence'] = probabilities

# Results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nCase 11 Overall:")
print(f"  Total records: {len(case11)}")
print(f"  Actual timestomped: {(y_case11 == 1).sum()}")
print(f"  Predicted timestomped: {(predictions == 1).sum()}")

# SetMACE specific results
if len(setmace) > 0:
    print(f"\nSetMACE File Results:")
    setmace_results = case11[case11['filename'].str.contains('SetMACE_SI_MACE_Copy_Manipulation', na=False)]
    for idx, row in setmace_results.iterrows():
        print(f"  Row {idx}:")
        print(f"    Filename: {row['filename']}")
        print(f"    Actual label: {row['timestomped']}")
        print(f"    Predicted: {row['predicted']}")
        print(f"    Confidence: {row['confidence']:.2%}")
        print(f"    Result: {'✓ CORRECT' if row['predicted'] == row['timestomped'] else '✗ INCORRECT'}")

# Overall accuracy on case 11
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"\nCase 11 Metrics:")
print(f"  Accuracy:  {accuracy_score(y_case11, predictions):.2%}")
if (y_case11 == 1).sum() > 0:
    print(f"  Precision: {precision_score(y_case11, predictions, zero_division=0):.2%}")
    print(f"  Recall:    {recall_score(y_case11, predictions, zero_division=0):.2%}")
    print(f"  F1-Score:  {f1_score(y_case11, predictions, zero_division=0):.4f}")

# Show top confident timestomped predictions
print(f"\nTop 5 Highest Confidence Timestomped Predictions:")
top5 = case11.nlargest(5, 'confidence')[['filename', 'timestomped', 'predicted', 'confidence']]
print(top5.to_string(index=False))

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if len(setmace) > 0 and all(setmace_results['predicted'] == setmace_results['timestomped']):
    print("\n✓ SUCCESS: Model correctly predicts SetMACE file as timestomped!")
    print("  The model is working correctly on the training data.")
    print("  The prototype preprocessing must be the issue.")
elif len(setmace) > 0:
    print("\n✗ FAILURE: Model incorrectly predicts SetMACE file!")
    print("  This suggests a deeper model training issue.")
else:
    print("\n⚠️  WARNING: No SetMACE file found in case 11")

print("\n" + "=" * 80)
