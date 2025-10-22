#!/usr/bin/env python3
"""Check demo results to see confidence scores for timestomped files"""

import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('full_pipeline_results/predictions.csv')

# Fill NaN values
df['is_timestomped'] = df['is_timestomped'].fillna(0)

# Get timestomped files
timestomped = df[df['is_timestomped'] == 1.0]

print(f"Ground truth timestomped files: {len(timestomped)}")

if len(timestomped) > 0:
    print(f"Max confidence: {timestomped['confidence'].max():.4f}")
    print(f"Mean confidence: {timestomped['confidence'].mean():.4f}")
    print(f"Min confidence: {timestomped['confidence'].min():.4f}")

    print(f"\nTop 10 highest confidence (actual timestomped files):")
    print(timestomped.nlargest(10, 'confidence')[['filename', 'confidence', 'prediction', 'risk_level', 'is_timestomped']])

    # Check how many were correctly identified
    correct = (timestomped['prediction'] == 1).sum()
    print(f"\nCorrectly identified as timestomped: {correct}/{len(timestomped)} ({correct/len(timestomped)*100:.1f}%)")
else:
    print("No timestomped files found in predictions!")