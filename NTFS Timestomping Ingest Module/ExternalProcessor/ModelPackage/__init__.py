# -*- coding: utf-8 -*-

"""
ModelPackage for NTFS Timestomping Detection

Modules:
    - data_preprocessing: Temporal alignment and event grouping
    - feature_engineering: Feature creation
    - TrainedModels: Serialized trained ML models (to be populated)
"""

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer'
]