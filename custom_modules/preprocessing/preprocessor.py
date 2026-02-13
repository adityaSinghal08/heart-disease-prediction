# preprocessor.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Handles:
    - Duplicate removal
    - Missing value imputation
    - Type enforcement
    - Basic cleaning

    Does NOT generate features.
    """

    def __init__(self):
        self.numeric_columns = None

    def fit(self, X, y=None):
        X = X.copy()

        # Identify numeric columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        return self

    def transform(self, X):
        X = X.copy()

        # Remove duplicates
        X = X.drop_duplicates()

        # Ensure numeric columns are numeric
        for col in self.numeric_columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Impute numeric missing values with median
        for col in self.numeric_columns:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())

        # Impute categorical missing values with mode
        categorical_cols = X.columns.difference(self.numeric_columns)
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0])

        return X.reset_index(drop=True)