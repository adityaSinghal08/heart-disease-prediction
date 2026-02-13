# preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Production-style preprocessor with hardcoded feature type registry.

    - Removes duplicates
    - Handles missing values
    - Encodes nominal categorical variables
    - Keeps binary as 0/1
    - Optionally scales numeric features
    - Works even if only subset of features are provided
    """

    def __init__(self, scaling: bool = False):
        self.scaling = scaling

        # ------------------------------
        # HARD-CODED FEATURE REGISTRY
        # ------------------------------

        self.binary_features = [
            "Sex",
            "FBS over 120",
            "Exercise angina",
            "Any_vessel_disease",
            "Severe_vessel_disease",
            "Abnormal_thallium",
            "Exercise_ischemia_flag",
            "Abnormal_EKG_flag",
            "Typical_angina_flag",
            "Low_HR_response_flag",
            "High_BP_flag",
            "High_Chol_flag",
            "FBS_over_120_flag"
        ]

        self.nominal_features = [
            "Chest pain type",
            "EKG results",
            "Thallium"
        ]

        self.ordinal_features = [
            "Slope of ST",
            "ST_severity_level",
            "Chest_pain_risk_score",
            "Age_group"
        ]

        self.numeric_features = [
            "Age",
            "BP",
            "Cholesterol",
            "Max HR",
            "ST depression",
            "Heart_rate_reserve",
            "ST_depression_normalized",
            "Age_squared",
            "BP_Chol_ratio",
            "Age_Sex_interaction",
            "Number of vessels fluro",
            "Metabolic_risk_score",
            "Exercise_risk_score"
        ]

        self.encoder = None
        self.scaler = None

        # These will be populated dynamically based on available columns
        self.active_binary = []
        self.active_nominal = []
        self.active_ordinal = []
        self.active_numeric = []

    # ------------------------------

    def fit(self, X, y=None):
        X = X.copy().drop_duplicates()

        # Identify which predefined columns are present
        self.active_binary = [c for c in self.binary_features if c in X.columns]
        self.active_nominal = [c for c in self.nominal_features if c in X.columns]
        self.active_ordinal = [c for c in self.ordinal_features if c in X.columns]
        self.active_numeric = [c for c in self.numeric_features if c in X.columns]

        # Fit encoder for nominal features
        if len(self.active_nominal) > 0:
            self.encoder = OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore"
            )
            self.encoder.fit(X[self.active_nominal])

        # Fit scaler for numeric features
        if self.scaling and len(self.active_numeric) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.active_numeric])

        return self

    # ------------------------------

    def transform(self, X):
        X = X.copy().drop_duplicates()

        output_df = pd.DataFrame(index=X.index)

        # ------------------------------
        # Missing value handling
        # ------------------------------

        # Numeric → median
        for col in self.active_numeric + self.active_ordinal:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(X[col].median())

        # Binary + nominal → mode
        for col in self.active_binary + self.active_nominal:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mode()[0])

        # ------------------------------
        # Binary features (keep as-is)
        # ------------------------------

        for col in self.active_binary:
            output_df[col] = X[col].astype(int)

        # ------------------------------
        # Ordinal features (keep numeric)
        # ------------------------------

        for col in self.active_ordinal:
            output_df[col] = X[col].astype(float)

        # ------------------------------
        # Numeric features
        # ------------------------------

        numeric_df = pd.DataFrame(index=X.index)

        for col in self.active_numeric:
            numeric_df[col] = X[col]

        if self.scaling and self.scaler is not None:
            numeric_df[self.active_numeric] = self.scaler.transform(
                numeric_df[self.active_numeric]
            )

        output_df = pd.concat([output_df, numeric_df], axis=1)

        # ------------------------------
        # Nominal OneHot encoding
        # ------------------------------

        if self.encoder is not None:
            encoded_array = self.encoder.transform(X[self.active_nominal])
            encoded_cols = self.encoder.get_feature_names_out(self.active_nominal)

            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoded_cols,
                index=X.index
            )

            output_df = pd.concat([output_df, encoded_df], axis=1)

        return output_df
