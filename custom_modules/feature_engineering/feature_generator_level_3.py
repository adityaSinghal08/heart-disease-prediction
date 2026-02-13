# feature_generator_level_3.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureGeneratorLevel3(BaseEstimator, TransformerMixin):
    """
    Generates background metabolic and risk-threshold features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        df = pd.DataFrame(index=X.index)

        # Threshold flags
        df["High_BP_flag"] = (X["BP"] >= 140).astype(int)
        df["High_Chol_flag"] = (X["Cholesterol"] >= 240).astype(int)
        df["FBS_over_120_flag"] = (X["FBS over 120"] == 1).astype(int)

        # Metabolic score
        df["Metabolic_risk_score"] = (
            df["High_BP_flag"] +
            df["High_Chol_flag"] +
            df["FBS_over_120_flag"]
        )

        # BP/Chol ratio
        df["BP_Chol_ratio"] = X["BP"] / X["Cholesterol"]

        # Age group
        df["Age_group"] = pd.cut(
            X["Age"],
            bins=[0, 40, 50, 60, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)

        return df
