# feature_generator_level_1.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureGeneratorLevel1(BaseEstimator, TransformerMixin):
    """
    Generates high-impact structural and stress-related features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        df = pd.DataFrame(index=X.index)

        # Vessel features
        df["Any_vessel_disease"] = (X["Number of vessels fluro"] > 0).astype(int)
        df["Severe_vessel_disease"] = (X["Number of vessels fluro"] >= 2).astype(int)

        # Thallium
        df["Abnormal_thallium"] = (X["Thallium"] != 3).astype(int)

        # Exercise ischemia
        df["Exercise_ischemia_flag"] = (
            (X["Exercise angina"] == 1) &
            (X["ST depression"] > 1)
        ).astype(int)

        # ST severity
        df["ST_severity_level"] = pd.cut(
            X["ST depression"],
            bins=[-1, 0, 1, 2, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # ST normalized
        df["ST_depression_normalized"] = X["ST depression"] / X["Max HR"]

        # Exercise risk score
        df["Exercise_risk_score"] = (
            X["Exercise angina"] +
            (X["ST depression"] > 0).astype(int)
        )

        return df
