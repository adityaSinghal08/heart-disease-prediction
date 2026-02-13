# feature_generator_level_2.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureGeneratorLevel2(BaseEstimator, TransformerMixin):
    """
    Generates contextual and interaction-based features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        df = pd.DataFrame(index=X.index)

        # Age features
        df["Age_squared"] = X["Age"] ** 2
        df["Age_Sex_interaction"] = X["Age"] * X["Sex"]

        # Heart rate features
        df["Heart_rate_reserve"] = X["Max HR"] - (220 - X["Age"])
        df["Low_HR_response_flag"] = (
            X["Max HR"] < 0.85 * (220 - X["Age"])
        ).astype(int)

        # Chest pain encoding
        df["Chest_pain_risk_score"] = X["Chest pain type"].map({
            1: 0,
            2: 1,
            3: 2,
            4: 3
        })

        # EKG abnormal
        df["Abnormal_EKG_flag"] = (X["EKG results"] != 0).astype(int)

        return df
