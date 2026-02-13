# feature_generator_level_2.py

import pandas as pd
import numpy as np


class FeatureGeneratorLevel2:
    """
    Generates contextual and interaction-based features.
    Returns only newly created features.
    """

    def generate(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=X.index)

        # Age-based features
        df["Age_squared"] = X["Age"] ** 2
        df["Age_Sex_interaction"] = X["Age"] * X["Sex"]

        # Heart rate features
        df["Heart_rate_reserve"] = X["Max HR"] - (220 - X["Age"])

        df["Low_HR_response_flag"] = (
            X["Max HR"] < 0.85 * (220 - X["Age"])
        ).astype(int)

        # Chest pain ordinal encoding
        df["Chest_pain_risk_score"] = X["Chest pain type"].map({
            1: 0,
            2: 1,
            3: 2,
            4: 3
        })

        # EKG abnormality
        df["Abnormal_EKG_flag"] = (X["EKG results"] != 0).astype(int)

        return df
