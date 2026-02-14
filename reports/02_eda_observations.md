# EDA Observations

## Purpose of This Document

This document summarizes key observations derived from exploratory data analysis (EDA), including KDE plots, violin plots, correlation heatmaps, and categorical risk distributions.

The goal is to identify:

* Strong predictive signals
* Weak or redundant features
* Monotonic risk trends
* Multicollinearity risks

---

# 1. Strongest Predictors (Clear Risk Separation)

## Exercise_ischemia_flag

* 0 → ~37% heart disease
* 1 → ~92% heart disease

Near-diagnostic level separation. Extremely strong composite feature.

---

## Severe_vessel_disease

* 0 → ~39%
* 1 → ~90%

Strong severity gradient. Confirms anatomical burden strongly predicts outcome.

---

## Any_vessel_disease

* 0 → ~30%
* 1 → ~80%

Presence of vessel blockage strongly associated with disease.

---

## Abnormal_thallium

* Normal → ~20%
* Abnormal → ~70–81%

Large perfusion-based separation.

---

## ST_severity_level

Clear monotonic increase:

* Level 0 → ~27%
* Level 1 → ~41%
* Level 2 → ~71%
* Level 3 → ~88%

Excellent ordinal feature behavior.

---

## Chest pain type

* Type 1 → ~11%
* Type 4 → ~70%

Large risk escalation across types.

---

# 2. Strong but Secondary Predictors

* Exercise angina (31% → 81%)
* Slope of ST (26% → 72%)
* Sex (18% → 56%)
* Age_group (29% → 57%)
* Low_HR_response_flag (36% → 71%)

These show meaningful separation but overlap with stronger stress/anatomical features.

---

# 3. Moderate Predictors

* Abnormal_EKG_flag (34% → 56%)
* Exercise_risk_score (monotonic trend)
* Age and Age_squared (older skew in positive class)

---

# 4. Weak Predictors

* High_BP_flag (~44% → 46%)
* High_Chol_flag (~40% → 49%)
* FBS over 120 (~44% → 50%)

Metabolic thresholds are weak standalone predictors.

---

# 5. Correlation Observations

Strong correlations with target:

* Exercise_risk_score (~0.51)
* ST_depression_normalized (~0.45)
* Number of vessels fluro (~0.44)
* ST_depression (~0.43)

Multicollinearity detected:

* ST_depression ↔ ST_depression_normalized (~0.98)
* Age ↔ Age_squared (~1.00)
* Max HR ↔ Heart_rate_reserve (~0.91)
* BP_Chol_ratio ↔ Cholesterol (~-0.76)

Implication: avoid redundant pairs in linear models.

---

# 6. Global Insight

Structural and stress-induced features dominate prediction.
Metabolic indicators act as background risk rather than diagnostic drivers.
Composite engineered features show strong validation from EDA.

---

*End of EDA Observations Document.*