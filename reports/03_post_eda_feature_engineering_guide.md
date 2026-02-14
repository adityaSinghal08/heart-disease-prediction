# Post-EDA Feature Engineering Guide

## Purpose of This Document

Following exploratory data analysis, features are now grouped into three Feature Engineering (FE) classes based on observed predictive strength.

This guide explains:

* Why features are grouped into levels
* What each level represents
* How the modeling pipeline should use them

---

# Feature Engineering Levels

## FE Level 1 – High Signal Features

These features demonstrated strong separation, monotonic trends, or high correlation with the target.

### Includes:

* Exercise_ischemia_flag
* Severe_vessel_disease
* Any_vessel_disease
* Abnormal_thallium
* ST_severity_level
* Number of vessels fluro
* ST_depression_normalized (or ST_depression, not both in linear models)
* Exercise_risk_score

### Rationale

These features represent:

* Structural heart damage
* Stress-induced ischemia
* Clear ordinal severity patterns

They form the core predictive backbone of the model.

---

## FE Level 2 – Contextual & Interaction Features

These features provide additional nuance but overlap with Level 1 signals.

### Includes:

* Chest pain type / Chest_pain_risk_score
* Slope of ST
* Sex
* Age
* Age_squared (choose carefully for linear models)
* Heart_rate_reserve
* Low_HR_response_flag
* Age_Sex_interaction
* Abnormal_EKG_flag

### Rationale

These features:

* Add demographic or physiological context
* Improve nonlinear modeling
* Enhance interpretability

They strengthen performance but are not independently dominant.

---

## FE Level 3 – Background Risk Indicators

These features showed weak standalone predictive power.

### Includes:

* BP
* Cholesterol
* FBS over 120
* High_BP_flag
* High_Chol_flag
* Metabolic_risk_score
* BP_Chol_ratio
* Age_group

### Rationale

These variables:

* Represent long-term risk exposure
* Provide minor additive value
* May be useful in large datasets
* Are candidates for ablation if regularization is applied

---

# Modeling Strategy

Baseline Model:
→ Original features + FE Level 1

Optimized Model:
→ FE Level 1 + FE Level 2

Regularized / Interpretable Model:
→ Gradually add FE Level 3 and evaluate via ablation

---

# Key Modeling Notes

1. Avoid multicollinearity in linear models
2. Tree-based models tolerate correlated features
3. Always validate additions via feature ablation
4. Use SHAP to confirm contribution directionality

---

*End of Post-EDA Feature Engineering Guide.*