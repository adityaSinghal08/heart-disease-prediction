# Feature Engineering Guide – Heart Disease Prediction

## Purpose of This Document

This document explains **what feature engineering we perform**, **why each feature exists**, and **how to interpret it**. It is intended for:

* Future contributors reading the codebase
* Reviewers evaluating modeling decisions
* Anyone trying to understand *how raw clinical variables are transformed into predictive signals*

The goal is **performance + interpretability**, not feature inflation.

---

## Design Philosophy

We follow four guiding principles:

1. **Clinical grounding** – Every feature reflects known cardiovascular risk factors
2. **Signal aggregation** – Combine weak signals into stronger composite indicators
3. **Non-linearity awareness** – Many medical risks do not scale linearly
4. **Model-agnostic usefulness** – Features should help tree models *and* linear models

All engineered features are grouped into **three importance levels** based on their expected contribution to model performance.

---

## Feature Importance Levels

### Level 1 – High Impact (Core Predictive Signals)

These features are most likely to improve model performance and frequently appear among top SHAP contributors.

#### 1. Any_vessel_disease

**What:** Indicates presence of coronary vessel blockage
**Why:** Direct anatomical evidence of disease

```python
(df["Number of vessels fluro"] > 0).astype(int)
```

---

#### 2. Severe_vessel_disease

**What:** Flags multiple affected vessels (≥2)
**Why:** Disease severity matters more than presence alone

```python
(df["Number of vessels fluro"] >= 2).astype(int)
```

---

#### 3. Abnormal_thallium

**What:** Non-normal thallium scan
**Why:** Indicates myocardial perfusion defects

```python
(df["Thallium"] != 3).astype(int)
```

---

#### 4. Exercise_ischemia_flag

**What:** Exercise angina combined with ST depression
**Why:** Captures stress-induced ischemia

```python
((df["Exercise angina"] == 1) & (df["ST depression"] > 1)).astype(int)
```

---

#### 5. Heart_rate_reserve

**What:** Difference between achieved and predicted max HR
**Why:** Measures cardiac stress tolerance

```python
df["Max HR"] - (220 - df["Age"])
```

---

#### 6. ST_depression_normalized

**What:** ST depression scaled by heart rate
**Why:** Same ST drop at different HRs implies different risk

```python
df["ST depression"] / df["Max HR"]
```

---

#### 7. ST_severity_level

**What:** Discretized ST depression severity
**Why:** Captures threshold effects

```python
pd.cut(df["ST depression"], [-1, 0, 1, 2, float("inf")], labels=[0,1,2,3])
```

---

#### 8. Global_heart_risk_index

**What:** Aggregate count of major risk factors
**Why:** Strong signal compression for models

```python
(
 (df["BP"] >= 140).astype(int) +
 (df["Cholesterol"] >= 240).astype(int) +
 (df["FBS over 120"] == 1).astype(int) +
 (df["Exercise angina"] == 1).astype(int) +
 (df["ST depression"] > 1).astype(int) +
 (df["EKG results"] != 0).astype(int) +
 (df["Number of vessels fluro"] > 0).astype(int)
)
```

---

#### 9. Metabolic_risk_score

**What:** Combined metabolic syndrome indicator
**Why:** Captures vascular stress from metabolic causes

```python
(
 (df["BP"] >= 140).astype(int) +
 (df["Cholesterol"] >= 240).astype(int) +
 (df["FBS over 120"] == 1).astype(int)
)
```

---

#### 10. Age_squared

**What:** Non-linear age risk
**Why:** Cardiac risk accelerates with age

```python
df["Age"] ** 2
```

---

## Level 2 – Moderate Impact (Context & Refinement)

These features add nuance and improve performance when combined with Level 1 features.

#### Abnormal_EKG_flag

```python
(df["EKG results"] != 0).astype(int)
```

#### Typical_angina_flag

```python
df["Chest pain type"].isin([3,4]).astype(int)
```

#### Chest_pain_risk_score

```python
{1:0, 2:1, 3:2, 4:3}[df["Chest pain type"]]
```

#### Low_HR_response_flag

```python
(df["Max HR"] < 0.85 * (220 - df["Age"])).astype(int)
```

#### Exercise_risk_score

```python
df["Exercise angina"] + (df["ST depression"] > 0).astype(int)
```

#### Age_Sex_interaction

```python
df["Age"] * df["Sex"]
```

#### BP_Chol_ratio

```python
df["BP"] / df["Cholesterol"]
```

#### High_BP_flag

```python
(df["BP"] >= 140).astype(int)
```

#### High_Chol_flag

```python
(df["Cholesterol"] >= 240).astype(int)
```

---

## Level 3 – Low Impact (Interpretability / Redundancy)

These features rarely improve metrics but help human understanding.

#### Age_group

```python
pd.cut(df["Age"], [0,40,50,60,100], labels=[0,1,2,3])
```

---

## Modeling Recommendations

* **Baseline model:** Original features + Level 1
* **Optimized model:** Level 1 + Level 2
* **Explainability-focused:** All levels with regularization

---

## Final Notes

Feature engineering here is **intentional, not exhaustive**.
If a feature does not add predictive or explanatory value, it should be removed.

Always validate additions using:

* Feature ablation
* SHAP value consistency
* Cross-validated performance

---

*End of feature engineering guide. Please note that this guide and the feature engineering recommendations are generated using a LLM. You can always explore further to design perhaps better features that may help you achieve even better outcomes.*