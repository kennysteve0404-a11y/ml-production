# UCI Adult Income Prediction: Production Machine Learning Pipeline

An end-to-end Machine Learning and MLOps pipeline built to predict whether an individual's annual income exceeds $50,000 based on census demographic data.

This project demonstrates modular data engineering, statistical EDA, data-leakage prevention via scikit-learn/imblearn pipelines, model bias awareness, and systematic benchmarking across multiple model architectures.

---

## 📌 Problem Statement & Context

In financial services and credit decisioning systems, predicting income tier accurately is vital for credit scoring and pre-approval offers. However, misclassification carries asymmetric business costs:
* **False Positive ($FP$):** Approving a high-risk borrower whose income is actually $\le \$50\text{K}$. High financial loss (potential default).
* **False Negative ($FN$):** Failing to target a qualified borrower earning $> \$50\text{K}$. Low financial loss (missed marketing lead).

This project focuses on building a production-ready classifier optimized for high precision while strictly preventing training-serving skew and data leakage.

---

## 🏗️ Technical Architecture & Key Pipeline Decisions

### 1. Centralized Configuration & Schema (`CONFIG` & `SCHEMA`)
To prevent hardcoded magic numbers and enable seamless parameter tuning, the project uses centralized dictionaries:
* **`CONFIG`**: Manages test split ratios (`test_size=0.20`), random seeds (`random_state=42`), drop columns, and classification thresholds.
* **`SCHEMA`**: Strictly defines explicit numerical and categorical feature arrays passed into preprocessing pipelines.

### 2. Data Cleaning & Feature Engineering
Data cleaning is encapsulated inside `clean_and_engineer_data()`:
* **Whitespace & Missing Values:** Stripped surrounding whitespace and standardized `?` characters to explicit `NaN` values.
* **Redundant Column Dropping:** Dropped `fnlwgt` (sample weight) and `education` (redundant with `education_num`).
* **Skew Transformation:** Applied `np.log1p` to heavily right-skewed features (`capital_gain`, `capital_loss`).

### 3. AI Ethics & Demographic Fairness Strategy ⚖️
* **Protected Attribute Handling:** The `race` column is retained in `df_processed` for post-training fairness audits and disparate impact testing, but **explicitly omitted from `SCHEMA['categorical_features']`**.
* **Impact:** Guarantees that model split rules and regression weights do not directly learn on demographic categories, maintaining regulatory compliance while keeping validation data intact for auditability.

---

## 📊 Exploratory Data Analysis (EDA) Summary

1. **Target Class Imbalance:**
   * $\le \$50\text{K}$ (Class 0): **75.92%**
   * $> \$50\text{K}$ (Class 1): **24.08%**
   * *Takeaway:* Baseline accuracy of a naive dummy model is $~76\%$. ROC-AUC, Precision, Recall, and Macro F1-Score are used as primary evaluation metrics.

2. **Top Numerical Predictors (Correlation Heatmap Analysis):**
   * `education_num` ($r = 0.34$): Strongest continuous linear predictor.
   * `capital_gain_log` ($r = 0.29$): High capital gains correlate strongly with high-income status.
   * `hours_per_week` ($r = 0.23$) and `age` ($r = 0.23$): Moderate positive influence.

3. **High-Cardinaity Regional Feature Engineering:**
   * Grouped 41 native countries into macro-regions. Individuals from **Asia & Middle East** demonstrated a high-income rate of **29.75%** vs. **24.58%** for the **United States**, providing strong categorical variance while reducing sparse one-hot encoding dimensions.

---

## ⚙️ Data Leakage Prevention & Preprocessing Pipeline

To ensure statistical parameters (imputation modes, medians, scaling bounds) are calculated **strictly on the training split**, all transformers are wrapped in a scikit-learn `ColumnTransformer`:

* **Numeric Sub-pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `RobustScaler()`
* **Categorical Sub-pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

```
  ┌─────────────────────────────────┐
  │      Stratified Split (80/20)   │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │       ColumnTransformer         │
  │  - Numeric: Median Impute+Scale │
  │  - Categorical: Mode Impute+OHE │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Estimator / Resampler (Fit)   │
  └─────────────────────────────────┘
```

---

## 🧪 Model Experimentation & Benchmark Matrix

Experiments were executed using `imblearn.pipeline.Pipeline` to guarantee resampling (SMOTE) occurs strictly within training folds during fit operations:

| Experiment Name | Model / Sampling Technique | ROC-AUC | Class 1 Recall | Class 1 Precision | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`1_Baseline_Logistic`** | Standard Logistic Regression | **0.900** | 0.608 | **0.721** | **0.781** |
| **`2_SMOTE_Logistic`** | SMOTE Resampling + Logistic Regression | 0.899 | **0.843** | 0.558 | 0.765 |
| **`3_Balanced_RandomForest`** | Random Forest (`class_weight='balanced'`) | 0.893 | 0.754 | 0.614 | 0.779 |

### Operational Metric Trade-offs
* **SMOTE** shifted the decision boundary to aggressively capture high earners (Recall increased to **84.3%**), but triggered a high False Positive rate (Precision dropped to **55.8%**).
* **Baseline Logistic Regression** achieved the highest overall ranking capacity (**0.900 ROC-AUC**) and highest Precision (**0.721** at default 0.50 threshold), making it the primary candidate for probability threshold optimization.

---

## 🔄 Cross-Validation Strategy

To ensure model stability across sampling partitions, all pipelines undergo **Stratified 5-Fold Cross-Validation** via `sklearn.model_selection.cross_validate`. This validates that mean ROC-AUC and Precision scores remain consistent without fold-level variance.

---

## 📍 Current Project Status & Next Steps

- [x] Data Loading & Modular Cleaning Script
- [x] Ethical Attribute Management Strategy (`race` dropping)
- [x] Exploratory Data Analysis & Feature Correlation Heatmaps
- [x] Leak-Free Preprocessing Pipeline Architecture
- [x] Multi-Model Experimentation Suite (Baseline vs. SMOTE vs. Cost-Sensitive Tree Ensembles)
- [x] Stratified 5-Fold Cross-Validation Benchmarking
- [ ] **Active Phase:** Error Analysis (Deep-dive on High-Confidence False Positives)
- [ ] Probability Threshold Optimization ($p \ge 0.65$ for Credit Risk Precision)
- [ ] Model Packaging (`joblib` Serialization)
- [ ] FastAPI Endpoint Deployment & Containerization (Docker)

---

## 🚀 How to Run locally

### Prerequisites
* Python 3.9+
* Required packages: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `seaborn`, `matplotlib`

### Execution
```bash
# Clone repository
git clone https://github.com/<your-username>/adult-income-mlops.git
cd adult-income-mlops

# Run main pipeline / notebook execution
python main.py