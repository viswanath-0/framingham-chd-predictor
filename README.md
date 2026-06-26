# Framingham Heart Study — CHD Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat&logo=plotly&logoColor=white)

End-to-end binary classification pipeline predicting 10-year coronary heart disease (CHD) risk from patient health records using a Bayesian-tuned Decision Tree classifier.

**Live Demo:** [framingham-chd-viswanath.streamlit.app](https://framingham-chd-viswanath.streamlit.app)

---

## Project Overview

| Phase | Description |
|-------|-------------|
| EDA | Feature distributions split by CHD status, missing value analysis, correlation heatmap |
| Preprocessing | Median imputation (skewed features), Mode imputation (binary/categorical) — fit on training set only |
| Feature Selection | VarianceThreshold, SelectKBest (f_classif), Mutual Information, RFE, DT Feature Importances |
| ML Pipeline | 4 Decision Tree models trained and compared across 5 metrics |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV, Bayesian Optuna |
| Cross Validation | KFold-5, KFold-10, StratifiedKFold-5, LOOCV, LeavePOut(p=2) |
| Bias-Variance | sklearn learning_curve, mlxtend bias_variance_decomp |
| Web App | Dark-themed Streamlit app with risk gauge, clinical interpretation, and feature comparison |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Framingham Heart Study (public health dataset) |
| Rows | 4,240 patients |
| Features | 15 (demographics, smoking, medical history, clinical measurements) |
| Target | TenYearCHD — binary (0=No CHD, 1=CHD risk within 10 years) |
| Class balance | 85% negative / 15% positive — imbalanced |
| Missing values | glucose (9.1%), education (2.5%), BPMeds (1.3%), others minor |

---

## Machine Learning Pipeline

### Why ROC-AUC as primary metric?
With 85% negative class, a naive model always predicting "No CHD" achieves 85% accuracy. ROC-AUC measures true discrimination regardless of class ratio — the correct metric for this imbalanced problem.

### Preprocessing
- **Median imputation**: glucose, totChol, BMI, heartRate, cigsPerDay (right-skewed)
- **Mode imputation**: BPMeds, education (binary/ordinal)
- All imputers fit only on training data to prevent leakage

### Feature Selection (5 methods)
| Method | Type | Output |
|--------|------|--------|
| VarianceThreshold (0.1) | Filter | Remove near-zero variance |
| SelectKBest (f_classif) | Filter | Top 10 by ANOVA F-score |
| Mutual Information | Filter | Top 10 by information gain |
| RFE (DecisionTree) | Wrapper | 10 features by recursive elimination |
| DT Feature Importances | Embedded | Top 10 by impurity reduction |

RFE selected features are used for training — it directly optimises for DT classification performance.

### Models Trained & Compared

| Model | Notes |
|-------|-------|
| DT Baseline | Default DecisionTreeClassifier |
| DT GridSearchCV | Exhaustive search: criterion, max_depth, min_samples_split, min_samples_leaf |
| DT RandomizedSearchCV | 40 random combinations + max_features |
| **DT Bayesian (Optuna)** | **40 Bayesian trials — best model** |

### Evaluation Metrics
Accuracy, Precision, Recall, F1 Score, ROC-AUC

### Cross Validation (on best model)
- KFold (k=5) — ROC-AUC
- KFold (k=10) — ROC-AUC
- StratifiedKFold (k=5) — ROC-AUC (recommended for imbalanced)
- LOOCV — Accuracy
- LeavePOut (p=2) — Accuracy

### Why Decision Tree?
- Medically interpretable — splits can be explained to clinicians
- No scaling required — works on raw feature values
- Bayesian tuning finds optimal depth constraints that balance recall vs precision
- Clear rules: "if age > 55 AND sysBP > 140 → higher risk"

---

## Web Application

4 tabs with dark red accent theme:

| Tab | Contents |
|-----|----------|
| Overview | KPIs, target imbalance chart, full feature description table |
| EDA & Statistics | Age/BP/glucose/BMI/cholesterol distributions by CHD status, correlation heatmap, missing value chart |
| Model Performance | Pipeline visual, feature selection chart, all 4 models ranked, ROC curves, confusion matrix, CV chart, feature importances |
| CHD Risk Predictor | 15-feature patient form, risk percentage card with gauge, elevated risk flags, population comparison table |

---

## Project Structure

```
framingham-chd-predictor/
├── app.py                          # Streamlit web application
├── framingham.csv                  # Framingham Heart Study dataset
├── Framingham_DT_Pipeline.ipynb    # Full classification pipeline notebook
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml                 # Dark red theme
```

---

## Run Locally

```bash
git clone https://github.com/viswanath-0/framingham-chd-predictor.git
cd framingham-chd-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> First load: ~20 seconds for model training. Cached after that.

---

## Key Results

| Metric | Value |
|--------|-------|
| Best model | DT Bayesian (Optuna) |
| ROC-AUC | ~0.70 – 0.74 |
| Recall | ~0.35 – 0.50 |
| Stratified CV Mean AUC | ~0.68 – 0.72 |
| Top features | age, sysBP, glucose, totChol |

---

## Skills Demonstrated

`Python` `Scikit-learn` `Decision Tree` `Classification` `Class Imbalance` `ROC-AUC` `Confusion Matrix` `Feature Selection` `RFE` `Mutual Information` `SelectKBest` `Hyperparameter Tuning` `GridSearchCV` `RandomizedSearchCV` `Optuna` `StratifiedKFold` `LOOCV` `Bias-Variance` `Medical Data` `Streamlit` `Plotly` `Git`

---

## Author

**Viswanath** — Data Science Portfolio

[![GitHub](https://img.shields.io/badge/GitHub-viswanath--0-181717?style=flat&logo=github)](https://github.com/viswanath-0)
