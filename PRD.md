# Product Requirements Document (PRD)

## Project: CrediShield AI™ - Enterprise Credit Risk Classification & Explainable AI Decision System

---

### Document Information
- **Project Title:** CrediShield AI™ - Autonomous Credit Risk Prediction & XAI Decision Platform
- **Project Version:** 1.0.0
- **Local Host Server Endpoint:** `http://localhost:8899/credishield`
- **Status:** Approved / In Implementation
- **Author:** Advanced Agentic AI Engineering
- **Target Audience:** Risk Analytics, Credit Policy Committee, Machine Learning Engineers, Regulatory Compliance Officers

---

## 1. Executive Summary & Product Vision
The objective of this project is to develop an end-to-end Machine Learning and Explainable AI (XAI) system capable of predicting borrower loan default risk. In credit lending, approving an applicant who ultimately defaults carries a catastrophic financial penalty (loss of principal balance), whereas rejecting a creditworthy borrower results in an opportunity cost of foregone interest. Standard off-the-shelf machine learning pipelines frequently optimize for raw accuracy, which is dangerously misleading in credit default scenarios characterized by 75–80% non-defaulters.

This system delivers:
1. **Cost-Optimal Default Classification:** Utilizing imbalanced learning techniques (class weighting, SMOTE, and strategic threshold optimization) to maximize defaulter recall while controlling false positive rejection rates.
2. **Transparent Decisioning (Explainable AI):** Providing mathematically grounded global and local feature attributions via SHAP (SHapley Additive exPlanations) to fulfill regulatory transparency requirements and generate clear adverse action notices.
3. **Demographic Fairness Auditing:** Quantifying prediction disparities across sensitive attributes (sex, marital status, age groups) to maintain equitable lending practices.
4. **Production-Ready Artifacts:** Delivering a serialized pipeline (`final_pipeline.joblib`), a model registry ledger (`metadata.json`), an interactive Streamlit risk assessment application, and a continuous monitoring framework utilizing the Population Stability Index (PSI).

---

## 2. Business Context & Problem Statement

### 2.1 The Credit Asymmetry Dilemma
Retail credit operations operate under fundamental asymmetric loss functions:
- **False Negative (Type II Error):** Lending institution classifies a future defaulter as "Safe" and extends credit. Cost = loss of outstanding loan principal minus recovery ($C_{FN} \approx 5 \times$).
- **False Positive (Type I Error):** Lending institution flags a creditworthy applicant as "Risky" and declines or routes to high-friction manual underwriting. Cost = lost origination fees and customer lifetime value ($C_{FP} \approx 1 \times$).

### 2.2 The Accuracy Paradox
In a portfolio with a 22.1% historical default rate, a naive majority-class classifier that blindly approves 100% of applicants achieves **77.9% accuracy** while capturing **0% of defaulters**, causing catastrophic credit losses. The product requirement mandates measuring performance via **Defaulter Recall, Precision, PR-AUC, ROC-AUC, and the Kolmogorov-Smirnov (KS) statistic**, explicitly deprecating raw accuracy as a primary KPI.

### 2.3 Regulatory and Governance Mandates
Under financial regulatory frameworks (e.g., Fair Credit Reporting Act, Equal Credit Opportunity Act, and Federal Reserve SR 11-7 Model Risk Management guidelines), lenders must:
1. Avoid arbitrary or black-box denial of credit.
2. Provide specific, verifiable adverse action reasons for decline.
3. Ensure models do not propagate demographic bias against protected classes.

---

## 3. Goals and Non-Goals

### 3.1 Primary Goals
- Ingest and rigorously audit the 30,000-record UCI Credit Card Default dataset without data leakage.
- Address severe class imbalance using stratified splitting, class re-weighting, and pipeline-encapsulated resampling (SMOTE / Random Under-Sampling).
- Benchmark 5+ distinct model families: Dummy Baseline, Regularized Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
- Perform Bayesian/Randomized hyperparameter optimization focused on Area Under the Precision-Recall Curve (PR-AUC).
- Calibrate predicted default probabilities using Platt Scaling / Isotonic Regression to ensure calibrated confidence intervals.
- Formulate an explicit business cost matrix ($C_{FN}:C_{FP} = 5:1$) to establish the optimal empirical decision threshold.
- Implement comprehensive global feature importance and granular local applicant explanations using TreeSHAP and KernelSHAP.
- Conduct a fairness audit evaluating Approval Rate, Disparate Impact, and Equalized Odds across demographic slices.
- Package the end-to-end pipeline into an interactive Streamlit UI and implement Population Stability Index (PSI) drift monitoring.

### 3.2 Non-Goals
- Modeling Loss Given Default (LGD) or Exposure at Default (EAD); this system focuses strictly on Probability of Default (PD).
- Implementing multi-tenant production microservices with live banking core integrations.
- Serving as a legal automated lending authority without mandatory human underwriter oversight.

---

## 4. User Personas & Use Cases

| Persona | Core Need | Key Feature / Output |
|---|---|---|
| **Credit Underwriter** | Rapid risk assessment and clear reasons for decline | Interactive scoring dashboard with risk tiering (Low/Medium/High) and local SHAP waterfall plots |
| **Risk Committee / CRO** | Portfolio-level risk metrics, loss minimization, cost trade-offs | Cost vs. Threshold optimization curves, calibration plots, PR-AUC benchmarks |
| **Compliance & Fair Lending Officer** | Audit trails and assurance of non-discriminatory outcomes | Demographic fairness disparity reports (Sex, Age, Education, Marriage) |
| **MLOps / Data Scientist** | Reproducibility, pipeline integrity, data drift detection | Serialized pipeline, Model Registry `metadata.json`, and PSI drift monitoring scripts |

---

## 5. Data Architecture & Schema Specifications

### 5.1 Dataset Overview
- **Source:** UCI Machine Learning Repository (`UCI_Credit_Card.csv`)
- **Sample Size:** 30,000 credit card accounts (Taiwan, April–September 2005)
- **Target Variable:** `default.payment.next.month` (Binary: `1` = Default, `0` = Non-Default)
- **Feature Count:** 24 input features + 1 target

### 5.2 Feature Catalog & Data Dictionaries

| Column Name | Type | Description | Permitted Values / Ranges |
|---|---|---|---|
| `ID` | Identifier | Account identifier | 1 to 30,000 (Exclude from modeling) |
| `LIMIT_BAL` | Numeric (Continuous) | Amount of credit granted (NT dollars) | > 0 |
| `SEX` | Categorical | Gender | 1 = Male, 2 = Female |
| `EDUCATION` | Categorical | Education level | 1 = Graduate, 2 = University, 3 = High School, 4 = Others (0, 5, 6 mapped to 4) |
| `MARRIAGE` | Categorical | Marital status | 1 = Married, 2 = Single, 3 = Others (0 mapped to 3) |
| `AGE` | Numeric (Integer) | Age in years | 21 to 79 |
| `PAY_0` to `PAY_6` | Numeric (Ordinal) | Repayment status from Sept (`PAY_0`) to April (`PAY_6`) | -2 = No consumption, -1 = Paid in full, 0 = Revolving credit, 1..8 = Months delayed |
| `BILL_AMT1` to `BILL_AMT6` | Numeric (Continuous) | Bill statement amount (NT dollars) from Sept (`BILL_AMT1`) to April (`BILL_AMT6`) | Real numbers (can be negative due to overpayment) |
| `PAY_AMT1` to `PAY_AMT6` | Numeric (Continuous) | Amount of previous payment (NT dollars) from Sept (`PAY_AMT1`) to April (`PAY_AMT6`) | $\ge 0$ |
| `default.payment.next.month` | Binary Target | Default status in October 2005 | 0 = No default, 1 = Default |

### 5.3 Data Quality & Hygiene Constraints
1. **Leakage Prevention:** No post-decision operational data used. All features reflect history up to the decision threshold.
2. **Category Normalization:** Education codes {0, 4, 5, 6} consolidated into code `4` ("Others"). Marriage code {0} consolidated into code `3` ("Others").
3. **Scaling & Encodings:** Continuous numerical features scaled using `RobustScaler` or `StandardScaler`. Categorical features one-hot encoded or treated with ordinal encoding. All transformation parameters learned exclusively on training folds.

---

## 6. Feature Engineering Blueprint

To capture behavioral credit risk patterns, domain-specific features are engineered:
1. **Credit Utilization Rate (`utilization_ratio`):**
   $$\text{utilization\_ratio} = \frac{\text{BILL\_AMT1}}{\text{LIMIT\_BAL} + 1e-5}$$
2. **Payment-to-Bill Ratio (`pay_to_bill_ratio`):**
   $$\text{pay\_to\_bill\_ratio} = \frac{\sum_{i=1}^{6} \text{PAY\_AMT}_i}{\sum_{i=1}^{6} \max(\text{BILL\_AMT}_i, 0) + 1e-5}$$
3. **Delinquency Severity Indices:**
   - `max_delay`: Maximum delay across all repayment months ($\max(\text{PAY\_0}, \dots, \text{PAY\_6})$).
   - `delay_count`: Total count of months where payment was delayed ($\ge 1$).
   - `recent_delay_flag`: Binary indicator if `PAY_0 > 0` (most recent month delinquent).
4. **Financial Trajectory (Bill Amount Trend):**
   $$\text{bill_trend} = \text{BILL\_AMT1} - \text{BILL\_AMT6}$$
5. **Demographic Stratification:**
   - `age_group`: Categorization into `<25`, `25-34`, `35-49`, `50+`.

---

## 7. Machine Learning Modeling Strategy

```
                          [Raw Data Ingestion]
                                   │
                     [Stratified 80/20 Train-Test Split]
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
        [Training Set]                           [Test Set (Held-Out)]
               │                                       │
      [Feature Engineering]                            │ (Never Touched During Training)
               │                                       │
     [ImbLearn Cross-Validation Pipeline]              │
     ├─ Preprocessor (Impute + Scale + Encode)        │
     ├─ Imbalance Strategy (Weights / SMOTE / RUS)     │
     └─ Candidate Estimator                            │
               │                                       │
    [Candidate Models Benchmarking]                    │
    ├─ Baseline: Dummy (Most Frequent)                 │
    ├─ Baseline: Logistic Regression (L2)              │
    ├─ Decision Tree Classifier                        │
    ├─ Random Forest Classifier                        │
    ├─ Gradient Boosting Classifier                    │
    └─ XGBoost Classifier                              │
               │                                       │
    [Hyperparameter Tuning & PR-AUC Optimization]      │
               │                                       │
    [Probability Calibration (Platt / Isotonic)]       │
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
                       [Test Set Model Evaluation]
                                   │
              [Cost-Matrix Threshold Optimization (5:1)]
                                   │
               [Explainability: Global & Local SHAP]
                                   │
                    [Fairness & Disparity Audit]
                                   │
             [Artifact Serialization: models/*.joblib]
```

---

## 8. Evaluation Framework & Business Cost Optimization

### 8.1 Primary Evaluation Metrics
1. **Recall on Default Class ($y=1$):** Measures the proportion of actual defaulters detected:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
2. **Precision on Default Class ($y=1$):** Measures the accuracy of flagged applicants:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **PR-AUC (Average Precision):** Preferred ranking metric over ROC-AUC for imbalanced credit default scenarios.
4. **ROC-AUC:** Measures global discrimination across all operating thresholds.
5. **Kolmogorov-Smirnov (KS) Statistic:** Maximum separation between cumulative distributions of defaulters and non-defaulters:
   $$KS = \max |F_{default}(s) - F_{non-default}(s)|$$
6. **Brier Score:** Evaluates mean squared error between predicted probabilities and binary outcomes:
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

### 8.2 Cost-Utility Function & Decision Thresholding
Given $C_{FN} = \$500$ (loss per missed default) and $C_{FP} = \$100$ (loss per rejected good customer):
$$\text{Total Cost}(\theta) = 5 \times FN(\theta) + 1 \times FP(\theta)$$
The system evaluates all candidate thresholds $\theta \in [0.01, 0.99]$ to determine:
$$\theta^* = \arg\min_{\theta} \text{Total Cost}(\theta)$$

---

## 9. Explainable AI (XAI) & Fairness Specifications

### 9.1 Explainability Requirements
- **Global Interpretability:**
  - Feature importance rankings (Random Forest / XGBoost MDI & Permutation Importance).
  - SHAP summary beeswarm plot demonstrating directionality and magnitude of feature impacts.
- **Local Interpretability:**
  - Waterfall / Force plots for individual predictions.
  - Automatic generation of the Top 3 adverse risk factors formatted in compliant plain English.

### 9.2 Fairness & Bias Auditing
The system systematically audits outcomes across demographic protected categories:
- **Sex:** Male ($1$) vs. Female ($2$).
- **Marital Status:** Married ($1$) vs. Single ($2$).
- **Age Bands:** Young adults ($<30$) vs. Mature borrowers ($30+$).
- **Metrics Tracked:**
  - Approval Rate / Adverse Impact Ratio (AIR).
  - False Positive Rate (FPR) parity.
  - Defaulter Recall (True Positive Rate) parity.

---

## 10. Operational Architecture & Deployment Plan

### 10.1 Artifact Deliverables
1. `PRD.md`: This comprehensive specification document.
2. `AGENTS.md`: Operational behavioral constraints.
3. `loan_default_classification.ipynb`: Complete self-contained notebook executing stages 01 through 05.
4. `models/final_pipeline.joblib`: Complete trained pipeline with preprocessors, feature transformations, and calibrated classifier.
5. `models/metadata.json`: Model version, training parameters, evaluation metrics, and optimal decision threshold.
6. `app.py`: Streamlit dashboard for real-time inference and SHAP local attribution.
7. `monitoring.py`: Script implementing Population Stability Index (PSI) drift detection.

### 10.2 Continuous Monitoring & Drift Alerting
- Feature distribution shift tracked via Population Stability Index (PSI):
  $$PSI = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left( \frac{\text{Actual}\%}{\text{Expected}\%} \right)$$
- **Alert Levels:**
  - $PSI < 0.10$: Stable; no action required.
  - $0.10 \le PSI < 0.25$: Moderate drift; log warning and queue feature review.
  - $PSI \ge 0.25$: Significant distribution shift; trigger automated retraining workflow.

---

## 11. Acceptance Criteria Checklist
- [x] Full dataset loaded from local path without modifications to original file.
- [x] Clear demonstration of the Accuracy Trap (dummy classifier baseline).
- [x] Rigorous stratified 80/20 train/test split.
- [x] Resampling and transformation strictly confined to training folds via `imblearn` pipeline.
- [x] Comparative evaluation of at least 5 classifiers and 4 imbalance treatments.
- [x] Formal cost minimization threshold selection.
- [x] Global and local SHAP explanations implemented.
- [x] Demographic fairness audit across Sex, Age, and Marriage.
- [x] Model serialized to `models/final_pipeline.joblib` and metadata registered in `models/metadata.json`.
- [x] Streamlit deployment application (`app.py`) provided for interactive evaluation.
