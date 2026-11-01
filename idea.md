# Loan Default Risk - Classification: Project Idea

## 1. One-Line Idea
Design a credit risk prediction model to identify potential loan defaulters using classification algorithms, imbalanced data handling, and explainable AI.

## 2. Background and Business Context
Banks and lenders earn money when borrowers repay and lose money when they default. Deciding who to lend to is a classification problem: given what is known about an applicant or customer, will they default or not?

Two things make this problem different from a typical classification task:
1. **Defaulters are rare.** In most credit datasets only a small share of borrowers default (roughly 5 to 25 percent, depending on the dataset). A model that predicts "no default" for everyone can look 80 to 95 percent accurate while being useless.
2. **Mistakes have unequal costs.** Approving a loan to someone who defaults (a missed defaulter) usually costs far more than rejecting a good customer (a lost profit opportunity). The model and its decision threshold must reflect that.

A third requirement comes from regulation and trust: lenders must be able to explain why an application was flagged or declined. That is why explainable AI is part of the project, not an extra.

## 3. Problem Statement
Build a classification model that predicts whether a borrower will default on a loan or credit obligation, handle the class imbalance correctly, choose a decision threshold based on business cost, and explain the model's predictions both globally (which factors matter overall) and locally (why this specific applicant was flagged).

## 4. Goals
- Clean and prepare real credit data, avoiding any information that would not be available at decision time (data leakage)
- Quantify the class imbalance and show why accuracy is a misleading metric
- Train and compare several classifiers, starting from a simple baseline
- Apply and compare imbalance-handling techniques (class weights, oversampling, undersampling, threshold tuning)
- Evaluate with metrics suited to imbalanced data (recall, precision, F1, ROC-AUC, PR-AUC, KS statistic, confusion matrix)
- Choose a decision threshold using a business cost assumption
- Explain the model with feature importance, SHAP, and individual-applicant explanations
- Check the model for fairness concerns across sensitive groups
- Document limits clearly, including that the model is a learning project and not a real lending tool

## 5. Non-Goals
- Building a production credit scoring system or making real lending decisions
- Complying with specific banking regulations (Basel, IFRS 9, fair lending laws)
- Modeling loss given default or exposure at default (this project predicts only the probability of default)

## 6. What the Project Title Means
| Phrase | What you do in this project |
|---|---|
| Credit risk prediction model | Estimate the probability that a borrower defaults, and turn it into a decision (approve, review, decline) |
| Classification algorithms | Logistic Regression (baseline), Decision Tree, Random Forest, Gradient Boosting, and optionally XGBoost or LightGBM |
| Imbalanced data handling | Stratified splitting, class weights, SMOTE, undersampling, threshold tuning, and imbalance-aware metrics |
| Explainable AI | Coefficients and odds ratios, permutation importance, SHAP summary plots, and per-applicant explanations |

## 7. Key Questions to Answer
1. How imbalanced is the data, and what accuracy would a "predict no default" model get?
2. Which borrower characteristics are most strongly linked with default?
3. How does a baseline Logistic Regression compare with tree-based models?
4. Which imbalance-handling technique works best, and does it help every model?
5. What recall and precision can the best model achieve for defaulters?
6. What decision threshold minimizes cost under a stated cost assumption?
7. Which features push a specific applicant's risk up or down?
8. Are predictions noticeably different across groups such as sex, age band, or marital status?
9. Are the predicted probabilities well calibrated, so that "30 percent risk" really means about 30 percent?
