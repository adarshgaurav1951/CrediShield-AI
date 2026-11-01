# Agent Rules

- **Project:** CrediShield AI™ (Enterprise Loan Default Risk & Explainable Intelligence System)
- **Local Host Server Endpoint:** `http://localhost:8899/credishield`
- **Primary Deliverables:** 
  - `PRD.md`: Formal Product Requirements Document detailing objectives, specifications, data schema, metrics, modeling strategy, fairness, and deployment architecture.
  - `loan_default_classification.ipynb`: Complete, executable Jupyter Notebook covering Stages 01 to 05 across 12 well-documented sections.
  - `models/final_pipeline.joblib`: Serialized end-to-end pipeline (preprocessing + optimal classifier).
  - `models/metadata.json`: Model registry with metadata, performance benchmarks, and decision threshold.
  - `app.py`: Streamlit scoring dashboard (Stage 06 Deployment) running at `http://localhost:8899/credishield` for interactive default probability prediction and local SHAP explanations.
  - `monitoring.py`: Stage 07 drift detection and Population Stability Index (PSI) alerting.
- **Constraints & Best Practices:**
  - Original data is in `UCI_Credit_Card.csv` and `data/raw/credit_default.csv` (never modify original CSV).
  - Always split data using `stratify=y` before any resampling to prevent synthetic data leakage.
  - Fit imputers, scalers, encoders, and resamplers strictly on training folds (using `imblearn.pipeline.Pipeline`).
  - Never report accuracy alone. Always evaluate with Recall, Precision, F1-Score, ROC-AUC, PR-AUC, and KS-Statistic.
  - Connect modeling decisions directly to business cost matrix ($C(FN) = 5 \times C(FP)$).
  - Ensure probability calibration is evaluated via Brier Score and reliability diagrams.
  - Include both global (SHAP summary, feature importances) and local (individual applicant waterfall plots) explainability.
  - Audit fairness across demographic subgroups (Sex, Education, Marriage, Age).
