# Project 2: Supervised Learning — Fraud Detection Pipeline

**Notebook:** `fraud_detection_pipeline.ipynb`

## Goal

Build and tune a classification model to identify fraudulent transactions in a highly imbalanced dataset.

## Dataset

**Configured path:** `data/creditcard.csv`, target column `Class`.

> **Status: not yet connected to real data.** As of the current notebook, `USE_SYNTHETIC_PLACEHOLDER = True`, so all results below were generated on a synthetic dataset (5,000 rows, ~1.5% positive class, via `sklearn.datasets.make_classification`) — this validates that the pipeline runs correctly, but the numbers below are **not real fraud detection results**. Before submission: download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, place it at `data/creditcard.csv`, and set `USE_SYNTHETIC_PLACEHOLDER = False`.

## Approach

**1. Class imbalance handling — SMOTE**
Generates synthetic minority-class examples rather than duplicating existing ones.

**2. Train/test split before resampling**
`imblearn.pipeline.Pipeline` (not scikit-learn's) is used so SMOTE only ever touches the training fold, never the test set — avoiding data leakage that would make evaluation metrics artificially optimistic.

**3. Two models, tuned with GridSearchCV**
Both wrapped in a full pipeline (scaler + SMOTE + classifier), tuned on ROC-AUC across 5-fold cross-validation.

**4. Evaluation on Precision, Recall, ROC-AUC — not Accuracy**
With a rare positive class, a model predicting "not fraud" every time would score high accuracy while catching nothing.

## Results (on synthetic placeholder data — see caveat above)

| Model | Best Params | CV ROC-AUC | Test Precision | Test Recall | Test ROC-AUC |
|-------|-------------|-----------|-----------------|-------------|--------------|
| Logistic Regression | `C=0.1` | 0.6717 | 0.032 | 0.667 | 0.726 |
| Random Forest | `n_estimators=100, max_depth=None` | 0.7141 | 0.188 | 0.200 | 0.832 |

Even on placeholder data, this illustrates the precision/recall tradeoff the project is built around: Logistic Regression catches more of the (synthetic) fraud cases (67% recall) but at the cost of far more false alarms (3% precision), while Random Forest is more conservative — fewer false alarms, but also catching less. Random Forest's higher ROC-AUC (0.83 vs 0.73) suggests it separates the classes better overall on this data. These specific numbers will change once run against real transaction data.

## Tech Stack

pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib

## Files

| File | Description |
|------|-------------|
| `fraud_detection_pipeline.ipynb` | Full pipeline: load → split → SMOTE + model pipelines → tuning → evaluation |

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
jupyter notebook fraud_detection_pipeline.ipynb
```

**Before submission:** update `CSV_PATH` and set `USE_SYNTHETIC_PLACEHOLDER = False` once the real dataset is downloaded.
