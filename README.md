# DecodeLabs Data Science Internship — Project Series

A four-part data science project series completed as part of a remote internship at DecodeLabs, covering the core stages of a real-world ML workflow: data cleaning and feature engineering, supervised classification on imbalanced data, unsupervised customer segmentation, and NLP-based sentiment analysis.

## Projects

| # | Project | Notebook | Domain | Core Techniques |
|---|---------|----------|--------|-----------------|
| 1 | [Advanced EDA & Feature Engineering](./project1_eda) | `week_one.ipynb` | E-commerce orders | Missing data handling, IQR outlier detection, feature engineering |
| 2 | [Fraud Detection Pipeline](./project2_fraud_detection) | `fraud_detection_pipeline.ipynb` | Financial transactions | SMOTE, Logistic Regression, Random Forest, hyperparameter tuning |
| 3 | [Customer Segmentation](./project3_customer_segmentation) | `marketing_campaign.ipynb` | Retail marketing | PCA, K-Means clustering, Elbow Method, Silhouette Score |
| 4 | [NLP & Sentiment Analysis](./project4_sentiment_analysis) | `sentiment_analysis.ipynb` | Movie reviews | Text pre-processing (NLTK), TF-IDF, Naive Bayes, SVM |

Each project folder contains its own README with dataset details, methodology, and results.

## Tech Stack

- **Language:** Python 3
- **Core libraries:** pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, NLTK

## Repository Structure

```
.
├── README.md
├── project1_eda/
│   ├── README.md
│   ├── week_one.ipynb
│   ├── Dataset for Data Analytics.csv
│   └── processed_dataset.csv
├── project2_fraud_detection/
│   ├── README.md
│   ├── fraud_detection_pipeline.ipynb
│   └── data/
│       └── creditcard.csv   (download separately — see project README)
├── project3_customer_segmentation/
│   ├── README.md
│   ├── marketing_campaign.ipynb
│   ├── marketing_campaign.csv
│   └── Segmented_customer_data.csv
└── project4_sentiment_analysis/
    ├── README.md
    ├── sentiment_analysis.ipynb
    └── IMDB_Dataset.csv
```

## Running the Notebooks

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib nltk jupyter
jupyter notebook
```

