# Project 3: Unsupervised Learning — Customer Segmentation

**Notebook:** `marketing_campaign.ipynb`

## Goal

Use distance-based algorithms to discover hidden mathematical groupings in unlabeled retail data, and translate the resulting clusters into actionable business personas.

## Dataset

`marketing_campaign.csv` — [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) (Kaggle), 2,240 customers × 29 columns.

## Approach & Results

**1. Data cleaning**
- Dropped `ID`, `Z_CostContact`, `Z_Revenue` (identifier / zero-variance columns)
- Dropped 3 rows with implausible birth years (1893–1899)
- `Income`: 24 missing values median-imputed; 8 outliers capped at 117,418 via IQR
- Consolidated `Alone`, `Absurd`, `YOLO` into `Single` — final distribution: Married 864, Together 579, Single 486, Divorced 231, Widow 77

**2. Feature engineering**
`Age`, `Customer_Tenure`, `Total_Spend`, `Total_Purchases`, `Total_Campaigns_Accepted` (sum of `AcceptedCmp1`–`AcceptedCmp5`), `Children`.

**3. Encoding & scaling**
`Education` ordinally encoded; `Marital_status` one-hot encoded. All 17 resulting features standardized before PCA.

**4. PCA**
Reduced to 2 components. Explained variance: **23.1% + 11.4% = 34.6%** of total variance.

**5. Choosing k — Elbow Method + Silhouette Score**

| k     | Inertia     | Silhouette Score |
|-------|-------------|------------------|
| 2     | 6471.19     | 0.4463           |
| 3     | 4173.95     | 0.4363           |
| **4** | **2618.93** | **0.4786**       |
| 5     | 2218.78     | 0.4458           |
| 6     | 1932.79     | 0.4261           |
| 7     | 1694.04     | 0.4036           |
| 8     | 1509.41     | 0.3805           |

**k = 4** selected — highest silhouette score.

**6. Final clusters & personas**

| Cluster | Age | Income | Children | Total Spend | Purchases | Campaigns Accepted | Web Visits/mo | Count |           Persona           |
|---------|-----|--------|----------|-------------|-----------|--------------------|---------------|-------|-----------------------------|
|    0    | 48.3| $41,499| 2.0      | $145        | 10.1      | 0.1                | 6.4           | 450   | Budget-Conscious Parents    |
|    1    | 44.2| $76,064| 0.0      | $1,384      | 20.2      | 0.8                | 2.6           | 512   | High-Value Engaged Spenders |
|    2    | 35.5| $30,199| 0.8      | $118        | 8.2       | 0.1                | 6.9           | 620   | Budget-Conscious Browsers   |
|    3    | 49.4| $60,544| 1.1      | $776        | 20.3      | 0.2                | 5.2           | 655   | Steady Family Spenders      |

Cluster 1 stands out clearly: highest income and spend by a wide margin, no children, and by far the most receptive to marketing campaigns (0.8 vs 0.1–0.2 for the rest) — yet visits the website least (2.6 visits/month), suggesting this segment buys through other channels (catalog/in-store) and is worth prioritizing for direct campaign outreach.

## Tech Stack

pandas, NumPy, scikit-learn, matplotlib

## Files

|           File                |                               Description                                      |
|-------------------------------|--------------------------------------------------------------------------------|
| `marketing_campaign.csv`      | Input: raw customer data                                                       |
| `marketing_campaign.ipynb`    | Full pipeline: cleaning → feature engineering → PCA → clustering → personas    |
| `Segmented_customer_data.csv` | Output: cleaned dataset with assigned cluster labels (2,237 rows × 35 columns) |

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook marketing_campaign.ipynb
```
