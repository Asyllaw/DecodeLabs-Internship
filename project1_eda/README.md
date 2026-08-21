# Project 1: Advanced EDA & Feature Engineering

**Notebook:** `week_one.ipynb`

## Goal

Transform raw, chaotic e-commerce order data into a mathematically clean dataset ready for machine learning algorithms.

## Dataset

`Dataset for Data Analytics.csv` — 1,200 e-commerce orders, 14 columns (OrderID, Date, CustomerID, Product, Quantity, UnitPrice, ShippingAddress, PaymentMethod, OrderStatus, TrackingNumber, ItemsInCart, CouponCode, ReferralSource, TotalPrice).

## Approach & Results

**1. Missing data**
Only `CouponCode` had missing values. Filled with the explicit category `"No Coupon Code"` rather than a statistical imputation, since a missing coupon code represents "no coupon used," not lost data.

**2. Outlier detection (IQR method)**
Applied to all four numeric columns. Results from the executed notebook:

| Column      | Lower bound | Upper bound | Outliers found |
|-------------|-------------|-------------|----------------|
| Quantity    | -1.00       | 7.00        | 0              |
| UnitPrice   | -317.20     | 1024.83     | 0              |
| ItemsInCart | -0.50       | 11.50       | 0              |
| TotalPrice  | -1341.41    | 3330.41     | 8              |

The 8 `TotalPrice` outliers were capped (clipped) to the upper bound rather than dropped, preserving the rest of each order's data. Boxplots are included in the notebook to visualize the distributions after treatment.

**3. Feature engineering**
Six new columns were derived:
- `OrderMonth`, `OrderDay` — calendar components extracted from `Date`
- `isweekend` — binary flag, `1` if the order was placed on a Saturday or Sunday (derived from `Date.dt.dayofweek`)
- `AveragePrice` — `TotalPrice / ItemsInCart`
- `HasCoupon` — binary flag from the cleaned `CouponCode` column
- `IsCancelledOrReturned` — binary flag from `OrderStatus`

## Tech Stack

pandas, NumPy, matplotlib

## Files

|            File                  |                          Description                                |
|----------------------------------|---------------------------------------------------------------------|
| `Dataset for Data Analytics.csv` | Input: raw e-commerce orders                                        |
| `week_one.ipynb`                 | Full EDA and feature engineering notebook                           |
| `processed_dataset.csv`          | Output: cleaned, feature-enriched dataset (1,200 rows × 20 columns) |

## How to Run

```bash
pip install pandas numpy matplotlib
jupyter notebook week_one.ipynb
```
