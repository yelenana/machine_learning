# Data

This project uses the **Store Item Demand Forecasting Challenge** dataset.

- **Source:** [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview) (Kaggle).
- **File:** `train.csv` — 913,000 rows × 4 columns, no missing values.
- **Coverage:** daily sales from **2013-01-01 to 2017-12-31** (5 years), for **10 stores** ×
  **50 items** (500 series total).

## Columns

| Column | Description |
|---|---|
| `date` | Calendar date of the sale (daily granularity) |
| `store` | Store ID (1–10) |
| `item` | Item ID (1–50) |
| `sales` | Number of units sold for that item, at that store, on that date |

## Notes

- **`sales` distribution:** mean ≈ 52.3, median = 47, std ≈ 28.8, range 0–231. Distribution is
  right-skewed (a long tail of high-sales days/items), consistent with the item-level spread
  seen in the EDA section of the notebook (some items sell ~20/day, others ~100+/day).
- The notebook works with a single series in depth — **Store 1, Item 1** — filtered from this
  file; see `notebook/Rybchynska_Olena__Time_Series_Analysis.ipynb`, Section 3 onward.
- The competition also provides `test.csv`, which isn't used here since the assignment scope is
  narrowed to evaluating forecasts against a held-out slice of `train.csv` (see the notebook's
  Evaluation Approach / train-validation split).

## Setup

Place `train.csv` in this `data/` folder. The notebook reads it directly:

```python
pd.read_csv(f'{path}/data/train.csv')
```
