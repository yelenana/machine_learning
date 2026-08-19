# Data

This project uses the **Bank Marketing** dataset.

- **Original source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/), used here
  via the cleaner [Kaggle mirror](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv).
- **File:** `bank-additional-full.csv` — semicolon-separated (`sep=';'`), 41,188 rows × 21
  columns.
- **Content:** direct marketing phone calls made by a Portuguese banking institution, with the
  goal of predicting whether the client subscribed to a term deposit.

## Columns

**Client data**

| Column | Description |
|---|---|
| `age` | Client's age (numeric) |
| `job` | Type of job (categorical) |
| `marital` | Marital status (categorical; `divorced` includes widowed) |
| `education` | Education level (categorical) |
| `default` | Has credit in default? (`no`, `yes`, `unknown`) |
| `housing` | Has a housing loan? (`no`, `yes`, `unknown`) |
| `loan` | Has a personal loan? (`no`, `yes`, `unknown`) |

**Last contact of the current campaign**

| Column | Description |
|---|---|
| `contact` | Contact communication type (`cellular`, `telephone`) |
| `month` | Last contact month |
| `day_of_week` | Last contact day of week |
| `duration` | Last contact duration, in seconds. **Excluded from modeling** — its value is only known after the call ends, so including it would leak the target (see the notebook, Section 1). |

**Other attributes**

| Column | Description |
|---|---|
| `campaign` | Number of contacts made during this campaign for this client (includes the last contact) |
| `pdays` | Days since the client was last contacted in a previous campaign (`999` = not previously contacted) |
| `previous` | Number of contacts made before this campaign for this client |
| `poutcome` | Outcome of the previous marketing campaign (`failure`, `nonexistent`, `success`) |

**Social & economic context**

| Column | Description |
|---|---|
| `emp.var.rate` | Employment variation rate (quarterly) |
| `cons.price.idx` | Consumer price index (monthly) |
| `cons.conf.idx` | Consumer confidence index (monthly) |
| `euribor3m` | 3-month Euribor rate (daily) |
| `nr.employed` | Number of employees (quarterly) |

**Target**

| Column | Description |
|---|---|
| `y` | Did the client subscribe to a term deposit? (`yes` / `no`) |

## Notes

- No true `NaN` values, but six columns use the literal string `"unknown"` in place of a missing
  value: `default` (8,597 rows, ~20.9%), `education` (1,731, ~4.2%), `housing` (990, ~2.4%),
  `loan` (990, ~2.4%), `job` (330, ~0.8%), `marital` (80, ~0.2%). Handled per-column in the
  notebook (Section 1 / Section 3) — mode-filled where rare, kept as its own category for
  `default` where imputing would distort the data too much.
- **Class balance:** 36,548 `no` (88.73%) vs. 4,640 `yes` (11.27%) — noticeably imbalanced, which
  is why ROC-AUC (rather than Accuracy) is used as the evaluation metric.

## Setup

Place `bank-additional-full.csv` in this `data/` folder. The notebook reads it directly:

```python
pd.read_csv(f'{path}/data/bank-additional-full.csv', sep=';')
```
