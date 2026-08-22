# Store Item Demand Forecasting — Single-Series Time Series Modeling

Time series forecasting on 5 years of daily sales for 50 items across 10 stores, comparing
baseline, statistical, and machine learning models on a single series before discussing how
the winning approach would scale to the full dataset.

## Business Task & Goal

Forecast next-month daily sales for each of 50 items across 10 stores (500 series total). To
keep the scope manageable, the project focuses on modeling and evaluating a single series in
depth — **Store 1, Item 1** — comparing forecasting approaches end-to-end (from naive baselines
to gradient boosting, statistical, and deep learning models), then closes with a discussion of
how the best-performing approach should be scaled to all 500 store-item pairs.

## Data

- **Source:** [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview) (Kaggle).
- **File:** `train.csv` — daily sales, 5 years (2013–2017), 10 stores × 50 items.
- **Columns:** `date`, `store`, `item`, `sales`.
- Raw data is not committed to this repository — see `data/README.md` for how to obtain it.

## Evaluation Approach

- **Primary metric:** MAPE (Mean Absolute Percentage Error) — used instead of the competition's
  own SMAPE, per the assignment's brief.
- **Split:** chronological train/validation split via `series.split_before('2017-10-01')` — the
  last ~3 months held out for validation, no shuffling (as is required for time series).
- **Backtesting:** the best model is additionally backtested over a rolling 1-year window with
  1-month-ahead forecasts, to check it isn't just overfitting the single validation cut.

## Approach & Tools

1. **EDA** — sales dynamics by item (with confidence bands across stores), sales distribution
   by store per item, annual sales trend by store.
2. **Single-series decomposition** — additive decomposition (trend / seasonality / residuals)
   with `statsmodels`, PACF, and `check_seasonality` (from `darts`) to identify the relevant
   lags and seasonal period.
3. **Modeling** — eight models compared on the same train/validation split:
   - Naive Seasonal (K=7) + Naive Drift (baseline)
   - XGBoost (`darts.models.XGBModel`, scaled features)
   - Exponential Smoothing
   - ARIMA
   - AutoARIMA
   - SARIMA (added after ARIMA/AutoARIMA failed to capture seasonality)
   - Prophet
   - RNN (LSTM)
4. **Backtesting** — the best model (Prophet) backtested over a rolling 1-year window.
5. **Scaling discussion** — how the single-series approach would extend to all 50×10 series.

**Tools:** Python, pandas, NumPy, statsmodels, [`darts`](https://unit8co.github.io/darts/),
XGBoost, Prophet, PyTorch / PyTorch Lightning (for the RNN), statsforecast (AutoARIMA backend),
matplotlib, seaborn. Developed and run in Google Colab.

## Results

| Model | Validation MAPE |
|---|---:|
| **Prophet** | **23.80%** |
| SARIMA(2,1,0)(1,1,0,7) | 27.81% |
| RNN (LSTM) | 33.61% |
| XGBoost | 37.62% |
| Exponential Smoothing | 39.01% |
| ARIMA(7,1,0) | 39.68% |
| Naive Seasonal(K=7) + Drift | 39.91% |
| AutoARIMA | 40.29% |

Prophet backtest (rolling 1-year window, 1-month-ahead forecasts): **MAPE 20.37%**.

## Conclusions

**Prophet performs best** (Validation MAPE 23.80%, backtest MAPE 20.37%), because it explicitly
models both weekly and annual seasonality at once — the key characteristic of this data. SARIMA
comes second once given an explicit seasonal component; plain ARIMA and AutoARIMA both failed to
capture seasonality at all (forecasts came out as nearly flat lines). The RNN (LSTM) beat the
naive baseline and XGBoost but underperformed Prophet/SARIMA — likely due to the small number of
training epochs (20) and short input window (30 days) used here.

**Scaling to all 50 items × 10 stores** — three options are worth considering: (1) 50
per-item models with store as a covariate, (2) a single global model trained on all 500 series
with `store`/`item` as features, or (3) 500 fully separate models if per-series accuracy is
critical and compute allows. See the notebook's Conclusions section for the full discussion,
including limitations (single-series validation only; the RNN likely has room to improve with
more epochs, a longer window, and calendar covariates).

## Installation & Usage

This project was built and run in **Google Colab**.

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd store-item-demand-forecasting
   ```
2. Open `notebook/Rybchynska_Olena__Time_Series_Analysis.ipynb` in Google Colab (or Jupyter).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the data (see `data/README.md`); update the `path` variable near the top of the
   notebook to point to where you saved it.
5. Run all cells top to bottom (**Runtime → Run all** in Colab).

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `statsmodels`, `darts`, `xgboost`,
`torch`, `pytorch-lightning`, `prophet`, `statsforecast`, `matplotlib`, `seaborn`.
