# Saved Models

This folder is populated automatically when you run `notebook/Rybchynska_Olena__Time_Series_Analysis.ipynb`
top to bottom — each model is saved right after training via `darts`'s own `.save()` method
(or `joblib` for the scaler), so nothing needs to be retrained to reuse it.

- `naive_seasonal.pkl`, `naive_drift.pkl` — the two components of the naive baseline
  (`NaiveSeasonal(K=7)` + `NaiveDrift`)
- `xgboost.pkl`, `scaler.joblib` — XGBoost model and the `Scaler` used to preprocess the series
  before feeding it in
- `exponential_smoothing.pkl` — Exponential Smoothing (`seasonal_periods=7`)
- `arima.pkl` — ARIMA(7,1,0)
- `auto_arima.pkl` — AutoARIMA (parameters chosen automatically)
- `sarima.pkl` — SARIMA(2,1,0)(1,1,0,7)
- `prophet_best_model.pkl` — Prophet — **best overall model** (see the notebook's Conclusions)
- `rnn_lstm.pt` (+ `rnn_lstm.pt.ckpt`) — RNN (LSTM), saved via darts' PyTorch-aware `.save()`

## Loading a saved model

```python
from darts.models import ARIMA  # or whichever model class you need

model = ARIMA.load('models/arima.pkl')  # each darts model class has a matching .load()
forecast = model.predict(30)
```

For the XGBoost model, load the scaler first (`joblib.load('models/scaler.joblib')`) to
preprocess new data the same way before calling `.predict()`.
