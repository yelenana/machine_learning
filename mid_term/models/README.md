# Saved Models

This folder is populated automatically when you run `notebook/Rybchynska_Olena__Mid-term_Project.ipynb`
top to bottom — each model and the fitted preprocessing objects are saved via `joblib` right
after training, so nothing needs to be retrained to reuse them.

- `scaler.joblib`, `encoder.joblib` — the `MinMaxScaler`/`OneHotEncoder` fitted on the training
  split (needed to preprocess any new raw data the same way before feeding it to a model)
- `logistic_regression.joblib` — baseline Logistic Regression
- `knn_tuned.joblib` — kNN tuned via `GridSearchCV` (`n_neighbors=24`)
- `decision_tree_tuned.joblib` — Decision Tree tuned via leaf-node sweep (`max_leaf_nodes=20`)
- `xgboost_baseline.joblib` — untuned XGBoost
- `xgboost_randomized_search.joblib` — XGBoost tuned with `RandomizedSearchCV`
- `xgboost_bayesian_optuna_best.joblib` — XGBoost tuned with Optuna (Bayesian Optimization) —
  **best overall model** (see the notebook's Conclusions)

## Loading a saved model

```python
import joblib

scaler = joblib.load('models/scaler.joblib')
encoder = joblib.load('models/encoder.joblib')
best_model = joblib.load('models/xgboost_bayesian_optuna_best.joblib')
```

To score new raw data, preprocess it with `src/process_script.py`'s `preprocess_new_data`
using the saved `scaler`/`encoder`/`input_cols`, then call `best_model.predict_proba(...)`.
