# Saved Models

Classical models (fit on the training set) are saved here via `joblib` for reuse without
retraining:

- `bow_vectorizer.joblib`, `tfidf_vectorizer.joblib` — fitted vectorizers (needed to transform
  new text the same way before feeding it to any of the BoW/TF-IDF models below)
- `log_reg_bow.joblib`, `log_reg_tfidf.joblib` — Logistic Regression (Models 1–2)
- `random_forest_bow.joblib` — Random Forest (Model 3)
- `xgboost_bow.joblib` — XGBoost on BoW (Model 4)
- `log_reg_cosine.joblib` — Logistic Regression on BERT cosine similarity (Model 5)
- `xgboost_combined.joblib` — XGBoost on BoW + cosine_sim + len_diff (Model 6, best non-BERT
  model, test log loss 0.332)

## BERT fine-tuned model (not included)

The fine-tuned `bert-base-uncased` model (Model 7, best overall, test log loss 0.323) is **not
saved here** — the full checkpoint is several hundred MB, too large for a standard git
repository. To reproduce it, rerun the BERT fine-tuning section of
`notebooks/quora_duplicate_detection.ipynb` — training takes roughly 9 minutes on a T4 GPU and
is fully reproducible (fixed random seeds throughout, verified across multiple reruns).

## Loading a saved model

```python
import joblib
from src.utils import tokenize  # required for the vectorizers to unpickle correctly

bow_vectorizer = joblib.load('models/bow_vectorizer.joblib')
xgb_combined = joblib.load('models/xgboost_combined.joblib')
```

Note: `bow_vectorizer`/`tfidf_vectorizer` use a custom tokenizer function (`tokenize`, from
`src/utils.py`). It must be importable — i.e. `src/` must be on the Python path — for
`joblib.load()` to unpickle them correctly.
