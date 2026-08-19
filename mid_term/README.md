# Bank Marketing — Term Deposit Subscription Prediction

Binary classification of bank telemarketing calls: predicting whether a client will subscribe
to a term deposit. Mid-term project for the ML/NLP course.

## Business Task & Goal

Predicting whether a customer will buy a product, start using a service, or take out a
subscription is a recurring problem across companies and domains. This project frames it as a
binary classification task for a bank: given data aggregated per client up to a point in time,
predict whether that client will open a term deposit (target variable `y`). Beyond raw
predictive accuracy, the project also focuses on explaining *why* the model makes the decisions
it does (feature importance, SHAP), since that explanation is a prerequisite for trusting a
model enough to move it to production.

## Data

- **Source:** [Bank Marketing (Kaggle)](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv),
  originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
- **File:** `bank-additional-full.csv` (semicolon-separated), 41,188 rows × 21 columns.
- **Target:** `y` — whether the client subscribed to a term deposit (`yes`/`no`), heavily
  imbalanced toward `no`.
- **Features:**
  - *Client data:* `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`.
  - *Last contact of the current campaign:* `contact`, `month`, `day_of_week`, `duration`
    (excluded from modeling — its value is only known after a call ends, so using it would leak
    the outcome; see the notebook's Section 1 for details).
  - *Other campaign attributes:* `campaign`, `pdays` (999 = no previous contact), `previous`,
    `poutcome`.
  - *Social/economic context (aggregated indicators):* `emp.var.rate`, `cons.price.idx`,
    `cons.conf.idx`, `euribor3m`, `nr.employed`.
- Raw data is not committed to this repository — see `data/README.md` for how to obtain it.

## Evaluation Approach

- **Primary metric:** ROC-AUC. Chosen because the dataset is imbalanced (most clients are `no`),
  so plain Accuracy is not informative enough — ROC-AUC measures how well each model separates
  the two classes regardless of the class ratio.
- **Split:** train / validation split via the project's `preprocess_data` function
  (`src/process_script.py`), with numerical features scaled by `MinMaxScaler` and categorical
  features one-hot encoded.

## Approach & Tools

1. **EDA** — data structure, missing-value analysis (including `"unknown"` treated as missing),
   class imbalance, numeric-feature distributions/outliers, correlation analysis.
2. **Preprocessing** — `"unknown"` converted to `NaN` and handled per-column (mode-fill where
   missing values are rare, kept as its own category for `default` where imputing would distort
   the data too much); `duration` dropped to avoid data leakage; numeric scaling and categorical
   encoding via `preprocess_data`.
3. **Modeling** — four model types trained: Logistic Regression, kNN (tuned via `GridSearchCV`),
   Decision Tree (tuned via `max_depth`/`max_leaf_nodes` sweep), and XGBoost.
4. **Hyperparameter tuning (boosting)** — two approaches compared: `RandomizedSearchCV`
   (scikit-learn) and Bayesian Optimization (Optuna).
5. **Result analysis** — feature importance for the best model, SHAP analysis (`shap.TreeExplainer`)
   for deeper interpretability, and a review of misclassified records (false negatives vs. false
   positives).

**Tools:** Python, pandas, NumPy, scikit-learn, XGBoost, Optuna, SHAP, matplotlib, seaborn.
Developed and run in Google Colab.

## Results

| Model | Hyperparameters | Train ROC-AUC | Validation ROC-AUC | Gap |
|---|---|---:|---:|---:|
| Logistic Regression | solver='liblinear' | 0.7933 | 0.8017 | 0.0084 |
| kNN (tuned) | n_neighbors=24 | 0.8474 | 0.7662 | 0.0811 |
| Decision Tree (tuned) | max_leaf_nodes=20 | 0.7886 | 0.8000 | 0.0113 |
| XGBoost baseline | n_estimators=200, max_depth=4 | 0.8443 | 0.8152 | 0.0291 |
| XGBoost + RandomizedSearchCV | optimized hyperparameters | 0.8245 | 0.8118 | 0.0127 |
| **XGBoost + Bayesian Optimization** | optimized hyperparameters | 0.8294 | **0.8162** | 0.0132 |

Error analysis on the best model: 695 false negatives (missed potential depositors) vs. 111
false positives — the model misses far more real depositors than it wrongly flags.

## Conclusions

**XGBoost tuned with Bayesian Optimization performs best** (Validation ROC-AUC 0.8162) with a
small train/validation gap, indicating good generalization. Boosting outperforms the linear and
distance-based models because it captures non-linear relationships — especially between the
macroeconomic indicators and prior-contact history — that Logistic Regression can't represent,
while tuning kept it from overfitting the way an unconstrained Decision Tree or a low-`k` kNN
did.

**Key findings:**
- Model quality is driven mainly by macroeconomic context (`nr.employed`, `emp.var.rate`,
  `euribor3m`, `cons.conf.idx`) rather than individual client attributes — confirmed by both
  feature importance and SHAP analysis.
- Whether a previous campaign succeeded (`poutcome_success`) is one of the strongest predictors
  of a positive response.
- Hyperparameter tuning mainly reduced overfitting (narrowed the train/validation gap) rather
  than substantially raising validation ROC-AUC.

**Limitations:** no dedicated test set was held out (all metrics are train vs. validation);
`duration` — likely the single most predictive feature in a live-call setting — was correctly
excluded to avoid data leakage; only XGBoost was tested among boosting algorithms; outliers were
deliberately kept rather than removed, which may partly explain the weaker Logistic
Regression/kNN scores. See the notebook's Conclusions section for the full discussion, including
further improvement ideas (test-set evaluation, threshold tuning, LightGBM/CatBoost comparison,
cyclical encoding of `month`/`day_of_week`).

## Installation & Usage

This project was built and run in **Google Colab**.

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd bank-marketing-term-deposit
   ```
2. Open `notebook/Rybchynska_Olena__Mid-term_Project.ipynb` in Google Colab (or Jupyter).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the data (see `data/README.md`) and place `process_script.py` in `src/`; update the
   `path` variables near the top of the notebook to point to where you saved them.
5. Run all cells top to bottom (**Runtime → Run all** in Colab).

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`optuna`, `shap`, `matplotlib`, `seaborn`.
