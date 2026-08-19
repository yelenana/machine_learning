# Reports

This folder is populated automatically when you run `notebook/Rybchynska_Olena__Mid-term_Project.ipynb`
top to bottom — each figure is saved as a PNG right after it's generated.

**EDA**
- `boxplot_age_campaign_euribor3m.png`, `boxplot_economic_indicators.png`,
  `boxplot_previous.png`, `boxplot_pdays.png`, `boxplot_nr_employed.png` — outlier checks for
  numerical features
- `correlation_heatmap.png` — correlation matrix of numerical features

**Modeling**
- `logistic_regression_roc_curve.png` — train vs. validation ROC curve (Logistic Regression)
- `decision_tree_visualization_depth2.png` — a depth-2 Decision Tree, for illustration
- `decision_tree_feature_importance.png` — top-10 feature importances (Decision Tree)
- `decision_tree_optimal_depth.png`, `decision_tree_optimal_leaf_nodes.png` — train/validation
  AUROC vs. `max_depth` / `max_leaf_nodes`, used to pick the tuned Decision Tree

**Interpretability**
- `shap_summary_beeswarm.png`, `shap_summary_bar.png` — SHAP analysis of the best model
  (XGBoost + Bayesian Optimization)
