# Quora Question Pairs — Duplicate Question Detection

Binary classification of question pairs from Quora: predicting whether two differently-worded
questions ask the same thing. Final project for the ML/NLP course.

## Business Task & Goal

Detecting duplicate questions is a common problem for any platform where users submit free-text
content: Q&A sites (merging duplicate threads, surfacing existing answers instead of letting
questions go unanswered), marketplaces (catching duplicate product listings), or support systems
(catching duplicate tickets). This project builds and compares several models that predict the
probability that two questions are duplicates, from simple word-count baselines up to a
fine-tuned transformer.

## Data

- **Source:** [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) (Kaggle),
  pre-split by the course into train/test.
- **Size:** 323,432 pairs (train), 80,858 pairs (test), split 80/20 stratified by `is_duplicate`
  from the original competition data.
- **Columns:** `qid1`, `qid2` (question IDs from the original dataset), `question1`,
  `question2` (raw text), `is_duplicate` (target, 0/1).
- **Class balance:** ~63% non-duplicate, ~37% duplicate.
- Raw data files are not committed to this repository (see `data/README.md` for how to obtain
  them) — they're large and available directly from the course/Kaggle.

## Evaluation Approach

- **Primary metric:** Log loss (Cross-Entropy Loss) between predicted probabilities and true
  labels — the model must output calibrated probabilities, not just hard 0/1 predictions.
- **Secondary metrics:** F1-score and accuracy, tracked to catch cases where a model games log
  loss without being practically useful (e.g., the majority-class baseline).
- **Split:** the provided training data is further split into train (80%) / validation (20%),
  stratified by `is_duplicate`, `random_state=42`. All vocabulary-fitting steps (BoW, TF-IDF) use
  only the training portion, to avoid leaking validation-set statistics. The separately provided
  `test.csv` is held out completely and used only once, for final evaluation.

## Approach & Tools

1. **EDA** — class balance, text length distributions (characters/words), word frequency
   analysis, length-difference between question pairs.
2. **Preprocessing** — NLTK tokenization, `SnowballStemmer`, stopword removal (stopwords are
   passed through the same tokenize/stem pipeline as the text, so contracted forms like
   `wouldn't` are filtered correctly).
3. **Feature extraction:**
   - Bag-of-Words and TF-IDF (`max_features=4000`, chosen via corpus coverage analysis — 90% of
     word occurrences covered by ~4,000 most frequent tokens).
   - BERT sentence embeddings (`all-MiniLM-L6-v2`) + cosine similarity between question pairs.
   - Word2Vec/GloVe were deliberately skipped in favor of going straight to BERT-based
     embeddings and fine-tuning, to avoid redundant work.
4. **Modeling** — baseline (majority class) plus 7 models: Logistic Regression (BoW, TF-IDF),
   Random Forest (BoW), XGBoost (BoW), Logistic Regression on BERT cosine similarity, XGBoost on
   BoW + cosine similarity + length-difference, and a fine-tuned `bert-base-uncased` (question
   pairs encoded as a single `[CLS] q1 [SEP] q2 [SEP]` sequence; trained on a 50,000-pair
   stratified subsample for 2 epochs due to compute/time constraints, best checkpoint by log
   loss kept).
5. **Result analysis** — feature importance (XGBoost), confusion matrix and misclassified-example
   inspection (BERT).

**Tools:** Python, pandas, NumPy, scikit-learn, NLTK, XGBoost, sentence-transformers, Hugging
Face `transformers`, PyTorch, matplotlib, seaborn. Developed and run in Google Colab (GPU
required for the BERT embedding and fine-tuning steps).

## Results

| Model | Features | Dataset | Log loss | F1 | Accuracy |
|---|---|---|---|---|---|
| Baseline | - | validation | 0.6585 | 0.000 | 0.6308 |
| Logistic Regression | BoW | validation | 0.5421 | 0.5827 | 0.7289 |
| Logistic Regression | TF-IDF | validation | 0.5419 | 0.5799 | 0.7315 |
| Random Forest | BoW | validation | 0.5913 | 0.3555 | 0.7005 |
| XGBoost | BoW | validation | 0.5176 | 0.5380 | 0.7426 |
| Logistic Regression | BERT cosine sim | validation | 0.4230 | 0.7115 | 0.7813 |
| XGBoost | BoW + cosine_sim + len_diff | validation | 0.3320 | 0.7948 | 0.8457 |
| **BERT fine-tuned** | raw text (50K subsample) | validation | **0.3222** | **0.8049** | **0.8566** |
| Baseline | - | test | 0.6585 | 0.000 | 0.6308 |
| Logistic Regression | BoW | test | 0.5422 | 0.5839 | 0.7286 |
| Logistic Regression | TF-IDF | test | 0.5420 | 0.5808 | 0.7309 |
| Random Forest | BoW | test | 0.5913 | 0.3540 | 0.6999 |
| XGBoost | BoW | test | 0.5183 | 0.5394 | 0.7424 |
| Logistic Regression | BERT cosine sim | test | 0.4221 | 0.7109 | 0.7810 |
| XGBoost | BoW + cosine_sim + len_diff | test | 0.3325 | 0.7934 | 0.8447 |
| **BERT fine-tuned** | raw text (50K subsample) | test | **0.3232** | **0.8034** | **0.8556** |

Test-set metrics closely match validation for every model (differences under 0.001–0.002 on
every metric), confirming the model comparison wasn't overfit to the validation set.

## Conclusions

**Fine-tuned BERT is the best-performing model**, even trained on only ~15% of the available
training data. It wins by learning a joint representation of both questions together, rather
than comparing two independently-built representations. **XGBoost on BoW + cosine similarity +
length difference is a close, much cheaper second place** — near-BERT accuracy without a GPU or
a transformer forward pass at inference time, making it the more practical production choice.

Key findings:
- Semantic similarity (BERT cosine similarity) is a far stronger signal than lexical overlap
  (BoW/TF-IDF) on its own, and lexical + semantic signals are complementary when combined.
- More model complexity isn't automatically better — a lightly-tuned Random Forest
  underperformed simple Logistic Regression on the same sparse features.
- Error analysis showed that a meaningful share of "model mistakes" are actually disagreements
  with debatable source labels — a known characteristic of the Quora Question Pairs dataset.

**Limitations:** BERT was fine-tuned on a data subsample due to time constraints; no
hyperparameter search was run for Random Forest/XGBoost; dataset label noise caps the achievable
score regardless of model quality. See the notebook's Task 8 section for the full discussion,
including a documented and since-fixed vocabulary-leakage issue in the BoW/TF-IDF vectorizers.

## Installation & Usage

This project was built and run in **Google Colab** (a GPU runtime is required for the BERT
embedding and fine-tuning cells — `Runtime → Change runtime type → T4 GPU`).

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd quora-duplicate-questions
   ```
2. Open `notebooks/quora_duplicate_detection.ipynb` in Google Colab (or Jupyter, for the
   non-GPU sections).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the data (see `data/README.md`) and update the `path` variable near the top of the
   notebook to point to where you saved `quora_question_pairs_train.csv.zip` and
   `quora_question_pairs_test.csv.zip`.
5. Run all cells top to bottom (**Runtime → Run all** in Colab). Full execution — including
   BERT embedding generation and fine-tuning — takes roughly 30–40 minutes on a T4 GPU.

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `nltk`, `xgboost`,
`matplotlib`, `seaborn`, `scipy`, `sentence-transformers`, `transformers`, `torch`.
