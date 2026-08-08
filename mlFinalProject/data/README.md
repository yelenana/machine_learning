# Data

This project uses the **Quora Question Pairs** dataset.

- **Original source:** [Kaggle — Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
- **Train/test split used in this project:** the course provided a pre-split version of the
  original dataset — 80/20, stratified by `is_duplicate` — as two files:
  - `quora_question_pairs_train.csv.zip`
  - `quora_question_pairs_test.csv.zip`

Using this exact split (rather than re-splitting the raw Kaggle data yourself) is required to
reproduce the results reported in the main README, since the notebook's `train_idx`/`val_idx`
split and all reported metrics are based on it.

## Columns

| Column | Description |
|---|---|
| `id` | Row index (kept from the original dataset) |
| `qid1`, `qid2` | Question IDs from the original dataset |
| `question1`, `question2` | Raw question text |
| `is_duplicate` | Target: 1 if the pair is a duplicate, 0 otherwise |

## Setup

Place both `.zip` files in this `data/` folder (they are excluded from version control via
`.gitignore` — see the repository root). The notebook reads them directly as zipped CSVs, no
manual extraction needed:

```python
pd.read_csv('data/quora_question_pairs_train.csv.zip', index_col=0)
```
