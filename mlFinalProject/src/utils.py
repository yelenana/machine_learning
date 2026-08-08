"""
Reusable helper functions for the Quora Question Pairs duplicate-detection project.
Extracted from the main notebook (notebooks/quora_duplicate_detection.ipynb) to avoid
duplicating the same code across preprocessing, modeling, and evaluation steps.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import log_loss, f1_score, accuracy_score

stemmer = SnowballStemmer(language='english')


def tokenize(text):
    """Tokenize, keep alphabetic tokens only, and stem (also lowercases as a side effect
    of SnowballStemmer.stem()). Stopwords are NOT removed here — pass a stopword list
    separately to CountVectorizer/TfidfVectorizer's `stop_words` argument."""
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    return [stemmer.stem(word) for word in tokens]


def get_stemmed_stopwords(raw_stopwords):
    """Stem each stopword through the same `tokenize()` pipeline used on the corpus, so
    stemmed forms (and multi-token contractions like "couldn't" -> "could") are correctly
    recognized as stopwords too. Passing raw (unstemmed) stopwords directly to a vectorizer
    whose tokenizer stems the text causes many stopwords to leak into the vocabulary."""
    stemmed = set()
    for word in raw_stopwords:
        stemmed.update(tokenize(word))
    return sorted(stemmed)


def evaluate_on_test(model, X_test, y_test, model_name, features_name, dataset="test"):
    """Evaluate a fitted sklearn-style model (.predict / .predict_proba) and return a
    results-table row dict with log loss, F1, and accuracy."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": model_name,
        "features": features_name,
        "dataset": dataset,
        "log_loss": log_loss(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
    }


class QuestionPairsDataset(Dataset):
    """PyTorch Dataset for BERT fine-tuning: encodes each (question1, question2) pair as a
    single [CLS] question1 [SEP] question2 [SEP] sequence. Used to build the train,
    validation, and test datasets for the Hugging Face Trainer."""

    def __init__(self, q1, q2, labels, tokenizer, max_length=64):
        self.q1 = list(q1)
        self.q2 = list(q2)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.q1)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.q1[idx],
            self.q2[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    """compute_metrics callback for the Hugging Face Trainer: converts raw logits to
    probabilities before computing log loss (log loss requires probabilities, not logits
    or hard labels)."""
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)
    return {
        'log_loss': log_loss(labels, probs),
        'f1': f1_score(labels, preds),
        'accuracy': accuracy_score(labels, preds),
    }
