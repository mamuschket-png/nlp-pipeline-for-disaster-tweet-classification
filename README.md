# Disaster Tweet Classification: An NLP Pipeline

A natural language processing project that classifies tweets as reporting real 
disasters or not. Built as part of a Data Science retraining program, this project 
demonstrates a full NLP pipeline from text preprocessing and vectorization to model 
evaluation and error analysis.

*Created for learning purposes. No claim of production-ready performance is made.*

---

## Pipeline

* Text preprocessing: lowercasing, non-ASCII removal, URL removal, lemmatization
* TF-IDF vectorization with hyperparameter tuning (ngram range, max features)
* Logistic Regression classifier with regularization tuning
* Three-way train/validation/test split for unbiased final evaluation
* Hyperparameter tuning optimized for recall on the disaster class via 10-fold cross-validation
* Classification threshold optimization for high disaster recall
* Feature importance analysis (coefficients and odds ratios) and systematic error analysis

---

## Results

Final model evaluated on the held-out test set, using the tuned hyperparameters 
(C=10, ngram_range=(1,2), max_features=10000) and an optimized classification 
threshold of 0.25:

| Metric | Value |
|---|---|
| Disaster Class Recall (Test Set) | 0.84 |
| Disaster Class Precision (Test Set) | 0.62 |
| Weighted F1 (Test Set) | 0.72 |

The pipeline deliberately favors recall over precision, reflecting the higher 
real-world cost of missing a real disaster compared to raising a false alarm.

---

## Project Structure

```
nlp-pipeline-for-disaster-tweet-classification/
├── Data/
│   └── train.csv                          # Input data (see below)
├── disaster_tweet_classification.ipynb    # Main notebook
├── requirements.txt
└── README.md
```

---

## Data

The dataset (`train.csv`) is included in the `Data/` folder.

---

## Requirements

```
pandas>=3.0.3
numpy>=2.4.6
matplotlib>=3.11.0
scikit-learn>=1.9.0
nltk>=3.9.4
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Known Limitations

* TF-IDF cannot resolve context-dependent or non-literal language — a known 
  failure mode documented in the error analysis section of the notebook
* Some high-weight features reflect geographic or temporal specificities of the 
  training data (e.g. "california", "hiroshima") rather than generalizable 
  disaster signals
* A non-negligible share of misclassifications is attributable to labeling 
  noise in the original dataset rather than model error

---

## Author

Matthias Muschket