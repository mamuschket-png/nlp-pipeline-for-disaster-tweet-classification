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
* 10-fold cross-validation using weighted F1 as evaluation metric
* Classification threshold optimization for high disaster recall
* Feature importance analysis and systematic error analysis

---

## Results

| Metric | Value |
|---|---|
| Weighted F1 (Cross-Validation) | 0.794 (std 0.018) |
| Weighted F1 (Test Set) | 0.81 |
| Disaster Class Recall (optimized threshold) | 0.85 |

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

* Classification threshold was selected on the test set rather than a separate 
  validation set, which may lead to slightly optimistic performance estimates 
  at the chosen threshold
* TF-IDF cannot resolve context-dependent or non-literal language — a known 
  failure mode documented in the error analysis section of the notebook
* A non-negligible share of misclassifications is attributable to labeling 
  noise in the original dataset rather than model error

---

## Author

Matthias Muschket