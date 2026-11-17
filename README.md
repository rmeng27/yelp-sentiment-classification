# Yelp Review Sentiment Classification

<p align="center">
  <strong>Binary sentiment classification of Yelp reviews using TF-IDF features and supervised machine-learning models.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-NLP-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TF--IDF-Text%20Features-4B8BBE" alt="TF-IDF">
  <img src="https://img.shields.io/badge/Best%20Accuracy-89.3%25-2E8B57" alt="Accuracy">
  <img src="https://img.shields.io/badge/ROC--AUC-0.96-6C63FF" alt="ROC-AUC">
</p>

<p align="center">
  <img src="figures/model_comparison.svg" alt="Yelp model comparison" width="850">
</p>

## Overview

This project builds an NLP classification pipeline to identify **positive vs. negative Yelp reviews** from review text.

A sampled subset of the Hugging Face `yelp_review_full` dataset is converted from five-star labels into a binary task:

- 1-2 stars -> negative
- 4-5 stars -> positive
- 3-star neutral reviews -> excluded

Text is cleaned, transformed with **TF-IDF**, and evaluated across Logistic Regression, Random Forest, and XGBoost. Hyperparameters are tuned with 5-fold grid search.

The best-performing model is **Logistic Regression**, achieving **89.3% test accuracy** and **0.960 ROC-AUC**.

## Model Results

| Model | Test Accuracy | ROC-AUC |
|---|---:|---:|
| **Logistic Regression** | **89.3%** | **0.960** |
| XGBoost | 86.2% | 0.939 |
| Random Forest | 84.3% | 0.930 |

## Pipeline

```text
Yelp review text
  → text normalization
  → binary sentiment labels
  → stratified train/test split
  → TF-IDF vectorization
  → model tuning with GridSearchCV
  → held-out evaluation
```

## Methods

- text normalization and stopword removal
- TF-IDF vectorization with a 5,000-feature cap
- stratified train/test split
- Logistic Regression
- Random Forest
- XGBoost
- 5-fold hyperparameter search
- evaluation with accuracy, precision, recall, F1-score, and ROC-AUC

## Interpretation

The linear Logistic Regression model performs best on this high-dimensional sparse feature space. That result is consistent with the structure of TF-IDF text representations, where linear decision boundaries can be highly effective and computationally efficient.

## Repository Structure

```text
yelp-sentiment-classification/
├── README.md
├── src/
│   └── train_models.py
├── requirements.txt
└── figures/
    └── model_comparison.svg
```

## Reproducibility

The cleaned portfolio script in [`src/train_models.py`](src/train_models.py) reproduces the project workflow using the public Hugging Face Yelp Review Full dataset.

## Limitations

- The project simplifies a five-class rating problem into binary sentiment.
- Neutral 3-star reviews are excluded.
- TF-IDF ignores word order and richer contextual semantics.
- Results are based on a sampled subset rather than the full Yelp corpus.
- The original academic analysis combined sampled source splits before creating a new stratified holdout; the portfolio script preserves that workflow for reproducibility.

---

<sub>Graduate machine-learning / NLP project, 2026.</sub>
