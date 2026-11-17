"""Yelp review sentiment classification.

Portfolio-cleaned version of the original graduate project workflow.
"""

import re
import nltk
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 42
SAMPLE_SIZE = 7000


def clean_text(text: str, stop_words: set[str]) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return " ".join(words)


def load_binary_yelp_sample() -> pd.DataFrame:
    dataset = load_dataset("yelp_review_full")
    train_df = pd.DataFrame(dataset["train"]).sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    test_df = pd.DataFrame(dataset["test"]).sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # Preserve the original project workflow: combine sampled source splits,
    # remove neutral reviews, and create a fresh stratified holdout.
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df[df["label"] != 2].copy()
    df["sentiment"] = (df["label"] >= 3).astype(int)
    return df


def evaluate_model(name, estimator, params, X_train, X_test, y_train, y_test):
    grid = GridSearchCV(estimator, params, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n=== {name} ===")
    print("Best parameters:", grid.best_params_)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

    return {"model": name, "estimator": best, "accuracy": accuracy, "roc_auc": auc}


def main():
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

    df = load_binary_yelp_sample()
    df["clean_text"] = df["text"].apply(lambda x: clean_text(x, stop_words))

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    results = []
    results.append(evaluate_model(
        "Logistic Regression",
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear"], "class_weight": ["balanced", None]},
        X_train_tfidf, X_test_tfidf, y_train, y_test
    ))
    results.append(evaluate_model(
        "Random Forest",
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [5, 10, 15], "class_weight": ["balanced", None]},
        X_train_tfidf, X_test_tfidf, y_train, y_test
    ))
    results.append(evaluate_model(
        "XGBoost",
        XGBClassifier(eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
        X_train_tfidf, X_test_tfidf, y_train, y_test
    ))

    summary = pd.DataFrame([
        {"Model": r["model"], "Accuracy": r["accuracy"], "ROC_AUC": r["roc_auc"]}
        for r in results
    ]).sort_values("Accuracy", ascending=False)

    print("\n=== Model Comparison ===")
    print(summary.to_string(index=False))

    best = max(results, key=lambda r: r["accuracy"])
    examples = [
        "The service, food, and music are all great, we had the best experience.",
        "The waiting time was really long, and the waiters had bad attitudes.",
    ]
    print(f"\nExample predictions using {best['model']}:")
    for review in examples:
        vector = tfidf.transform([clean_text(review, stop_words)])
        pred = best["estimator"].predict(vector)[0]
        print(f"- {'positive' if pred == 1 else 'negative'}: {review}")


if __name__ == "__main__":
    main()
