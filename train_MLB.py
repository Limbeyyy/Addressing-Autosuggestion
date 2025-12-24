# train_naive_bayes.py
import os
import json
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def main(args):
    os.makedirs('models', exist_ok=True)

    # -----------------------------
    # Load training data
    # -----------------------------
    df = pd.read_csv(args.data_path)
    print("Columns:", df.columns.tolist())

    # Use 'input' as the ONLY feature
    feature_cols = ['input']

    # Convert into list of dicts
    X_dict = df[feature_cols].astype(str).to_dict(orient='records')

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['target'].astype(str))

    # -----------------------------
    # Train / Validation split
    # -----------------------------
    X_train_dict, X_val_dict, y_train, y_val = train_test_split(
        X_dict,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Vectorization
    # -----------------------------
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(X_train_dict)
    X_val = vec.transform(X_val_dict)

    # -----------------------------
    # Train Naive Bayes
    # -----------------------------
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # -----------------------------
    # Evaluate accuracy
    # -----------------------------
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"✔ Validation Accuracy: {accuracy:.4f}")

    # -----------------------------
    # Save model & encoders
    # -----------------------------
    joblib.dump(clf, 'models/naive_bayes.joblib')
    joblib.dump(vec, 'models/vectorizer.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')

    # -----------------------------
    # Save metadata
    # -----------------------------
    meta = {
        'feature_cols': feature_cols,
        'model': 'MultinomialNB',
        'validation_accuracy': accuracy,
        'num_samples': len(df)
    }

    with open('models/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("✔ Model, vectorizer, encoder, and metadata saved in 'models/'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/train.csv',
        help='CSV file with training data'
    )
    args = parser.parse_args()
    main(args)
