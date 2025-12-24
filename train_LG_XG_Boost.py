# train_gbdt.py
import os
import json
import argparse
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from xgboost.callback import EarlyStopping, EvaluationMonitor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main(args):
    os.makedirs('models', exist_ok=True)

    # Load training data
    df = pd.read_csv(args.data_path)
    print("Columns:", df.columns.tolist())

    # Use 'input' as the ONLY feature
    feature_cols = ['input']

    # Convert into list of dicts
    X_dict = df[feature_cols].astype(str).to_dict(orient='records')


    # Vectorize categorical features
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['target'].astype(str))

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # -----------------------------
    # LightGBM
    # -----------------------------
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }

    lgb_model = lgb.train(  lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=10)
    ])

    # Predict & evaluate
    y_pred = lgb_model.predict(X_val)
    y_pred_labels = y_pred.argmax(axis=1)
    acc = accuracy_score(y_val, y_pred_labels)
    print(f"LightGBM validation accuracy: {acc:.4f}")

    # Save LightGBM model
    lgb_model.save_model('models/lightgbm.txt')

    # -----------------------------
    # XGBoost
    # -----------------------------
 
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(le.classes_),
        tree_method="hist",
        callbacks=[
            EarlyStopping(rounds=50, save_best=True),
            EvaluationMonitor(period=10)
        ]
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )


    y_pred_xgb = xgb_model.predict(X_val)
    acc_xgb = accuracy_score(y_val, y_pred_xgb)
    print(f"XGBoost validation accuracy: {acc_xgb:.4f}")

    # Save XGBoost model
    joblib.dump(xgb_model, 'models/xgboost.joblib')

    # Save vectorizer and label encoder
    joblib.dump(vec, 'models/vectorizer.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')

    # Save metadata
    meta = {'feature_cols': feature_cols}
    with open('models/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f)

    print("GBDT models (LightGBM + XGBoost) trained and saved in 'models/' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train.csv', help='CSV file with training data')
    args = parser.parse_args()
    main(args)

