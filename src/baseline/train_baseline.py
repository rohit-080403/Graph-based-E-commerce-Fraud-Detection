import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import joblib

DATA_PATH = "data/processed_sample.parquet"
MODEL_OUT = "src/baseline/xgb_baseline.model"


EXCLUDE_COLS = [
    "TransactionID", "isFraud",
    "card_id", "address_id", "device_id", "email_domain"
]


def load_and_prepare():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Train: {len(train_df)} rows ({train_df['isFraud'].mean():.4f} fraud rate)")
    print(f"Test:  {len(test_df)} rows ({test_df['isFraud'].mean():.4f} fraud rate)")

    return train_df, test_df


def encode_categoricals(train_df, test_df, feature_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()

    cat_cols = train_df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    print(f"Encoding {len(cat_cols)} categorical columns")

    for col in cat_cols:
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col] = test_df[col].fillna("missing").astype(str)

        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))  
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    return train_df, test_df


def train_model(train_df, test_df, feature_cols):
    X_train, y_train = train_df[feature_cols], train_df["isFraud"]
    X_test, y_test = test_df[feature_cols], test_df["isFraud"]

    # Handle class imbalance (~3.5% fraud rate)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=1,              # CHANGED from -1 — single-threaded, fully deterministic
        tree_method="hist",    # ADDED — explicit deterministic histogram algorithm
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, X_test, y_test


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Baseline XGBoost Results ---")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_df, test_df = load_and_prepare()

    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    train_df, test_df = encode_categoricals(train_df, test_df, feature_cols)

    model, X_test, y_test = train_model(train_df, test_df, feature_cols)
    evaluate(model, X_test, y_test)

    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")