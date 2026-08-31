"""
src/train.py
Standalone training script — exports model artifacts for API deployment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def train_and_export():
    # 1. Load dataset
    print("[INFO] Loading dataset...")
    df = pd.read_csv("data/creditcard.csv")

    # 2. Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 3. Stratified train-test split
    print("[INFO] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Scale Amount and Time (fit on train only)
    print("[INFO] Fitting StandardScaler on Amount & Time...")
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    # 5. Apply SMOTE on training data only
    print("[INFO] Applying SMOTE on training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE: {np.bincount(y_train_resampled)}")

    # 6. Train XGBoost
    print("[INFO] Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train_resampled, y_train_resampled)

    # 7. Evaluate on test set
    print("[INFO] Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\n", classification_report(y_test, y_pred, digits=4))

    # 8. Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_xgboost.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("[SUCCESS] Saved: models/fraud_xgboost.joblib")
    print("[SUCCESS] Saved: models/scaler.joblib")


if __name__ == "__main__":
    train_and_export()