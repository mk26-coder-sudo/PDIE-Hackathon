import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────

TRACK1_FEATURES = [
    "F1_debt_burden",
    "F2_income_volatility",
    "F3_lending_app_score",
    "F4_savings_depletion",
    "F5_salary_delay",
    "F6_autodebit_fail",
    "F7_spending_cuts",
]

TRACK2_FEATURES = TRACK1_FEATURES + [
    "F8_stress_velocity",
    "F9_foresight_momentum",
    "F10_neighborhood_risk",
]

TARGET = "actual_default"

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────

def load_data(csv_path: str):
    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"[INFO] Default rate: {df[TARGET].mean():.2%}")
    return df


# ─────────────────────────────────────────
# TRAIN TRACK 1 — XGBoost
# ─────────────────────────────────────────

def train_track1(df: pd.DataFrame):
    print("\n[TRACK 1] Training XGBoost on 7 snapshot features...")

    X = df[TRACK1_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(model, cv=3, method="isotonic")
    calibrated.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"[TRACK 1] AUC Score: {auc:.4f}")
    print(f"[TRACK 1] Classification Report:")
    print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))

    return calibrated, auc


# ─────────────────────────────────────────
# TRAIN TRACK 2 — LightGBM
# ─────────────────────────────────────────

def train_track2(df: pd.DataFrame):
    print("\n[TRACK 2] Training LightGBM on ambiguous band (PD 15-25%)...")

    # Only train on ambiguous zone customers
    ambiguous = df[
        (df["pd_pct"] >= 15) & (df["pd_pct"] <= 25)
    ].copy()

    print(f"[TRACK 2] Ambiguous band size: {len(ambiguous)} rows")

    if len(ambiguous) < 50:
        print("[TRACK 2] WARNING: Too few ambiguous samples. Using full dataset.")
        ambiguous = df.copy()

    X = ambiguous[TRACK2_FEATURES]
    y = ambiguous[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    # Calibrate
    calibrated = CalibratedClassifierCV(model, cv=3, method="isotonic")
    calibrated.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"[TRACK 2] AUC Score: {auc:.4f}")

    return calibrated, auc


# ─────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────

def save_models(track1_model, track2_model, output_dir="app/models"):
    os.makedirs(output_dir, exist_ok=True)

    path1 = os.path.join(output_dir, "xgboost_track1.pkl")
    path2 = os.path.join(output_dir, "lightgbm_track2.pkl")

    joblib.dump(track1_model, path1)
    joblib.dump(track2_model, path2)

    print(f"\n[INFO] Models saved:")
    print(f"  Track 1 → {path1}")
    print(f"  Track 2 → {path2}")


# ─────────────────────────────────────────
# MAIN — Run Training Pipeline
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Path to your generated CSV
    CSV_PATH = "data/pdie_dataset.csv"

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset not found at {CSV_PATH}")
        print("[INFO] Please generate it using the dataset generator")
        print("[INFO] and save it as: data/pdie_dataset.csv")
        exit(1)

    # Load
    df = load_data(CSV_PATH)

    # Train both models
    track1_model, track1_auc = train_track1(df)
    track2_model, track2_auc = train_track2(df)

    # Save
    save_models(track1_model, track2_model)

    print(f"\n{'='*40}")
    print(f"  TRAINING COMPLETE")
    print(f"  Track 1 (XGBoost)  AUC: {track1_auc:.4f}")
    print(f"  Track 2 (LightGBM) AUC: {track2_auc:.4f}")
    print(f"{'='*40}")
