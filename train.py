"""Train the loan approval models and save them for the web app.

Mirrors the steps in Loan_Approval_Prediction.ipynb:
    clean column names -> drop duplicates -> encode -> 80/20 stratified split
    -> Logistic Regression + Random Forest -> compare -> keep Random Forest.

Usage:
    python train.py                      # uses data/loan_approval_dataset.csv
    python train.py --csv path/to.csv    # custom dataset
    python train.py --demo               # synthetic data (for trying the app)

If the CSV is missing, a synthetic demo dataset with the same schema is used so
the app still runs. Replace it with the real file and retrain for real results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "loan_approval_dataset.csv"
MODEL_PATH = ROOT / "model" / "loan_model.joblib"

FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# Same encoding LabelEncoder produced in the notebook (alphabetical order).
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"No": 0, "Yes": 1}
# Target: 1 = Approved (clearer for probabilities than the notebook's 0/1 flip).
TARGET_MAP = {"Rejected": 0, "Approved": 1}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ("education", "self_employed", "loan_status"):
        df[col] = df[col].astype(str).str.strip()
    return df


def synthetic_dataset(n: int = 4269, seed: int = 42) -> pd.DataFrame:
    """Demo data with the same columns and ranges as the Kaggle dataset."""
    rng = np.random.default_rng(seed)
    round_to = lambda a, step: (np.round(a / step) * step).astype(int)

    income = round_to(rng.uniform(2e5, 9.9e6, n), 1e5)
    loan_amount = np.clip(round_to(income * rng.uniform(0.3, 4.5, n), 1e5), 3e5, 3.95e7)
    cibil = rng.integers(300, 901, n)
    res = round_to(income * rng.uniform(0, 3.0, n), 1e5)
    com = round_to(income * rng.uniform(0, 2.0, n), 1e5)
    lux = np.clip(round_to(income * rng.uniform(0.2, 4.0, n), 1e5), 3e5, None)
    bank = round_to(income * rng.uniform(0, 1.5, n), 1e5)

    ratio = loan_amount / income
    assets = (res + com + lux + bank) / loan_amount
    z = (cibil - 560) / 28 - 0.25 * (ratio - 2.5) + 0.15 * np.clip(assets - 2, -2, 3)
    p = 1 / (1 + np.exp(-z))
    approved = rng.random(n) < p

    return pd.DataFrame(
        {
            "loan_id": np.arange(1, n + 1),
            "no_of_dependents": rng.integers(0, 6, n),
            "education": rng.choice(["Graduate", "Not Graduate"], n),
            "self_employed": rng.choice(["No", "Yes"], n),
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": rng.integers(2, 21, n),
            "cibil_score": cibil,
            "residential_assets_value": res,
            "commercial_assets_value": com,
            "luxury_assets_value": lux,
            "bank_asset_value": bank,
            "loan_status": np.where(approved, "Approved", "Rejected"),
        }
    )


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["education"] = df["education"].map(EDUCATION_MAP)
    df["self_employed"] = df["self_employed"].map(SELF_EMPLOYED_MAP)
    y = df["loan_status"].map(TARGET_MAP)
    X = df[FEATURES]
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError(
            "Unexpected values found. Expected education in "
            f"{list(EDUCATION_MAP)}, self_employed in {list(SELF_EMPLOYED_MAP)}, "
            f"loan_status in {list(TARGET_MAP)}."
        )
    return X, y


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
    }


def main(csv: Path | None = None, demo: bool = False) -> dict:
    if demo or csv is None or not Path(csv).exists():
        source = "synthetic"
        df = synthetic_dataset()
        print("Using synthetic demo data (no CSV found).")
    else:
        source = "csv"
        df = load_csv(Path(csv))
        print(f"Loaded {len(df)} rows from {csv}")

    df = df.drop_duplicates()
    X, y = encode(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling keeps Logistic Regression stable on rupee-sized values.
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logistic.fit(X_train, y_train)

    forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    forest.fit(X_train, y_train)

    pred_lr = logistic.predict(X_test)
    pred_rf = forest.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_rf).ravel()

    bundle = {
        "model": forest,
        "features": FEATURES,
        "baseline": X_train.median().to_dict(),
        "ranges": {
            c: [float(X[c].min()), float(X[c].max())]
            for c in ("income_annum", "loan_amount", "loan_term", "cibil_score")
        },
        "meta": {
            "data_source": source,
            "rows": int(len(df)),
            "approval_rate": round(float(y.mean()), 4),
            "test_size": int(len(y_test)),
            "selected_model": "Random Forest",
            "metrics": {
                "Logistic Regression": scores(y_test, pred_lr),
                "Random Forest": scores(y_test, pred_rf),
            },
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "feature_importance": {
                f: round(float(v), 4)
                for f, v in sorted(
                    zip(FEATURES, forest.feature_importances_), key=lambda kv: -kv[1]
                )
            },
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    for name, m in bundle["meta"]["metrics"].items():
        print(f"{name:<20} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print(f"Saved model to {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--demo", action="store_true", help="use synthetic data")
    args = parser.parse_args()
    main(args.csv, args.demo)
