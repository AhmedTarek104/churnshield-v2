"""Data ingestion: download from Kaggle or generate synthetic data."""
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "Churn_Modelling.csv"


def try_kaggle_download() -> bool:
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "shubhammeshram579/bank-customer-churn-prediction",
                "-p", RAW_DIR, "--unzip"
            ],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(OUTPUT_FILE):
            print("Kaggle download succeeded.")
            return True
        print(f"Kaggle failed: {result.stderr.strip()}")
        return False
    except Exception as e:
        print(f"Kaggle unavailable: {e}")
        return False


def generate_synthetic_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    geography = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.50, 0.25, 0.25])
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])

    credit_score = rng.integers(300, 851, size=n)
    age = rng.integers(18, 93, size=n)
    tenure = rng.integers(0, 11, size=n)
    balance = np.where(
        rng.random(n) < 0.35, 0.0,
        rng.uniform(5_000, 250_000, size=n)
    )
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.50, 0.46, 0.03, 0.01])
    has_cr_card = rng.integers(0, 2, size=n)
    is_active = rng.integers(0, 2, size=n)
    salary = rng.uniform(10_000, 200_000, size=n)

    # Logistic model for realistic, learnable churn signal (~20% overall)
    # Baseline calibrated to yield ~20% mean p_churn given the effects below
    score = np.full(n, -4.5, dtype=float)
    score += (geography == "Germany") * 1.5
    score += (is_active == 0) * 1.3
    score += (age > 60).astype(float) * 1.0
    score += (age > 45).astype(float) * 0.5
    score += (balance > 50_000).astype(float) * (is_active == 0).astype(float) * 1.8
    score += (num_products >= 3).astype(float) * 0.8
    score += (tenure <= 2).astype(float) * 0.5
    score += (credit_score < 580).astype(float) * 0.4
    score += rng.normal(0, 0.4, n)  # mild residual noise

    p_churn = 1.0 / (1.0 + np.exp(-score))

    exited = (rng.random(n) < p_churn).astype(int)

    df = pd.DataFrame({
        "RowNumber": range(1, n + 1),
        "CustomerId": rng.integers(10_000_000, 20_000_000, size=n),
        "Surname": [f"Customer_{i}" for i in range(1, n + 1)],
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": np.round(balance, 2),
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": np.round(salary, 2),
        "Exited": exited,
    })
    return df


def validate(df: pd.DataFrame) -> None:
    assert len(df) == 10_000, f"Expected 10000 rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Missing values found"
    churn_rate = df["Exited"].mean()
    assert 0.15 <= churn_rate <= 0.25, f"Churn rate {churn_rate:.2%} outside 15-25%"
    print(f"\nValidation passed:")
    print(f"  Rows         : {len(df):,}")
    print(f"  Churn rate   : {churn_rate:.2%}")
    print(f"  Missing vals : 0")
    print("\nSummary statistics:")
    print(df.describe().to_string())


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)

    if not try_kaggle_download():
        print("Generating synthetic data…")
        df = generate_synthetic_data()
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")

    df = pd.read_csv(OUTPUT_FILE)
    print(f"\nLoaded {len(df):,} rows from {OUTPUT_FILE}")
    validate(df)
    print("\nINGESTION COMPLETE")
