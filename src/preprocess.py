"""Stratified train/val/test split."""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROC_DIR, "features.csv"))
    print(f"Loaded {len(df):,} rows, churn rate {df['Exited'].mean():.2%}")

    with open(os.path.join(PROC_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    # Drop non-feature id columns for model use, keep them in splits for reference
    X = df[feature_names]
    y = df["Exited"]

    # 70 / 10 / 20 stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        df, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42  # 0.125 * 0.8 = 0.10
    )

    X_train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    X_val.to_csv(os.path.join(PROC_DIR, "val.csv"), index=False)
    X_test.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)

    for name, split in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        cr = split["Exited"].mean()
        print(f"  {name:<6}: {len(split):>5} rows | churn rate {cr:.2%}")

    print("\nPREPROCESSING COMPLETE")
