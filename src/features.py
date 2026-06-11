"""Feature engineering: builds all features including CLV estimate."""
import json
import os
import pickle

import numpy as np
import pandas as pd

RAW_FILE = r"G:\churnshield_v2\data\raw\Churn_Modelling.csv"
PROC_DIR = r"G:\churnshield_v2\data\processed"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Geography / Gender encoding
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gen_map = {"Female": 0, "Male": 1}
    out["Geography"] = out["Geography"].map(geo_map)
    out["Gender"] = out["Gender"].map(gen_map)

    # --- Engineered features ---
    out["balance_salary_ratio"] = out["Balance"] / (out["EstimatedSalary"] + 1)

    out["products_per_year"] = out["NumOfProducts"] / (out["Tenure"] + 1)

    out["age_group"] = pd.cut(
        out["Age"],
        bins=[0, 30, 45, 60, 200],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)

    out["engagement_score"] = (
        out["IsActiveMember"] * 0.5
        + out["HasCrCard"] * 0.2
        + np.minimum(out["NumOfProducts"] / 4, 1) * 0.3
    ).round(4)

    out["credit_score_band"] = pd.cut(
        out["CreditScore"],
        bins=[0, 579, 669, 739, 799, 900],
        labels=[0, 1, 2, 3, 4],
        right=True,
    ).astype(int)

    out["is_zero_balance"] = (out["Balance"] == 0).astype(int)

    out["high_value_customer"] = (
        (out["Balance"] > out["Balance"].quantile(0.75))
        & (out["EstimatedSalary"] > out["EstimatedSalary"].quantile(0.75))
    ).astype(int)

    out["tenure_stability"] = pd.cut(
        out["Tenure"],
        bins=[-1, 2, 5, 20],
        labels=[0, 1, 2],
        right=True,
    ).astype(int)

    out["age_tenure_ratio"] = out["Age"] / (out["Tenure"] + 1)

    out["inactive_high_balance"] = (
        (out["IsActiveMember"] == 0) & (out["Balance"] > 50_000)
    ).astype(int)

    # --- Synthetic CLV ---
    base_value = out["EstimatedSalary"] * 0.03
    product_multiplier = 1 + (out["NumOfProducts"] * 0.2)
    tenure_multiplier = 1 + (out["Tenure"] * 0.08)
    balance_bonus = out["Balance"] * 0.01 / 1000
    credit_bonus = (out["CreditScore"] - 300) / 550 * 0.3

    out["clv_estimated"] = (
        base_value * product_multiplier * tenure_multiplier + balance_bonus + credit_bonus
    ).clip(50, 5_000).round(2)

    return out


def print_correlations(df: pd.DataFrame, target: str = "Exited") -> None:
    engineered = [
        "balance_salary_ratio", "products_per_year", "age_group",
        "engagement_score", "credit_score_band", "is_zero_balance",
        "high_value_customer", "tenure_stability", "age_tenure_ratio",
        "inactive_high_balance", "clv_estimated",
    ]
    corrs = df[engineered + [target]].corr()[target].drop(target)
    print("\nFeature correlations with Exited:")
    for feat, val in corrs.sort_values(key=abs, ascending=False).items():
        print(f"  {feat:<30} {val:+.4f}")
    top5 = corrs.abs().nlargest(5)
    print("\nTop 5 most correlated features (|r|):")
    for feat, val in top5.items():
        print(f"  {feat:<30} {val:.4f}")


if __name__ == "__main__":
    os.makedirs(PROC_DIR, exist_ok=True)

    df_raw = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df_raw):,} rows")

    df_feat = build_features(df_raw)

    feature_cols = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary",
        "balance_salary_ratio", "products_per_year", "age_group",
        "engagement_score", "credit_score_band", "is_zero_balance",
        "high_value_customer", "tenure_stability", "age_tenure_ratio",
        "inactive_high_balance", "clv_estimated",
        "Exited",
    ]

    # Keep original IDs for downstream use
    id_cols = ["RowNumber", "CustomerId"] if "CustomerId" in df_feat.columns else []
    save_cols = id_cols + feature_cols

    df_out = df_feat[[c for c in save_cols if c in df_feat.columns]]
    df_out.to_csv(os.path.join(PROC_DIR, "features.csv"), index=False)

    feat_names = [c for c in feature_cols if c != "Exited"]
    with open(os.path.join(PROC_DIR, "feature_names.json"), "w") as f:
        json.dump(feat_names, f, indent=2)

    # Minimal encoders dict (mappings used during inference)
    encoders = {
        "geography_map": {"France": 0, "Germany": 1, "Spain": 2},
        "gender_map": {"Female": 0, "Male": 1},
    }
    with open(os.path.join(PROC_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    print(f"\nFeatures saved: {len(feat_names)} feature columns")
    print("Engineered features:")
    engineered = [
        "balance_salary_ratio", "products_per_year", "age_group",
        "engagement_score", "credit_score_band", "is_zero_balance",
        "high_value_customer", "tenure_stability", "age_tenure_ratio",
        "inactive_high_balance", "clv_estimated",
    ]
    for f in engineered:
        print(f"  + {f}")

    print_correlations(df_out)
    print("\nFEATURE ENGINEERING COMPLETE")
