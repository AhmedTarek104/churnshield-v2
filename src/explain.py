"""SHAP explainability, feature importance, retention action recommender."""
import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent
PROC_DIR    = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGS_DIR    = OUTPUTS_DIR / "figures"

RISK_LABELS = {
    "inactive_high_balance": "Inactive account with high balance",
    "is_zero_balance": "Zero account balance",
    "engagement_score": "Low product engagement",
    "age_group": "Customer age above 45",
    "NumOfProducts": "Too many products (overexposed)",
    "tenure_stability": "Short customer relationship",
    "balance_salary_ratio": "Disproportionate balance vs income",
    "Geography": "German market (higher churn segment)",
    "IsActiveMember": "Inactive member",
    "Age": "Customer age risk factor",
    "Balance": "Balance level risk",
    "CreditScore": "Credit score risk",
}


def get_action(customer: pd.Series) -> str:
    if customer.get("inactive_high_balance", 0) == 1:
        return "Priority call: offer wealth management consultation"
    if customer.get("is_zero_balance", 0) == 1:
        return "Offer savings account with bonus interest rate"
    if customer.get("engagement_score", 1.0) < 0.3:
        return "Send personalized product recommendation email"
    if customer.get("Age", 0) > 55:
        return "Assign dedicated senior relationship manager"
    if customer.get("NumOfProducts", 0) >= 3:
        return "Review product portfolio — simplify offering"
    return "Standard retention: loyalty reward or fee waiver"


def explain_customer(customer_series: pd.Series, model, explainer, feat_names: list) -> dict:
    x = customer_series[feat_names].values.reshape(1, -1)
    prob = model.predict_proba(x)[0, 1]

    shap_values = explainer(x)
    if hasattr(shap_values, "values"):
        sv = shap_values.values[0]
    else:
        sv = shap_values[0]

    # Top 3 risk factors
    indices = np.argsort(np.abs(sv))[::-1][:3]
    top_factors = []
    for i in indices:
        fname = feat_names[i]
        direction = "increases" if sv[i] > 0 else "decreases"
        label = RISK_LABELS.get(fname, fname.replace("_", " ").title())
        top_factors.append(f"{label} ({direction} risk, magnitude={abs(sv[i]):.3f})")

    action = get_action(customer_series)
    return {
        "churn_probability": round(float(prob), 4),
        "top_3_risk_factors": top_factors,
        "recommended_action": action,
    }


def build_explainer(model, X_sample: np.ndarray):
    model_name_lower = type(model).__name__.lower()
    inner = model
    # Unwrap sklearn pipeline
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("clf", model)

    if "logistic" in model_name_lower or (hasattr(inner, "coef_")):
        return shap.LinearExplainer(inner, X_sample, feature_perturbation="interventional")
    else:
        try:
            return shap.TreeExplainer(inner)
        except Exception:
            return shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 100))


if __name__ == "__main__":
    os.makedirs(FIGS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()
    with open(os.path.join(PROC_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)

    test_df = pd.read_csv(os.path.join(PROC_DIR, "test_with_predictions.csv"))
    X_test = test_df[feat_names].values

    print(f"Computing SHAP values for {model_name}…")

    # Use inner model for tree-based SHAP
    inner_model = model
    if hasattr(model, "named_steps"):
        inner_model = model.named_steps.get("clf", model)
        scaler = model.named_steps.get("scaler", None)
        X_shap = scaler.transform(X_test) if scaler else X_test
    else:
        X_shap = X_test

    try:
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer(X_shap)
    except Exception:
        print("TreeExplainer failed, using KernelExplainer (slower)…")
        bg = shap.sample(X_shap, 100)
        explainer = shap.KernelExplainer(inner_model.predict_proba, bg)
        shap_values = explainer.shap_values(X_shap[:500])

    # Extract raw array
    if hasattr(shap_values, "values"):
        sv_array = shap_values.values
        if sv_array.ndim == 3:
            sv_array = sv_array[:, :, 1]
    else:
        sv_array = shap_values
        if isinstance(sv_array, list):
            sv_array = sv_array[1]

    mean_abs_shap = np.abs(sv_array).mean(axis=0)
    fi_df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(15)

    fi_df.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance.csv"), index=False)
    print("\nTop 15 features by mean |SHAP|:")
    print(fi_df.to_string(index=False))

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["feature"][::-1], fi_df["mean_abs_shap"][::-1], color="#CC0000")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance — SHAP (Global)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "shap_importance.png"), dpi=150)
    plt.close(fig)

    # Beeswarm
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        if hasattr(shap_values, "values"):
            shap_plot_vals = shap_values
            if shap_plot_vals.values.ndim == 3:
                shap_plot_vals = shap.Explanation(
                    values=shap_plot_vals.values[:, :, 1],
                    base_values=shap_plot_vals.base_values[:, 1] if shap_plot_vals.base_values.ndim > 1 else shap_plot_vals.base_values,
                    data=shap_plot_vals.data,
                    feature_names=feat_names,
                )
            shap.plots.beeswarm(shap_plot_vals, max_display=15, show=False)
        else:
            shap.summary_plot(sv_array, X_shap[:500], feature_names=feat_names,
                              max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Beeswarm plot skipped: {e}")

    # Save explainer (use general wrapper for inference)
    explainer_info = {
        "model_name": model_name,
        "feat_names": feat_names,
    }
    with open(os.path.join(MODELS_DIR, "shap_explainer.pkl"), "wb") as f:
        pickle.dump({"explainer": explainer, "explainer_info": explainer_info}, f)

    # Demo local explanation
    sample = test_df.iloc[0]
    result = explain_customer(sample, model, explainer, feat_names)
    print(f"\nSample local explanation:")
    print(f"  Churn probability : {result['churn_probability']:.2%}")
    print(f"  Risk factors      :")
    for r in result["top_3_risk_factors"]:
        print(f"    • {r}")
    print(f"  Action            : {result['recommended_action']}")

    print("\nEXPLAINABILITY COMPLETE")
