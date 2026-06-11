"""ChurnShield v2 — Production FastAPI."""
import json
import os
import pickle
import sys
import warnings
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import BatchPredictionResponse, CustomerInput, PredictionResponse

MODELS_DIR = r"G:\churnshield_v2\models"
PROC_DIR = r"G:\churnshield_v2\data\processed"
OUTPUTS_DIR = r"G:\churnshield_v2\outputs"

API_VERSION = "2.0.0"

# Global state
_state: dict = {}


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


def customer_to_features(c: CustomerInput) -> pd.Series:
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gen_map = {"Female": 0, "Male": 1}

    geo = geo_map[c.Geography]
    gen = gen_map[c.Gender]

    balance_salary_ratio = c.Balance / (c.EstimatedSalary + 1)
    products_per_year = c.NumOfProducts / (c.Tenure + 1)

    age = c.Age
    if age <= 30:
        age_group = 0
    elif age <= 45:
        age_group = 1
    elif age <= 60:
        age_group = 2
    else:
        age_group = 3

    engagement_score = round(
        c.IsActiveMember * 0.5 + c.HasCrCard * 0.2 + min(c.NumOfProducts / 4, 1) * 0.3, 4
    )

    cs = c.CreditScore
    if cs < 580:
        credit_score_band = 0
    elif cs < 670:
        credit_score_band = 1
    elif cs < 740:
        credit_score_band = 2
    elif cs < 800:
        credit_score_band = 3
    else:
        credit_score_band = 4

    is_zero_balance = int(c.Balance == 0)

    tenure_stab = 0 if c.Tenure <= 2 else (1 if c.Tenure <= 5 else 2)

    age_tenure_ratio = c.Age / (c.Tenure + 1)

    inactive_high_balance = int(c.IsActiveMember == 0 and c.Balance > 50_000)

    base_value = c.EstimatedSalary * 0.05
    product_mult = 1 + c.NumOfProducts * 0.3
    tenure_mult = 1 + c.Tenure * 0.1
    balance_bonus = c.Balance * 0.02 / 1000
    credit_bonus = (c.CreditScore - 300) / 550 * 0.5
    clv = float(
        np.clip(base_value * product_mult * tenure_mult + balance_bonus + credit_bonus, 100, 10_000)
    )

    return pd.Series({
        "CreditScore": c.CreditScore,
        "Geography": geo,
        "Gender": gen,
        "Age": c.Age,
        "Tenure": c.Tenure,
        "Balance": c.Balance,
        "NumOfProducts": c.NumOfProducts,
        "HasCrCard": c.HasCrCard,
        "IsActiveMember": c.IsActiveMember,
        "EstimatedSalary": c.EstimatedSalary,
        "balance_salary_ratio": balance_salary_ratio,
        "products_per_year": products_per_year,
        "age_group": age_group,
        "engagement_score": engagement_score,
        "credit_score_band": credit_score_band,
        "is_zero_balance": is_zero_balance,
        "high_value_customer": 0,
        "tenure_stability": tenure_stab,
        "age_tenure_ratio": age_tenure_ratio,
        "inactive_high_balance": inactive_high_balance,
        "clv_estimated": round(clv, 2),
        # raw for action recommender
        "_age": c.Age,
        "_inactive_high_balance": inactive_high_balance,
        "_is_zero_balance": is_zero_balance,
        "_engagement_score": engagement_score,
        "_num_products": c.NumOfProducts,
    })


def get_action(feat: pd.Series) -> str:
    if feat.get("_inactive_high_balance", 0) == 1:
        return "Priority call: offer wealth management consultation"
    if feat.get("_is_zero_balance", 0) == 1:
        return "Offer savings account with bonus interest rate"
    if feat.get("_engagement_score", 1.0) < 0.3:
        return "Send personalized product recommendation email"
    if feat.get("_age", 0) > 55:
        return "Assign dedicated senior relationship manager"
    if feat.get("_num_products", 0) >= 3:
        return "Review product portfolio — simplify offering"
    return "Standard retention: loyalty reward or fee waiver"


def get_shap_factors(feat_series: pd.Series, feat_names: list) -> list:
    try:
        explainer_bundle = _state.get("explainer_bundle")
        model = _state["model"]
        if explainer_bundle is None:
            return ["SHAP not available"]

        explainer = explainer_bundle["explainer"]
        x = feat_series[feat_names].values.reshape(1, -1)

        inner = model
        if hasattr(model, "named_steps"):
            inner = model.named_steps.get("clf", model)
            scaler = model.named_steps.get("scaler", None)
            x_shap = scaler.transform(x) if scaler else x
        else:
            x_shap = x

        shap_vals = explainer(x_shap)
        if hasattr(shap_vals, "values"):
            sv = shap_vals.values[0]
            if sv.ndim == 2:
                sv = sv[:, 1]
        else:
            sv = shap_vals
            if isinstance(sv, list):
                sv = sv[1][0]
            else:
                sv = sv[0]

        indices = np.argsort(np.abs(sv))[::-1][:3]
        factors = []
        for i in indices:
            fname = feat_names[i]
            direction = "increases" if sv[i] > 0 else "decreases"
            label = RISK_LABELS.get(fname, fname.replace("_", " ").title())
            factors.append(f"{label} ({direction} risk)")
        return factors
    except Exception as e:
        return [f"Factor analysis error: {str(e)[:50]}"]


def make_prediction(c: CustomerInput) -> PredictionResponse:
    model = _state["model"]
    feat_names = _state["feat_names"]
    thresholds = _state["thresholds"]
    model_name = _state["model_name"]

    feat = customer_to_features(c)
    x = feat[feat_names].values.reshape(1, -1)

    prob = float(model.predict_proba(x)[0, 1])
    biz_thresh = thresholds["business_optimal"]
    pred = int(prob >= biz_thresh)

    risk = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.35 else "LOW")
    clv = float(feat["clv_estimated"])
    retention_value = round(clv * prob * 0.30, 2)

    factors = get_shap_factors(feat, feat_names)
    action = get_action(feat)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_level=risk,
        retention_value_eur=retention_value,
        top_risk_factors=factors,
        recommended_action=action,
        model_version=f"{model_name}-{API_VERSION}",
        threshold_used=biz_thresh,
        business_optimal=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    try:
        with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
            _state["model"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
            _state["model_name"] = f.read().strip()
        with open(os.path.join(PROC_DIR, "feature_names.json")) as f:
            _state["feat_names"] = json.load(f)
        with open(os.path.join(MODELS_DIR, "thresholds.json")) as f:
            _state["thresholds"] = json.load(f)
        try:
            with open(os.path.join(MODELS_DIR, "shap_explainer.pkl"), "rb") as f:
                _state["explainer_bundle"] = pickle.load(f)
        except Exception:
            _state["explainer_bundle"] = None
        try:
            with open(os.path.join(OUTPUTS_DIR, "metrics.json")) as f:
                _state["metrics"] = json.load(f)
        except Exception:
            _state["metrics"] = {}
        print(f"Model loaded: {_state['model_name']}")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        _state["model"] = None
    yield


app = FastAPI(
    title="ChurnShield v2",
    description="Production-grade customer churn prediction API",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _state.get("model") is not None,
        "model_name": _state.get("model_name", "unknown"),
        "api_version": API_VERSION,
    }


@app.get("/model/info")
def model_info():
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    m = _state.get("metrics", {})
    t = _state.get("thresholds", {})
    return {
        "model_name": _state["model_name"],
        "test_auc": m.get("test_auc"),
        "test_f1": m.get("test_f1"),
        "f1_optimal_threshold": t.get("f1_optimal"),
        "business_optimal_threshold": t.get("business_optimal"),
        "n_features": len(_state.get("feat_names", [])),
        "api_version": API_VERSION,
    }


@app.get("/metrics")
def metrics():
    if not _state.get("metrics"):
        raise HTTPException(status_code=404, detail="Metrics not available")
    return _state["metrics"]


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return make_prediction(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(customers: List[CustomerInput]):
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Max 500 customers per batch")

    predictions = [make_prediction(c) for c in customers]
    high = sum(1 for p in predictions if p.risk_level == "HIGH")
    medium = sum(1 for p in predictions if p.risk_level == "MEDIUM")
    total_rv = sum(p.retention_value_eur for p in predictions)

    return BatchPredictionResponse(
        predictions=predictions,
        summary={
            "total": len(predictions),
            "high_risk": high,
            "medium_risk": medium,
            "low_risk": len(predictions) - high - medium,
            "total_retention_value_at_risk": round(total_rv, 2),
        },
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
