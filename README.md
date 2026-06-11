# ChurnShield v2

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![CatBoost](https://img.shields.io/badge/CatBoost-AUC_0.838-orange)
![Dash](https://img.shields.io/badge/Dash-2.17-red)
![MLflow](https://img.shields.io/badge/MLflow-2.11-blue)

## Business Problem

Banks lose 20–25% of customers annually to churn, each representing thousands of euros in lifetime value. Traditional reactive approaches catch churners after they've already left. ChurnShield v2 predicts churn 30 days in advance and tells relationship managers exactly who to call and why.

## Solution

A production-grade ML system that:
- Predicts individual churn probability with calibrated confidence
- Identifies the top 3 risk factors per customer using SHAP explainability
- Recommends specific retention actions (not just "call them")
- Quantifies exact revenue at risk per customer (CLV × probability × 30% retention rate)
- Optimizes outreach threshold for maximum net business value, not just accuracy

## Live URLs

- **Dashboard**: https://churnshield-v2.onrender.com/
- **API Docs**: coming soon (FastAPI /docs)

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | CatBoost |
| Test AUC | 0.838 |
| Test F1 | 0.585 |
| Catch Rate | 77.8% |
| Business-Optimal Threshold | 0.20 |
| F1-Optimal Threshold | 0.44 |
| Net Business Value | €457,613 |
| Avg Customer CLV | €4,011 |
| Customers Analyzed | 10,000 |

## Architecture

```
data/raw/           → Raw bank customer data (10,000 customers)
src/ingest.py       → Data ingestion (Kaggle or synthetic)
src/features.py     → 10+ engineered features + CLV estimation
src/preprocess.py   → Stratified 70/10/20 train/val/test split
src/train.py        → 5 models, MLflow tracking, best model selection
src/evaluate.py     → Full evaluation + business threshold optimization
src/explain.py      → SHAP global + local explainability
api/main.py         → FastAPI: /predict, /predict/batch, /health, /metrics
dashboard/app.py    → 5-tab Dash dashboard (dark theme)
```

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/ingest.py
python src/features.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/explain.py

# 3. Start the dashboard
python dashboard/app.py

# 4. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. (Optional) MLflow UI
mlflow ui --host 127.0.0.1 --port 5000
```

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 125000,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 70000
  }'
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | CatBoost, LightGBM, RandomForest, HistGradientBoosting, LogisticRegression |
| Explainability | SHAP (TreeExplainer) |
| Experiment Tracking | MLflow |
| API | FastAPI + Pydantic v2 |
| Dashboard | Plotly Dash + Bootstrap |
| Deployment | Render |
| Language | Python 3.13 |
