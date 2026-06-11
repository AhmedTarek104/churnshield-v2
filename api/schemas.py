"""Pydantic schemas for ChurnShield API."""
from typing import List
from pydantic import BaseModel, Field, field_validator


class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    Geography: str = Field(..., description="France, Germany, or Spain")
    Gender: str = Field(..., description="Male or Female")
    Age: int = Field(..., ge=18, le=95, description="Customer age (18-95)")
    Tenure: int = Field(..., ge=0, le=10, description="Years as customer (0-10)")
    Balance: float = Field(..., ge=0, description="Account balance (>= 0)")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products (1-4)")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0/1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0/1)")
    EstimatedSalary: float = Field(..., gt=0, description="Estimated annual salary (> 0)")

    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, v: str) -> str:
        if v not in ("France", "Germany", "Spain"):
            raise ValueError("Geography must be France, Germany, or Spain")
        return v

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ("Male", "Female"):
            raise ValueError("Gender must be Male or Female")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "CreditScore": 600,
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 125000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 70000.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    retention_value_eur: float
    top_risk_factors: List[str]
    recommended_action: str
    model_version: str
    threshold_used: float
    business_optimal: bool


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: dict
