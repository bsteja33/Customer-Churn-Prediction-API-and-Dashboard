"""
api/app.py
----------
FastAPI application exposing a /predict endpoint for real-time churn inference.

Startup:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /          → Health check
  GET  /health    → JSON health status
  POST /predict   → Single-customer churn prediction
  POST /predict/batch → Batch churn prediction (list of customers)
"""

import logging
import pathlib
import sys
from contextlib import asynccontextmanager

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

# Resolve project root
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

# Load API config
_config_path = ROOT / "config.yaml"
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)

api_cfg = _cfg.get("api", {})
_model_path = ROOT / _cfg["model"]["save_path"]

# Resolve prediction threshold from config
try:
    _threshold = float(_cfg.get("model", {}).get("threshold", 0.5))
except (TypeError, ValueError):
    _threshold = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the model artifact into RAM at server startup."""
    if not _model_path.exists():
        logger.error("MODEL_NOT_FOUND: %s", _model_path)
        raise RuntimeError(
            f"\n{'='*60}\n"
            "CRITICAL: Model artifact not found!\n"
            f"Expected location: {_model_path}\n"
            "Please run the training pipeline first:\n"
            "  python src/train.py\n"
            f"{'='*60}"
        )
    app.state.model = joblib.load(_model_path)
    logger.info("MODEL_PRELOADED_INTO_RAM: %s", _model_path)
    yield
    # Cleanup on shutdown
    del app.state.model
    logger.info("MODEL_RELEASED")


app = FastAPI(
    title=api_cfg.get("title", "Customer Churn Prediction API"),
    description=api_cfg.get("description", "Predict telecom customer churn."),
    version=api_cfg.get("version", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response Schemas

class CustomerFeatures(BaseModel):
    """
    Represents a single customer's raw feature set.
    All fields mirror the original dataset columns (post-cleaning).
    Optional fields default to None; the model pipeline handles imputation.
    """
    model_config = ConfigDict(strict=False, populate_by_name=True)

    Gender: Optional[str] = Field(None, json_schema_extra={"example": "Female"})
    Married: Optional[str] = Field(None, json_schema_extra={"example": "Yes"})
    Number_of_Dependents: Optional[int] = Field(None, alias="Number of Dependents", json_schema_extra={"example": 0})
    Number_of_Referrals: Optional[int] = Field(None, alias="Number of Referrals", json_schema_extra={"example": 2})
    Tenure_in_Months: Optional[int] = Field(None, alias="Tenure in Months", json_schema_extra={"example": 9})
    Offer: Optional[str] = Field(None, json_schema_extra={"example": "None"})
    Phone_Service: Optional[str] = Field(None, alias="Phone Service", json_schema_extra={"example": "Yes"})
    Avg_Monthly_Long_Distance_Charges: Optional[float] = Field(
        None, alias="Avg Monthly Long Distance Charges", json_schema_extra={"example": 42.39}
    )
    Multiple_Lines: Optional[str] = Field(None, alias="Multiple Lines", json_schema_extra={"example": "No"})
    Internet_Service: Optional[str] = Field(None, alias="Internet Service", json_schema_extra={"example": "Yes"})
    Internet_Type: Optional[str] = Field(None, alias="Internet Type", json_schema_extra={"example": "Cable"})
    Avg_Monthly_GB_Download: Optional[float] = Field(
        None, alias="Avg Monthly GB Download", json_schema_extra={"example": 16.0}
    )
    Online_Security: Optional[str] = Field(None, alias="Online Security", json_schema_extra={"example": "No"})
    Online_Backup: Optional[str] = Field(None, alias="Online Backup", json_schema_extra={"example": "Yes"})
    Device_Protection_Plan: Optional[str] = Field(
        None, alias="Device Protection Plan", json_schema_extra={"example": "No"}
    )
    Premium_Tech_Support: Optional[str] = Field(
        None, alias="Premium Tech Support", json_schema_extra={"example": "Yes"}
    )
    Streaming_TV: Optional[str] = Field(None, alias="Streaming TV", json_schema_extra={"example": "Yes"})
    Streaming_Movies: Optional[str] = Field(None, alias="Streaming Movies", json_schema_extra={"example": "No"})
    Streaming_Music: Optional[str] = Field(None, alias="Streaming Music", json_schema_extra={"example": "No"})
    Unlimited_Data: Optional[str] = Field(None, alias="Unlimited Data", json_schema_extra={"example": "Yes"})
    Contract: Optional[str] = Field(None, json_schema_extra={"example": "One Year"})
    Paperless_Billing: Optional[str] = Field(None, alias="Paperless Billing", json_schema_extra={"example": "Yes"})
    Payment_Method: Optional[str] = Field(None, alias="Payment Method", json_schema_extra={"example": "Credit Card"})
    Monthly_Charge: Optional[float] = Field(None, alias="Monthly Charge", json_schema_extra={"example": 65.6})
    Total_Charges: Optional[float] = Field(None, alias="Total Charges", json_schema_extra={"example": 593.3})
    Total_Refunds: Optional[float] = Field(None, alias="Total Refunds", json_schema_extra={"example": 0.0})
    Total_Extra_Data_Charges: Optional[float] = Field(
        None, alias="Total Extra Data Charges", json_schema_extra={"example": 0.0}
    )
    Total_Long_Distance_Charges: Optional[float] = Field(
        None, alias="Total Long Distance Charges", json_schema_extra={"example": 381.51}
    )
    Total_Revenue: Optional[float] = Field(None, alias="Total Revenue", json_schema_extra={"example": 974.81})
    Age: Optional[int] = Field(None, json_schema_extra={"example": 37})


class ChurnResponse(BaseModel):
    prediction: int = Field(..., description="1 = Churned, 0 = Stayed")
    churn_probability: float = Field(..., description="Model confidence score [0, 1]")
    retention_risk: str = Field(..., description="High / Medium / Low risk tier")


class BatchChurnResponse(BaseModel):
    results: list[ChurnResponse]
    total_records: int
    high_risk_count: int


# Endpoints

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate feature engineering for inference without disk I/O."""
    if "Total Revenue" in df.columns and "Tenure in Months" in df.columns:
        df["Revenue_per_Tenure"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Revenue"] / df["Tenure in Months"]
        )
    if "Total Charges" in df.columns and "Tenure in Months" in df.columns:
        df["Charges_per_Month"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Charges"] / df["Tenure in Months"]
        )
    service_cols = [
        c for c in df.columns
        if any(k in c for k in ["Security", "Backup", "Protection", "Tech Support", "Streaming"])
    ]
    if service_cols:
        df["Total_Services"] = (df[service_cols] == "Yes").sum(axis=1)
    else:
        df["Total_Services"] = 0
    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"], bins=[0, 30, 50, 70, 120],
            labels=["Young", "Adult", "Senior", "Elderly"]
        ).astype(str)
        df = df.drop(columns=["Age"])
    return df


def _classify(proba: float) -> dict:
    """Build a response dict from a churn probability."""
    return {
        "prediction": int(proba >= _threshold),
        "churn_probability": round(proba, 4),
        "retention_risk": "High" if proba >= 0.70 else "Medium" if proba >= 0.40 else "Low",
    }


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Customer Churn Prediction API — visit /docs for Swagger UI"}


@app.get("/health", tags=["Health"])
def health_check(request: Request):
    """Returns API health status and model availability."""
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "model"),
        "model_path": str(_model_path),
    }


@app.post("/predict", response_model=ChurnResponse, tags=["Inference"])
def predict_endpoint(customer: CustomerFeatures, request: Request):
    """
    Predict churn probability for a single customer.

    Accepts raw customer feature JSON and returns:
    - **prediction**: 1 (will churn) or 0 (will stay)
    - **churn_probability**: model confidence [0.0 – 1.0]
    - **retention_risk**: High (>=70%) / Medium (40-70%) / Low (<40%)
    """
    try:
        artifact = request.app.state.model
        pipeline = artifact["pipeline"]

        record = customer.model_dump(by_alias=True, exclude_none=False)
        record = {k: v for k, v in record.items() if v is not None}

        df = _engineer_features(pd.DataFrame([record]))
        proba = float(pipeline.predict_proba(df)[0][1])
        return ChurnResponse(**_classify(proba))

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("PREDICTION_ERROR")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchChurnResponse, tags=["Inference"])
def predict_batch_endpoint(customers: list[CustomerFeatures], request: Request):
    """
    Predict churn probability for a batch of customers.

    Returns individual results for each record plus aggregate risk counts.
    """
    if not customers:
        raise HTTPException(status_code=422, detail="Batch cannot be empty.")
    try:
        artifact = request.app.state.model
        pipeline = artifact["pipeline"]

        records = [
            {k: v for k, v in c.model_dump(by_alias=True, exclude_none=False).items()
             if v is not None}
            for c in customers
        ]
        df = _engineer_features(pd.DataFrame(records))
        probas = pipeline.predict_proba(df)[:, 1]

        results = [ChurnResponse(**_classify(float(p))) for p in probas]
        high_risk = sum(1 for r in results if r.retention_risk == "High")

        return BatchChurnResponse(
            results=results,
            total_records=len(results),
            high_risk_count=high_risk,
        )
    except Exception as e:
        logger.exception("BATCH_PREDICTION_ERROR")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
