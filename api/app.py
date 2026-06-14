"""FastAPI application for churn prediction and retention scripts."""

import json
import logging
import os
import pathlib
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field, ConfigDict
from src.feature_engineering import engineer_features_inference

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for log aggregators (Datadog, ELK).
    Includes all extra context fields and exception tracebacks."""

    BASE_KEYS = frozenset({
        "timestamp", "level", "logger", "message", "module", "line",
    })

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in self.BASE_KEYS and not key.startswith("_"):
                try:
                    data[key] = value
                except TypeError:
                    data[key] = str(value)
        if record.exc_info and record.exc_info[1] is not None:
            data["exception"] = str(record.exc_info[1])
        return json.dumps(data)


logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.handlers.clear()
logger.addHandler(_handler)
logger.propagate = False

_config_path = ROOT / "config.yaml"
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)

_model_path = ROOT / _cfg["model"]["save_path"]

_threshold = 0.5
try:
    _threshold = float(_cfg["model"]["threshold"])
except (KeyError, TypeError, ValueError):
    pass

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _model_path.exists():
        logger.error("MODEL_NOT_FOUND", extra={"path": str(_model_path)})
        raise RuntimeError(
            f"Model artifact not found at {_model_path}. "
            "Run 'python src/train.py' first."
        )
    artifact = joblib.load(_model_path)
    app.state.model = artifact["pipeline"]
    app.state.expected_features = getattr(
        app.state.model, "feature_name_", None
    )
    logger.info("MODEL_LOADED", extra={"path": str(_model_path)})
    if app.state.expected_features is not None:
        logger.info(
            "EXPECTED_FEATURES",
            extra={"count": len(app.state.expected_features)},
        )
    yield
    del app.state.model
    del app.state.expected_features
    logger.info("MODEL_RELEASED")


app = FastAPI(
    title="Enterprise Churn Engine",
    description="Production churn prediction API powered by Polars + LightGBM (ML) and "
    "Groq. Evaluates customer churn risk and generates "
    "retention scripts.",
    version="v1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000)
    logger.info(
        "REQUEST_COMPLETE",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": elapsed,
        },
    )
    return response


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(strict=False)
    Gender: Optional[str] = Field(None)
    SeniorCitizen: Optional[int] = Field(None)
    Partner: Optional[int] = Field(None)
    Dependents: Optional[int] = Field(None)
    tenure: Optional[int] = Field(None)
    PhoneService: Optional[int] = Field(None)
    MultipleLines: Optional[int] = Field(None)
    InternetService: Optional[int] = Field(None)
    OnlineSecurity: Optional[int] = Field(None)
    OnlineBackup: Optional[int] = Field(None)
    DeviceProtection: Optional[int] = Field(None)
    TechSupport: Optional[int] = Field(None)
    StreamingTV: Optional[int] = Field(None)
    StreamingMovies: Optional[int] = Field(None)
    Contract: Optional[str] = Field(None)
    PaperlessBilling: Optional[int] = Field(None)
    PaymentMethod: Optional[str] = Field(None)
    MonthlyCharges: Optional[float] = Field(None)
    TotalCharges: Optional[float] = Field(None)
    Married: Optional[int] = Field(None)
    NumberOfDependents: Optional[int] = Field(None)
    NumberOfReferrals: Optional[int] = Field(None)
    SatisfactionScore: Optional[int] = Field(None)
    InternetType: Optional[str] = Field(None)
    Offer: Optional[str] = Field(None)
    Age: Optional[int] = Field(None)
    AvgMonthlyGBDownload: Optional[int] = Field(None)
    AvgMonthlyLongDistanceCharges: Optional[float] = Field(None)
    CLTV: Optional[int] = Field(None)
    Under30: Optional[int] = Field(None)
    UnlimitedData: Optional[int] = Field(None)
    StreamingMusic: Optional[int] = Field(None)
    ReferredAFriend: Optional[int] = Field(None)
    TotalRefunds: Optional[float] = Field(None)
    TotalExtraDataCharges: Optional[int] = Field(None)
    TotalLongDistanceCharges: Optional[float] = Field(None)
    TotalRevenue: Optional[float] = Field(None)


class ChurnResponse(BaseModel):
    prediction: int = Field(..., description="1 = Churned, 0 = Stayed")
    churn_probability: float = Field(..., description="Model confidence score [0, 1]")
    retention_risk: str = Field(..., description="High / Medium / Low risk tier")


class BatchChurnResponse(BaseModel):
    results: list[ChurnResponse]
    total_records: int
    high_risk_count: int


class RetentionScriptRequest(BaseModel):
    risk_level: str = Field(
        ...,
        description="Risk tier returned by the prediction endpoint.",
        json_schema_extra={"example": "High"},
    )
    reasons: str = Field(
        ...,
        description="Key churn reasons.",
        json_schema_extra={"example": "Customer cited billing confusion and lack of usage."},
    )


class RetentionScriptResponse(BaseModel):
    script: str = Field(
        ...,
        description="A 2-sentence retention script for the customer service agent.",
    )


_engineer_features = engineer_features_inference


def _classify(proba: float) -> dict:
    return {
        "prediction": int(proba >= _threshold),
        "churn_probability": round(proba, 4),
        "retention_risk": "High" if proba >= 0.70 else "Medium" if proba >= 0.40 else "Low",
    }


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Customer Churn Prediction API. Visit /docs for Swagger UI."}


@app.get("/health", tags=["Health"])
def health_check(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "model"),
        "model_path": str(_model_path),
    }


@app.post("/predict", response_model=ChurnResponse, tags=["Machine Learning"])
def predict_endpoint(customer: CustomerFeatures, request: Request):
    try:
        pipeline = request.app.state.model
        expected = request.app.state.expected_features

        record = customer.model_dump(by_alias=False, exclude_none=False)
        record = {k: v for k, v in record.items() if v is not None}

        df = _engineer_features(pd.DataFrame([record]))

        if expected is not None:
            for col in expected:
                if col not in df.columns:
                    df[col] = 0
            df = df[[c for c in expected if c in df.columns]]

        proba = float(pipeline.predict_proba(df)[0][1])
        return ChurnResponse(**_classify(proba))

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("PREDICTION_ERROR", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchChurnResponse, tags=["Machine Learning"])
def predict_batch_endpoint(customers: list[CustomerFeatures], request: Request):
    if not customers:
        raise HTTPException(status_code=422, detail="Batch cannot be empty.")
    try:
        pipeline = request.app.state.model
        expected = request.app.state.expected_features

        records = [
            {k: v for k, v in c.model_dump(by_alias=False, exclude_none=False).items()
             if v is not None}
            for c in customers
        ]
        df = _engineer_features(pd.DataFrame(records))

        if expected is not None:
            for col in expected:
                if col not in df.columns:
                    df[col] = 0
            df = df[[c for c in expected if c in df.columns]]

        probas = pipeline.predict_proba(df)[:, 1]
        results = [ChurnResponse(**_classify(float(p))) for p in probas]
        high_risk = sum(1 for r in results if r.retention_risk == "High")
        return BatchChurnResponse(
            results=results,
            total_records=len(results),
            high_risk_count=high_risk,
        )

    except Exception as e:
        logger.exception("BATCH_PREDICTION_ERROR", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post(
    "/generate_retention_script",
    response_model=RetentionScriptResponse,
    tags=["Generative AI"],
)
def generate_retention_script(req: RetentionScriptRequest):
    prompt = (
        f"Write a 2-sentence retention script for a customer service agent. "
        f"The customer has a {req.risk_level} churn risk. "
        f"Key reasons: {req.reasons}. "
        f"Keep it concise and actionable."
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
        )
        script = "[Generated by Llama-3] " + response.choices[0].message.content.strip()
    except Exception:
        script = (
            "[Fallback Script] "
            "We value you as a customer. "
            "Let me review your account and find a solution that works for you."
        )
    return RetentionScriptResponse(script=script)
