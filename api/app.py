"""FastAPI application for churn prediction and retention scripts."""

import asyncio
import json
import logging
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.feature_engineering import engineer_features_inference
from src.config import MODEL_CONFIG

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

_model_path = ROOT / MODEL_CONFIG["save_path"]
_threshold = float(MODEL_CONFIG.get("threshold", 0.5))


# Groq initialization is lazy loaded in _generate_script
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _model_path.exists():
        logger.error("MODEL_NOT_FOUND", extra={"path": str(_model_path)})
        raise RuntimeError(
            f"Model artifact not found at {_model_path}. "
            "Run 'python src/train.py' first."
        )
    artifact = joblib.load(_model_path)
    pipeline = artifact.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "predict_proba"):
        raise RuntimeError(
            f"Model artifact at {_model_path} is corrupted: "
            "missing pipeline with predict_proba."
        )
    app.state.model = pipeline
    app.state.expected_features = getattr(
        pipeline, "feature_name_", None
    )

    executor = ThreadPoolExecutor(max_workers=4)
    app.state.executor = executor

    logger.info("MODEL_LOADED", extra={"path": str(_model_path)})
    if app.state.expected_features is not None:
        logger.info(
            "EXPECTED_FEATURES",
            extra={"count": len(app.state.expected_features)},
        )
    yield
    executor.shutdown(wait=False)
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
    model_config = ConfigDict(strict=True, extra="forbid")
    Gender: str | None = Field(None)
    SeniorCitizen: int | None = Field(None)
    Partner: int | None = Field(None)
    Dependents: int | None = Field(None)
    tenure: int | None = Field(None)
    PhoneService: int | None = Field(None)
    MultipleLines: int | None = Field(None)
    InternetService: int | None = Field(None)
    OnlineSecurity: int | None = Field(None)
    OnlineBackup: int | None = Field(None)
    DeviceProtection: int | None = Field(None)
    TechSupport: int | None = Field(None)
    StreamingTV: int | None = Field(None)
    StreamingMovies: int | None = Field(None)
    Contract: str | None = Field(None)
    PaperlessBilling: int | None = Field(None)
    PaymentMethod: str | None = Field(None)
    MonthlyCharges: float | None = Field(None)
    TotalCharges: float | None = Field(None)
    Married: int | None = Field(None)
    NumberOfDependents: int | None = Field(None)
    NumberOfReferrals: int | None = Field(None)
    SatisfactionScore: int | None = Field(None)
    InternetType: str | None = Field(None)
    Offer: str | None = Field(None)
    Age: int | None = Field(None)
    AvgMonthlyGBDownload: int | None = Field(None)
    AvgMonthlyLongDistanceCharges: float | None = Field(None)
    CLTV: int | None = Field(None)
    Under30: int | None = Field(None)
    UnlimitedData: int | None = Field(None)
    StreamingMusic: int | None = Field(None)
    ReferredAFriend: int | None = Field(None)
    TotalRefunds: float | None = Field(None)
    TotalExtraDataCharges: int | None = Field(None)
    TotalLongDistanceCharges: float | None = Field(None)
    TotalRevenue: float | None = Field(None)

    @field_validator("*")
    @classmethod
    def _reject_nan_infinity(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("NaN or Infinity is not allowed for float fields")
        return v


class ChurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: int = Field(..., description="1 = Churned, 0 = Stayed")
    churn_probability: float = Field(..., description="Model confidence score [0, 1]")
    retention_risk: str = Field(..., description="High / Medium / Low risk tier")


class BatchChurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[ChurnResponse]
    total_records: int
    high_risk_count: int


class RetentionScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
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


def _process_prediction(pipeline, expected, record):
    df = _engineer_features(pd.DataFrame([record]))
    if expected is not None:
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in expected if c in df.columns]]
    proba = float(pipeline.predict_proba(df)[0][1])
    return _classify(proba)


@app.post("/predict", response_model=ChurnResponse, tags=["Machine Learning"])
async def predict_endpoint(customer: CustomerFeatures, request: Request):
    try:
        pipeline = request.app.state.model
        expected = request.app.state.expected_features

        record = customer.model_dump(by_alias=False, exclude_none=False)
        record = {k: v for k, v in record.items() if v is not None}

        loop = asyncio.get_running_loop()
        result_dict = await loop.run_in_executor(
            request.app.state.executor, _process_prediction, pipeline, expected, record
        )
        return ChurnResponse(**result_dict)

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except TypeError as te:
        logger.exception("PREDICTION_TYPE_ERROR", extra={"error": str(te)})
        raise HTTPException(status_code=500, detail="Type evaluation failed during prediction.")
    except Exception as e:
        logger.exception("PREDICTION_ERROR", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")


def _process_batch_prediction(pipeline, expected, records):
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


@app.post("/predict/batch", response_model=BatchChurnResponse, tags=["Machine Learning"])
async def predict_batch_endpoint(customers: list[CustomerFeatures], request: Request):
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

        loop = asyncio.get_running_loop()
        batch_response = await loop.run_in_executor(
            request.app.state.executor, _process_batch_prediction, pipeline, expected, records
        )
        return batch_response

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("BATCH_PREDICTION_ERROR", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Batch prediction failed due to an internal error.")


def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def _generate_script(prompt: str) -> str:
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            timeout=10.0,
        )
        return "[Generated by Llama-3] " + response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM_GENERATION_FAILED", extra={"error": str(exc)})
        return (
            "[Fallback Script] "
            "We value you as a customer. "
            "Let me review your account and find a solution that works for you."
        )


@app.post(
    "/generate_retention_script",
    response_model=RetentionScriptResponse,
    tags=["Generative AI"],
)
async def generate_retention_script(request_payload: RetentionScriptRequest, request: Request):
    """Generate a retention script using Groq LLM."""
    prompt = (
        f"Write a 2-sentence retention script for a customer service agent. "
        f"The customer has a {request_payload.risk_level} churn risk. "
        f"Key reasons: {request_payload.reasons}. "
        f"Keep it concise and actionable."
    )
    loop = asyncio.get_running_loop()
    script = await loop.run_in_executor(request.app.state.executor, _generate_script, prompt)
    return RetentionScriptResponse(script=script)
