"""
predict.py
----------
Inference module: loads the persisted model artifact and runs prediction
on a single customer record or a batch of records.

Can be used:
  - Directly as a Python module (imported by the FastAPI app)
  - As a CLI tool for quick batch predictions from a CSV

Usage (CLI):
  python src/predict.py --input '{"Tenure in Months": 12, "Monthly Charge": 70, ...}'
  python src/predict.py --csv path/to/batch.csv
"""

import argparse
import json
import logging
import pathlib

import joblib
import numpy as np
import pandas as pd

from typing import Any
from src.data_preprocessing import load_config

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = ROOT / "models" / "churn_model.pkl"

# Configure logger.
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    _log_dir = ROOT / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(_log_dir / "predictions.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Module cache.
_ARTIFACT_CACHE: dict[str, Any] | None = None


def _load_artifact(model_path: pathlib.Path) -> dict:
    """Load and cache the model artifact dict."""
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please run 'python src/train.py' first."
            )
        _ARTIFACT_CACHE = joblib.load(model_path)
        logger.info("MODEL_LOADED: %s", model_path)
    return _ARTIFACT_CACHE


def predict_single(
    customer: dict,
    model_path: pathlib.Path = _DEFAULT_MODEL_PATH,
) -> dict:
    """
    Predict churn probability for a single customer record.

    Parameters
    ----------
    customer : dict
        Raw customer feature dictionary (same schema as the training CSV,
        excluding identifier / post-hoc columns).
    model_path : pathlib.Path
        Path to the saved model artifact.

    Returns
    -------
    dict with keys:
        prediction        : int   — 1 (Churned) or 0 (Stayed)
        churn_probability : float — model confidence [0, 1]
        retention_risk    : str   — "High" / "Medium" / "Low"
    """
    artifact = _load_artifact(model_path)
    pipeline = artifact["pipeline"]

    # DataFrame conversion.
    df = pd.DataFrame([customer])

    # Engineer features.
    # Revenue_per_Tenure
    if "Total Revenue" in df.columns and "Tenure in Months" in df.columns:
        df["Revenue_per_Tenure"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Revenue"] / df["Tenure in Months"]
        )

    # Charges_per_Month
    if "Total Charges" in df.columns and "Tenure in Months" in df.columns:
        df["Charges_per_Month"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Charges"] / df["Tenure in Months"]
        )

    # Total_Services (dynamic detection)
    service_cols = [
        col for col in df.columns
        if any(k in col for k in ["Security", "Backup", "Protection", "Tech Support", "Streaming"])
    ]
    if service_cols:
        service_count = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        for col in service_cols:
            service_count += (df[col] == "Yes").astype(int)
        df["Total_Services"] = service_count
    else:
        df["Total_Services"] = 0

    # Age_Group
    if "Age" in df.columns:
        df["Age_Group"] = (
            pd.cut(
                df["Age"],
                bins=[0, 30, 50, 70, 120],
                labels=["Young", "Adult", "Senior", "Elderly"],
            ).astype(str)
        )
        df = df.drop(columns=["Age"])

    # Config threshold.
    try:
        config_path = ROOT / "config.yaml"
        cfg = load_config(config_path)
        threshold = float(cfg.get("model", {}).get("threshold", 0.5))
    except Exception:
        threshold = 0.5

    churn_proba = float(pipeline.predict_proba(df)[0][1])
    prediction = int(churn_proba >= threshold)

    if churn_proba >= 0.70:
        risk = "High"
    elif churn_proba >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    result = {
        "prediction": prediction,
        "churn_probability": round(churn_proba, 4),
        "retention_risk": risk,
    }

    try:
        logger.info(
            "PREDICTION_GENERATED",
            extra={"input": json.dumps(customer), "output": json.dumps(result)}
        )
    except Exception:
        logger.info(f"PREDICTION_GENERATED: INPUT={customer} OUTPUT={result}")

    return result


def predict_batch(
    df: pd.DataFrame,
    model_path: pathlib.Path = _DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """
    Run batch predictions on a DataFrame.

    Returns the original DataFrame augmented with prediction columns.
    """
    artifact = _load_artifact(model_path)
    pipeline = artifact["pipeline"]

    probas = pipeline.predict_proba(df)[:, 1]
    preds = (probas >= 0.5).astype(int)

    result = df.copy()
    result["churn_probability"] = np.round(probas, 4)
    result["prediction"] = preds
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Customer Churn Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="JSON string of a single customer record")
    group.add_argument("--csv", type=str, help="Path to a CSV for batch prediction")
    parser.add_argument("--model", default=str(_DEFAULT_MODEL_PATH), help="Model path")
    args = parser.parse_args()

    mpath = pathlib.Path(args.model)

    if args.input:
        record = json.loads(args.input)
        result = predict_single(record, model_path=mpath)
        print(json.dumps(result, indent=2))
    else:
        df = pd.read_csv(args.csv)
        out = predict_batch(df, model_path=mpath)
        print(out[["churn_probability", "prediction"]].to_string())
