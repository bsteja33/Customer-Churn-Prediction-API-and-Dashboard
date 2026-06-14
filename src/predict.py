"""Inference module — loads the persisted model and runs single/batch prediction."""

import argparse
import json
import logging
import pathlib

import joblib
import numpy as np
import pandas as pd

from typing import Any
from src.data_preprocessing import load_config
from src.feature_engineering import engineer_features_inference

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = ROOT / "models" / "churn_model.pkl"

if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

_ARTIFACT_CACHE: dict[str, Any] | None = None


def _load_artifact(model_path: pathlib.Path) -> dict:
    """Load and cache the model artifact."""
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
    """Predict churn probability for a single customer record."""
    artifact = _load_artifact(model_path)
    pipeline = artifact["pipeline"]

    df = pd.DataFrame([customer])
    df = engineer_features_inference(df)

    try:
        cfg = load_config(str(ROOT / "config.yaml"))
        _threshold = float(cfg["model"]["threshold"])
    except Exception:
        _threshold = 0.5

    churn_proba = float(pipeline.predict_proba(df)[0][1])
    prediction = int(churn_proba >= _threshold)

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

    logger.info("PREDICTION_GENERATED: INPUT=%s OUTPUT=%s", json.dumps(customer), json.dumps(result))

    return result


def predict_batch(
    df: pd.DataFrame,
    model_path: pathlib.Path = _DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Run batch predictions on a DataFrame."""
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
