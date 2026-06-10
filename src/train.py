"""Streaming training pipeline — loads data, engineers features, trains LightGBM."""

import argparse
import logging
import os
import pathlib
import re
import sys

import joblib
import lightgbm as lgb
import mlflow
import polars as pl
import yaml
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pl.DataFrame:
    """Stream data from Hugging Face — never loads the full remote dataset.

    Takes up to ``max_samples`` rows from the stream and returns a Polars
    DataFrame. Supports HF_TOKEN for gated/private datasets.
    """
    max_samples = cfg.get("data", {}).get("max_samples", 50000)

    if cfg["data"].get("raw_path"):
        csv_path = ROOT / cfg["data"]["raw_path"]
        logger.info("LOAD_LOCAL_CSV: %s", csv_path)
        df = pl.read_csv(csv_path)
        if df.height > max_samples:
            df = df.head(max_samples)
            logger.info("TRUNCATED_TO_MAX_SAMPLES: %d", max_samples)
        logger.info("RAW_SHAPE: %d rows, %d cols", df.height, df.width)
        return df

    dataset_name = cfg["data"]["dataset"]
    hf_token = os.environ.get("HF_TOKEN", None)

    logger.info(
        "LOAD_HF_DATASET: %s (streaming=True, max_samples=%d)",
        dataset_name, max_samples,
    )

    try:
        dataset = load_dataset(
            dataset_name, split="train", streaming=True, token=hf_token,
        )
    except Exception as exc:
        logger.error("HF_DATASET_FAILED: %s \u2014 %s", dataset_name, exc)
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}'. "
            "If gated/private, set the HF_TOKEN environment variable. "
            f"Original error: {exc}"
        )

    records = []
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
        records.append(sample)
        log_interval = max(max_samples // 10, 1000)
        if (idx + 1) % log_interval == 0:
            logger.info("STREAM_PROGRESS: %d/%d rows collected", idx + 1, max_samples)

    df = pl.DataFrame(records)
    logger.info("STREAM_COMPLETE: %d rows, %d cols", df.height, df.width)
    return df


def engineer_features(df: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, pl.Series]:
    """Pure Polars feature engineering — no numpy / pandas in this step."""
    df = df.clone()

    drop_cols = [c for c in cfg["data"]["drop_columns"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
        logger.info("DROPPED_COLUMNS: %s", drop_cols)

    for tc_col in ["TotalCharges", "Total Charges"]:
        if tc_col in df.columns:
            if df[tc_col].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.when(pl.col(tc_col).str.strip_chars() == "")
                    .then(None)
                    .otherwise(pl.col(tc_col))
                    .alias(tc_col)
                )
                df = df.with_columns(pl.col(tc_col).cast(pl.Float64).fill_null(0.0))
                logger.info("%s_CLEANED: cast to Float64, nulls filled with 0.0", tc_col)
            break

    for sc_col in ["SeniorCitizen", "Senior Citizen"]:
        if sc_col in df.columns:
            df = df.with_columns(pl.col(sc_col).cast(pl.Utf8))
            logger.info("%s_CAST: cast to Utf8 for categorical encoding", sc_col)
            break

    pos = cfg["data"]["positive_class"]
    neg = cfg["data"]["negative_class"]
    target_col = cfg["data"]["target_column"]

    df = df.filter(pl.col(target_col).is_in([pos, neg]))

    if target_col != "Churn":
        df = df.with_columns(
            (pl.col(target_col) == pos).cast(pl.Int8).alias("Churn")
        )
        df = df.drop(target_col)
    else:
        df = df.with_columns(pl.col("Churn").cast(pl.Int8))

    churn_rate = df["Churn"].mean()
    logger.info("TARGET_CREATED: POS_RATE=%.2f%%", churn_rate * 100)

    cat_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    if cat_cols:
        df = df.to_dummies(columns=cat_cols)

    X = df.drop("Churn")
    y = df["Churn"]

    sanitized = {c: re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in X.columns}
    X = X.rename(sanitized)

    logger.info(
        "FEATURE_ENGINEERING_DONE: %d features, %d samples",
        X.width, X.height,
    )
    return X, y


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    seed = cfg["model"]["random_state"]

    df = load_data(cfg)
    X, y = engineer_features(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_pandas(), y.to_pandas(),
        test_size=cfg["model"]["test_size"],
        random_state=seed,
        stratify=y.to_pandas(),
    )
    logger.info("TRAIN_TEST_SPLIT: train=%d  test=%d", len(X_train), len(X_test))

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "churn_prediction"))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "random_state": seed,
        "verbosity": -1,
        **cfg.get("lightgbm", {}),
    }

    with mlflow.start_run() as run:
        mlflow.log_params(lgb_params)
        logger.info("MLFLOW_RUN_STARTED: run_id=%s", run.info.run_id)

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="auc",
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", roc_auc)
        logger.info("TEST_ROC_AUC: %.4f", roc_auc)

        model_path = ROOT / cfg["model"]["save_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {"pipeline": model}
        joblib.dump(artifact, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("MODEL_SAVED: %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Churn Prediction Model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
