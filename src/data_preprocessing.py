"""Load churn datasets and perform structural cleaning."""

import logging
import pathlib

import pandas as pd
import yaml
from datasets import load_dataset

logger = logging.getLogger(__name__)


ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_local_csv(config: dict) -> pd.DataFrame:
    raw_path = (ROOT / config["data"]["raw_path"]).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{raw_path}'. "
            "Ensure the CSV file exists at the configured path."
        )
    logger.info("LOAD_LOCAL_CSV: %s", raw_path)
    return pd.read_csv(raw_path)


def _load_hf_dataset(config: dict) -> pd.DataFrame:
    dataset_name = config["data"]["dataset"]
    logger.info("LOAD_HF_DATASET: %s (streaming=True)", dataset_name)
    ds = load_dataset(dataset_name, split="train", streaming=True)
    records = list(ds)
    return pd.DataFrame(records)


def load_and_clean(config: dict) -> pd.DataFrame:
    if config["data"].get("raw_path"):
        df = _load_local_csv(config)
    else:
        df = _load_hf_dataset(config)

    logger.info("RAW_SHAPE: %s", df.shape)

    drop_cols = [c for c in config["data"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info("DROPPED_COLUMNS: %s", drop_cols)

    pos = config["data"]["positive_class"]
    neg = config["data"]["negative_class"]
    target_col = config["data"]["target_column"]

    df = df[df[target_col].isin([pos, neg])].copy()
    if target_col != "Churn":
        df["Churn"] = df[target_col].apply(
            lambda v: 1 if v == pos else (0 if v == neg else None)
        ).astype(int)
        df = df.drop(columns=[target_col])
    else:
        df["Churn"] = df["Churn"].astype(int)
    logger.info("BINARY_TARGET_CREATED: POS_RATE=%.2f%%", df["Churn"].mean() * 100)

    logger.info("FINAL_SHAPE=%s", df.shape)
    return df
