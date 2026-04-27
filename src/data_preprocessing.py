"""
data_preprocessing.py
---------------------
Responsible for loading the raw telecom churn dataset, performing structural
cleaning, filtering the target variable, and handling missing values.

All decisions (columns to drop, class names) are driven by config.yaml so
that this module requires zero hard-coded changes for new datasets.
"""

import logging
import pathlib
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the central YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_clean(config: dict) -> pd.DataFrame:
    """
    Load the raw CSV, drop irrelevant identifier/leakage columns,
    filter to binary churn classes, create the binary target, and
    impute missing values.

    Parameters
    ----------
    config : dict
        Parsed config.yaml contents.

    Returns
    -------
    pd.DataFrame
        A clean DataFrame with a binary 'Churn' column (1 = Churned, 0 = Stayed).
    """
    # Resolve project root.
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    raw_path = (ROOT / config["data"]["raw_path"]).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{raw_path}'. "
            "Ensure 'data/telecom_customer_churn.csv' exists in the project root."
        )

    logger.info("LOAD_DATASET: %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("RAW_SHAPE: %s", df.shape)

    # ── 1. Drop non-predictive / post-hoc leakage columns ───────────────────
    drop_cols = [c for c in config["data"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info("DROPPED_COLUMNS: %s", drop_cols)

    # ── 2. Filter to binary classification target ────────────────────────────
    pos = config["data"]["positive_class"]   # "Churned"
    neg = config["data"]["negative_class"]   # "Stayed"
    target_col = config["data"]["target_column"]

    df = df[df[target_col].isin([pos, neg])].copy()
    df[target_col] = df[target_col].astype(str)
    df["Churn"] = df[target_col].map({pos: 1, neg: 0}).astype(int)
    df = df.drop(columns=[target_col])
    logger.info("BINARY_TARGET_CREATED: POS_RATE=%.2f%%", df["Churn"].mean() * 100)

    logger.info("PIPELINE_IMPUTATION_DELEGATED: FINAL_SHAPE=%s", df.shape)
    return df
