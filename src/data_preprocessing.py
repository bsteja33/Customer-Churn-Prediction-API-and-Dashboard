"""Load churn datasets and perform structural cleaning."""

import logging
import pathlib
from typing import Dict, Any

import pandas as pd
from datasets import load_dataset

from src.config import DATA_CONFIG

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_local_csv(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load data from a local CSV file.

    :param config: The data configuration.
    :type config: dict
    :return: DataFrame containing the local dataset.
    :rtype: pd.DataFrame
    """
    raw_path = (ROOT / config["raw_path"]).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{raw_path}'. "
            "Ensure the CSV file exists at the configured path."
        )
    logger.info("LOAD_LOCAL_CSV: %s", raw_path)
    return pd.read_csv(raw_path)


def _load_hf_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Stream and load data from Hugging Face datasets.

    :param config: The data configuration.
    :type config: dict
    :return: DataFrame containing the remote dataset.
    :rtype: pd.DataFrame
    """
    dataset_name = config["dataset"]
    logger.info("LOAD_HF_DATASET: %s (streaming=True)", dataset_name)
    ds = load_dataset(dataset_name, split="train", streaming=True)
    records = list(ds)
    return pd.DataFrame(records)


def load_and_clean() -> pd.DataFrame:
    """
    Load data (local or remote) and apply structural cleaning.

    :return: Cleaned DataFrame ready for feature engineering.
    :rtype: pd.DataFrame
    """
    if DATA_CONFIG.get("raw_path"):
        dataframe = _load_local_csv(DATA_CONFIG)
    else:
        dataframe = _load_hf_dataset(DATA_CONFIG)

    logger.info("RAW_SHAPE: %s", dataframe.shape)

    drop_cols = [c for c in DATA_CONFIG["drop_columns"] if c in dataframe.columns]
    dataframe = dataframe.drop(columns=drop_cols, errors="ignore")
    logger.info("DROPPED_COLUMNS: %s", drop_cols)

    pos = DATA_CONFIG["positive_class"]
    neg = DATA_CONFIG["negative_class"]
    target_col = DATA_CONFIG["target_column"]

    dataframe = dataframe[dataframe[target_col].isin([pos, neg])].copy()
    if target_col != "Churn":
        dataframe["Churn"] = dataframe[target_col].apply(
            lambda v: 1 if v == pos else (0 if v == neg else None)
        ).astype(int)
        dataframe = dataframe.drop(columns=[target_col])
    else:
        dataframe["Churn"] = dataframe["Churn"].astype(int)
    logger.info("BINARY_TARGET_CREATED: POS_RATE=%.2f%%", dataframe["Churn"].mean() * 100)

    logger.info("FINAL_SHAPE=%s", dataframe.shape)
    return dataframe
