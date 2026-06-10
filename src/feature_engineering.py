"""Feature engineering transformations for churn prediction."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """Build feature matrix, target vector, and column type lists."""
    df = df.copy()

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

    _STREAMING_SERVICES = [
        col
        for col in df.columns
        if any(keyword in col for keyword in ["Security", "Backup", "Protection", "Tech Support", "Streaming"])
    ]
    if not _STREAMING_SERVICES:
        logger.warning("NO_SERVICE_COLUMNS_DETECTED")
    service_count = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    for svc in _STREAMING_SERVICES:
        if svc in df.columns:
            service_count += (df[svc] == "Yes").astype(int)
    df["Total_Services"] = service_count

    if "Age" in df.columns:
        df["Age_Group"] = (
            pd.cut(
                df["Age"],
                bins=[0, 30, 50, 70, 120],
                labels=["Young", "Adult", "Senior", "Elderly"],
            )
            .astype(str)
        )
        df = df.drop(columns=["Age"])

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    logger.info("FEATURE_ENGINEERING_COMPLETE: NUM=%d CAT=%d SAMPLES=%d", len(num_cols), len(cat_cols), len(X))
    return X, y, cat_cols, num_cols
