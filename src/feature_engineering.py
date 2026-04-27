"""
feature_engineering.py
-----------------------
Responsible for constructing advanced derived features from the clean
DataFrame and returning the feature matrix (X), target vector (y),
and the column-type lists required by the scikit-learn ColumnTransformer.

Design decisions:
- Interaction ratios capture "velocity" better than raw absolute values.
- Age binning converts a continuous demographic into an ordinal group.
- Service count creates a single composite engagement signal.
- Column type lists are dynamically inferred — not hard-coded — so the
  pipeline remains robust if upstream columns are added or removed.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Apply all feature engineering transformations and return model-ready data.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame produced by data_preprocessing.load_and_clean().

    Returns
    -------
    X : pd.DataFrame         — feature matrix
    y : pd.Series            — binary target (Churn)
    cat_cols : list[str]     — categorical column names (for OHE)
    num_cols : list[str]     — numeric column names (for scaling)
    """
    df = df.copy()

    # 1. Revenue-per-Tenure
    # Higher ratio → customer pays more per month → potentially high-value risk
    if "Total Revenue" in df.columns and "Tenure in Months" in df.columns:
        df["Revenue_per_Tenure"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Revenue"] / df["Tenure in Months"]
        )

    # 2. Charges Velocity
    if "Total Charges" in df.columns and "Tenure in Months" in df.columns:
        df["Charges_per_Month"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Charges"] / df["Tenure in Months"]
        )

    # 3. Total Services Subscribed
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

    # 4. Demographic Age Binning
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

    # 5. Separate features and target
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # 6. Infer column types dynamically
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    logger.info("FEATURE_ENGINEERING_COMPLETE: NUM=%d CAT=%d SAMPLES=%d", len(num_cols), len(cat_cols), len(X))
    return X, y, cat_cols, num_cols
