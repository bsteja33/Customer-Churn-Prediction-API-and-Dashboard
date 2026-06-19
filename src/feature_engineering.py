"""Feature engineering transformations for churn prediction."""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def col_map() -> dict:
    """Map snake_case field names to actual dataset column names."""
    return {
        "Gender": "Gender",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "tenure": "Tenure in Months",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection Plan",
        "TechSupport": "Premium Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "MonthlyCharges": "Monthly Charge",
        "TotalCharges": "Total Charges",
        "Married": "Married",
        "NumberOfDependents": "Number of Dependents",
        "NumberOfReferrals": "Number of Referrals",
        "SatisfactionScore": "Satisfaction Score",
        "InternetType": "Internet Type",
        "Offer": "Offer",
        "Age": "Age",
        "AvgMonthlyGBDownload": "Avg Monthly GB Download",
        "AvgMonthlyLongDistanceCharges": "Avg Monthly Long Distance Charges",
        "CLTV": "CLTV",
        "Under30": "Under 30",
        "UnlimitedData": "Unlimited Data",
        "StreamingMusic": "Streaming Music",
        "ReferredAFriend": "Referred a Friend",
        "TotalRefunds": "Total Refunds",
        "TotalExtraDataCharges": "Total Extra Data Charges",
        "TotalLongDistanceCharges": "Total Long Distance Charges",
        "TotalRevenue": "Total Revenue",
    }


BINARY_FIELDS = {
    "Partner", "Dependents", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Paperless Billing",
    "Married", "Under 30", "Unlimited Data", "Streaming Music",
    "Referred a Friend",
}


def engineer_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw input DataFrame into model-ready feature matrix.

    Applies column renaming, binary-field coercion, one-hot encoding,
    and regex column sanitization. Designed for the inference path
    (API and CLI predict).
    """
    mapping = col_map()
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename)

    for col in BINARY_FIELDS:
        if col in df.columns:
            df[col] = df[col].map({1: "Yes", 0: "No"}).fillna("No")

    if "Total Charges" in df.columns:
        df["Total Charges"] = (
            df["Total Charges"]
            .replace(r"^\s*$", "0.0", regex=True)
            .astype(float)
        )

    if "Senior Citizen" in df.columns:
        df["Senior Citizen"] = df["Senior Citizen"].astype(str)

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(0)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in df.columns]

    return df


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

    service_cols = [
        col for col in df.columns
        if any(k in col for k in ["Security", "Backup", "Protection", "Tech Support", "Streaming"])
    ]
    if not service_cols:
        logger.warning("NO_SERVICE_COLUMNS_DETECTED")
    service_count = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    for svc in service_cols:
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

    del df

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    logger.info("FEATURE_ENGINEERING_COMPLETE: NUM=%d CAT=%d SAMPLES=%d", len(num_cols), len(cat_cols), len(X))
    return X, y, cat_cols, num_cols
