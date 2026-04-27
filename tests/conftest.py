"""
tests/conftest.py
-----------------
Session-scoped fixture that builds a minimal churn_model.pkl artifact
before the test suite runs. This ensures test_api.py and test_predict.py
pass in CI environments where no pre-trained model exists.

The micro-pipeline is trained on 50 synthetic rows and is structurally
identical to the production artifact so the ColumnTransformer schema matches.
"""

import pathlib
import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"

# Numeric and categorical columns must mirror the production pipeline exactly
_NUM_COLS = [
    "Number of Dependents", "Number of Referrals", "Tenure in Months",
    "Avg Monthly Long Distance Charges", "Avg Monthly GB Download",
    "Monthly Charge", "Total Charges", "Total Refunds",
    "Total Extra Data Charges", "Total Long Distance Charges",
    "Total Revenue", "Revenue_per_Tenure", "Charges_per_Month",
    "Total_Services",
]
_CAT_COLS = [
    "Gender", "Married", "Offer", "Phone Service", "Multiple Lines",
    "Internet Service", "Internet Type", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support", "Streaming TV",
    "Streaming Movies", "Streaming Music", "Unlimited Data", "Contract",
    "Paperless Billing", "Payment Method", "Age_Group",
]


def _make_synthetic_df(n: int = 60) -> pd.DataFrame:
    """Return a synthetic DataFrame with the same schema as the training data."""
    rng = np.random.default_rng(42)
    cat_maps = {
        "Gender": ["Male", "Female"],
        "Married": ["Yes", "No"],
        "Offer": ["None", "Offer A", "Offer B", "Unknown"],
        "Phone Service": ["Yes", "No"],
        "Multiple Lines": ["Yes", "No", "Unknown"],
        "Internet Service": ["Yes", "No"],
        "Internet Type": ["DSL", "Fiber Optic", "Cable", "Unknown"],
        "Online Security": ["Yes", "No", "Unknown"],
        "Online Backup": ["Yes", "No", "Unknown"],
        "Device Protection Plan": ["Yes", "No", "Unknown"],
        "Premium Tech Support": ["Yes", "No", "Unknown"],
        "Streaming TV": ["Yes", "No", "Unknown"],
        "Streaming Movies": ["Yes", "No", "Unknown"],
        "Streaming Music": ["Yes", "No", "Unknown"],
        "Unlimited Data": ["Yes", "No"],
        "Contract": ["Month-to-Month", "One Year", "Two Year"],
        "Paperless Billing": ["Yes", "No"],
        "Payment Method": ["Bank Withdrawal", "Credit Card", "Mailed Check"],
        "Age_Group": ["Young", "Adult", "Senior", "Elderly"],
    }
    data = {}
    for col in _NUM_COLS:
        data[col] = rng.uniform(0, 100, size=n)
    for col, choices in cat_maps.items():
        data[col] = rng.choice(choices, size=n)
    
    # Deterministic target: Month-to-Month contract = Churn
    df = pd.DataFrame(data)
    y = (df["Contract"] == "Month-to-Month").astype(int)
    return df, pd.Series(y, name="Churn")


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifact():
    """
    Build and persist a micro churn_model.pkl before the session starts.
    Tears down (deletes) the artifact only if it was created by this fixture
    (i.e., it did not already exist before the test run).
    """
    created_by_fixture = False

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        X, y = _make_synthetic_df(n=500)

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipe, _NUM_COLS),
            ("cat", cat_pipe, _CAT_COLS),
        ])
        clf = HistGradientBoostingClassifier(max_iter=20, random_state=42)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X, y)

        artifact = {
            "pipeline": pipeline,
            "num_cols": _NUM_COLS,
            "cat_cols": _CAT_COLS,
        }
        joblib.dump(artifact, MODEL_PATH)
        created_by_fixture = True

    yield

    if created_by_fixture and MODEL_PATH.exists():
        MODEL_PATH.unlink()
