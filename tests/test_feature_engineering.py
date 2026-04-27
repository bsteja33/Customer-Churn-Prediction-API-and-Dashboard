"""tests/test_feature_engineering.py"""
import pandas as pd
from src.data_preprocessing import load_config, load_and_clean
from src.feature_engineering import engineer_features


def test_engineered_features():
    """Run feature engineering and verify expected columns appear."""
    cfg = load_config()
    df = load_and_clean(cfg)
    X, y, cat_cols, num_cols = engineer_features(df)

    # Engineered columns should exist when source columns are present
    if "Total Revenue" in df.columns and "Tenure in Months" in df.columns:
        assert "Revenue_per_Tenure" in X.columns
    if "Total Charges" in df.columns and "Tenure in Months" in df.columns:
        assert "Charges_per_Month" in X.columns

    # Composite service count column should always be present
    assert "Total_Services" in X.columns

    # Age binning column appears if Age column existed in source
    if "Age" in df.columns:
        assert "Age_Group" in X.columns

    # Target alignment
    assert len(y) == len(X)

    # Verify inferred column type lists are accurate
    for col in cat_cols:
        assert X[col].dtype == object or pd.api.types.is_categorical_dtype(X[col])
    for col in num_cols:
        assert pd.api.types.is_numeric_dtype(X[col])
