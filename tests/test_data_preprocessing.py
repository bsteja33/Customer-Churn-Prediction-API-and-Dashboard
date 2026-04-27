"""
tests/test_data_preprocessing.py
"""
import pandas as pd
from src.data_preprocessing import load_config, load_and_clean


def test_load_and_clean():
    """Load the dataset and verify basic preprocessing steps."""
    cfg = load_config()
    df = load_and_clean(cfg)
    # Basic DataFrame sanity
    assert isinstance(df, pd.DataFrame)
    # Target column should be present and binary (0/1)
    assert "Churn" in df.columns
    assert set(df["Churn"].unique()) <= {0, 1}
    # Dropped columns from config should not be present
    for col in cfg["data"]["drop_columns"]:
        assert col not in df.columns
