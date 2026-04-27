import pathlib
import pytest
import pandas as pd
from src.predict import predict_single, _DEFAULT_MODEL_PATH

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_sample_record():
    """Load a single row from the raw CSV, drop only the columns that
    the training pipeline removes. Keep NaN values as-is so the
    pipeline's SimpleImputer handles them identically to production."""
    csv_path = ROOT / "data" / "telecom_customer_churn.csv"
    df = pd.read_csv(csv_path, nrows=1)

    # Drop identifier, leakage, and target columns (mirrors data_preprocessing.py)
    drop_cols = [
        "Customer ID", "City", "Zip Code", "Latitude", "Longitude",
        "Churn Category", "Churn Reason", "Customer Status",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df.iloc[0].to_dict()


def test_predict_single_output_format():
    """Load the persisted model and verify the prediction dictionary shape."""
    model_path = pathlib.Path(_DEFAULT_MODEL_PATH)
    if not model_path.exists():
        pytest.skip("Model artifact not found - run the training script first.")

    record = _load_sample_record()
    result = predict_single(record, model_path=model_path)
    assert isinstance(result, dict)
    expected_keys = {"prediction", "churn_probability", "retention_risk"}
    assert set(result.keys()) == expected_keys
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["retention_risk"] in ("High", "Medium", "Low")
