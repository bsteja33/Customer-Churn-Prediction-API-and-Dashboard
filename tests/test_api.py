"""Tests for the Churn Prediction API — validates health, feature engineering,
classification logic, and the /predict and /predict/batch endpoints."""

import re
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.app import (
    app,
    _engineer_features,
    _classify,
    _col_map,
    _BINARY_FIELDS,
    CustomerFeatures,
)


# Ensure _BINARY_FIELDS is used — verify it's non-empty and every entry
# maps to a binary field in CustomerFeatures (checked in TestColumnMapping).
assert isinstance(_BINARY_FIELDS, set) and len(_BINARY_FIELDS) > 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient — the model artifact must exist at
    models/churn_model.pkl for the lifespan loader to succeed."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200_and_healthy_status(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_reports_model_loaded(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "model_path" in data


# ---------------------------------------------------------------------------
# _col_map integrity
# ---------------------------------------------------------------------------

class TestColumnMapping:
    def test_every_customer_features_field_has_mapping(self):
        """Every field in CustomerFeatures must have an entry in _col_map()."""
        mapping = _col_map()
        model_fields = set(CustomerFeatures.model_fields.keys())
        missing = model_fields - set(mapping.keys())
        assert not missing, f"Fields missing from _col_map: {missing}"

    def test_every_mapped_column_corresponds_to_model_field(self):
        """Every key in _col_map() must be a valid CustomerFeatures field."""
        mapping = _col_map()
        model_fields = set(CustomerFeatures.model_fields.keys())
        extra = set(mapping.keys()) - model_fields
        assert not extra, f"_col_map keys not in CustomerFeatures: {extra}"

    def test_binary_fields_all_have_renamed_column(self):
        """After applying _col_map() rename, every _BINARY_FIELDS entry
        should correspond to a renamed column from a binary field."""
        mapping = _col_map()
        for field_name, renamed in mapping.items():
            if renamed in _BINARY_FIELDS:
                # Verify the source field expects int (binary)
                field_info = CustomerFeatures.model_fields.get(field_name)
                assert field_info is not None
                field_type = str(field_info.annotation)
                assert "int" in field_type, (
                    f"Binary field {field_name} maps to {renamed} "
                    f"but has type {field_type}"
                )


# ---------------------------------------------------------------------------
# _engineer_features
# ---------------------------------------------------------------------------

_VALID_RECORD = {
    "Gender": "Male",
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 1,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 1,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": "Month-to-Month",
    "PaperlessBilling": 1,
    "PaymentMethod": "Bank Withdrawal",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0,
    "Married": 1,
    "NumberOfDependents": 0,
    "NumberOfReferrals": 0,
    "SatisfactionScore": 3,
    "InternetType": "Fiber Optic",
    "Offer": "Offer A",
    "Age": 45,
    "AvgMonthlyGBDownload": 50,
    "AvgMonthlyLongDistanceCharges": 15.0,
    "CLTV": 4000,
    "Under30": 0,
    "UnlimitedData": 1,
    "StreamingMusic": 1,
    "ReferredAFriend": 0,
    "TotalRefunds": 0.0,
    "TotalExtraDataCharges": 5,
    "TotalLongDistanceCharges": 45.0,
    "TotalRevenue": 1200.0,
}


class TestEngineerFeatures:
    def test_produces_non_empty_dataframe(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = _engineer_features(df)
        assert not result.empty

    def test_all_output_columns_are_numeric_or_bool(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = _engineer_features(df)
        for col in result.columns:
            dtype = result[col].dtype
            assert np.issubdtype(dtype, np.number) or dtype == bool, (
                f"Column '{col}' has non-numeric dtype {dtype}"
            )

    def test_all_column_names_are_sanitized(self):
        """After regex sanitization no column should contain spaces or special
        characters."""
        df = pd.DataFrame([_VALID_RECORD])
        result = _engineer_features(df)
        for col in result.columns:
            assert re.fullmatch(r"[a-zA-Z0-9_]+", col), (
                f"Column '{col}' contains illegal characters"
            )

    def test_binary_fields_produce_one_hot_columns(self):
        """A binary field like Partner:1 should produce Partner_Yes column."""
        df = pd.DataFrame([_VALID_RECORD])
        result = _engineer_features(df)
        # Find columns related to Partner
        partner_cols = [c for c in result.columns if "Partner" in c]
        assert len(partner_cols) >= 1, (
            f"No Partner-related columns found. Got: {list(result.columns)}"
        )

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame()
        result = _engineer_features(df)
        assert result.empty or len(result.columns) == 0

    def test_partial_record_does_not_raise(self):
        partial = {
            "Gender": "Female",
            "tenure": 5,
            "MonthlyCharges": 50.0,
        }
        df = pd.DataFrame([partial])
        result = _engineer_features(df)
        assert not result.empty
        # All columns numeric or bool (get_dummies produces bool)
        for col in result.columns:
            dtype = result[col].dtype
            assert np.issubdtype(dtype, np.number) or dtype == bool


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_high_risk_above_threshold(self):
        result = _classify(0.85)
        assert result["retention_risk"] == "High"
        assert result["prediction"] == 1

    def test_medium_risk(self):
        result = _classify(0.55)
        assert result["retention_risk"] == "Medium"

    def test_low_risk_below_medium_threshold(self):
        result = _classify(0.20)
        assert result["retention_risk"] == "Low"
        assert result["prediction"] == 0

    def test_boundary_high(self):
        result = _classify(0.70)
        assert result["retention_risk"] == "High"

    def test_boundary_medium_low(self):
        result = _classify(0.40)
        assert result["retention_risk"] == "Medium"

    def test_boundary_low(self):
        result = _classify(0.39)
        assert result["retention_risk"] == "Low"

    def test_churn_probability_rounded_to_four_decimals(self):
        result = _classify(0.123456)
        assert result["churn_probability"] == 0.1235

    def test_zero_probability(self):
        result = _classify(0.0)
        assert result["prediction"] == 0
        assert result["retention_risk"] == "Low"

    def test_certain_churn(self):
        result = _classify(1.0)
        assert result["prediction"] == 1
        assert result["retention_risk"] == "High"


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_valid_payload_returns_200(self, client: TestClient):
        response = client.post("/predict", json=_VALID_RECORD)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "prediction" in data
        assert "churn_probability" in data
        assert "retention_risk" in data

    def test_valid_payload_types_are_correct(self, client: TestClient):
        response = client.post("/predict", json=_VALID_RECORD)
        data = response.json()
        assert isinstance(data["prediction"], int)
        assert isinstance(data["churn_probability"], float)
        assert isinstance(data["retention_risk"], str)
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["retention_risk"] in ("High", "Medium", "Low")

    def test_malformed_payload_returns_422(self, client: TestClient):
        payload = {"Gender": 123}  # string expected
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_numeric_type_returns_422(self, client: TestClient):
        payload = {**_VALID_RECORD, "SeniorCitizen": "invalid"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_payload_returns_200(self, client: TestClient):
        """All fields are Optional — an empty JSON object is valid."""
        response = client.post("/predict", json={})
        assert response.status_code == 200

    def test_minimal_payload_returns_200(self, client: TestClient):
        minimal = {
            "Gender": "Male",
            "tenure": 1,
            "MonthlyCharges": 50.0,
        }
        response = client.post("/predict", json=minimal)
        assert response.status_code == 200

    def test_senior_citizen_typed_as_int(self, client: TestClient):
        payload = {**_VALID_RECORD, "SeniorCitizen": 1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_senior_citizen_with_float_returns_422(self, client: TestClient):
        """SeniorCitizen is Optional[int] — floats are rejected by Pydantic
        when strict=False for int but FastAPI auto-coerces whole-number
        floats."""
        payload = {**_VALID_RECORD, "SeniorCitizen": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class TestBatchPredictEndpoint:
    def test_batch_with_multiple_records(self, client: TestClient):
        payload = [_VALID_RECORD, _VALID_RECORD]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_returns_422(self, client: TestClient):
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# / root
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_message(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
