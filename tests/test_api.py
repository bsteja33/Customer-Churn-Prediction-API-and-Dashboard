"""Tests for the Churn Prediction API — validates health, feature engineering,
classification logic, and the /predict and /predict/batch endpoints.

All Groq API calls are mocked — no external credentials required."""

from src.feature_engineering import col_map, BINARY_FIELDS, engineer_features_inference
from api.app import (
    app,
    _classify,
    CustomerFeatures,
)
import re
import sys
import pathlib
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

assert isinstance(BINARY_FIELDS, set) and len(BINARY_FIELDS) > 0


# Mock helpers

def _mock_groq_response(text: str = "We value you as a customer."):
    """Build a MagicMock that mimics Groq's chat completion response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_groq_mock(return_value=None, side_effect=None):
    """Create a mock Groq instance that patches api.app.Groq."""
    instance = MagicMock()
    if side_effect:
        instance.chat.completions.create.side_effect = side_effect
    else:
        instance.chat.completions.create.return_value = return_value or _mock_groq_response()
    return instance


# Fixtures

@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient — the model artifact must exist at
    models/churn_model.pkl for the lifespan loader to succeed."""
    with TestClient(app) as c:
        yield c


# /health

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


# col_map integrity

class TestColumnMapping:
    def test_every_customer_features_field_has_mapping(self):
        mapping = col_map()
        model_fields = set(CustomerFeatures.model_fields.keys())
        missing = model_fields - set(mapping.keys())
        assert not missing, f"Fields missing from col_map: {missing}"

    def test_every_mapped_column_corresponds_to_model_field(self):
        mapping = col_map()
        model_fields = set(CustomerFeatures.model_fields.keys())
        extra = set(mapping.keys()) - model_fields
        assert not extra, f"col_map keys not in CustomerFeatures: {extra}"

    def test_binary_fields_all_have_renamed_column(self):
        mapping = col_map()
        for field_name, renamed in mapping.items():
            if renamed in BINARY_FIELDS:
                field_info = CustomerFeatures.model_fields.get(field_name)
                assert field_info is not None
                field_type = str(field_info.annotation)
                assert "int" in field_type, (
                    f"Binary field {field_name} maps to {renamed} "
                    f"but has type {field_type}"
                )


# engineer_features_inference

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
        result = engineer_features_inference(df)
        assert not result.empty

    def test_all_output_columns_are_numeric_or_bool(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = engineer_features_inference(df)
        for col in result.columns:
            dtype = result[col].dtype
            assert np.issubdtype(dtype, np.number) or dtype == bool, (
                f"Column '{col}' has non-numeric dtype {dtype}"
            )

    def test_all_column_names_are_sanitized(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = engineer_features_inference(df)
        for col in result.columns:
            assert re.fullmatch(r"[a-zA-Z0-9_]+", col), (
                f"Column '{col}' contains illegal characters"
            )

    def test_binary_fields_produce_one_hot_columns(self):
        df = pd.DataFrame([_VALID_RECORD])
        result = engineer_features_inference(df)
        partner_cols = [c for c in result.columns if "Partner" in c]
        assert len(partner_cols) >= 1, (
            f"No Partner-related columns found. Got: {list(result.columns)}"
        )

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame()
        result = engineer_features_inference(df)
        assert result.empty or len(result.columns) == 0

    def test_partial_record_does_not_raise(self):
        partial = {
            "Gender": "Female",
            "tenure": 5,
            "MonthlyCharges": 50.0,
        }
        df = pd.DataFrame([partial])
        result = engineer_features_inference(df)
        assert not result.empty
        for col in result.columns:
            dtype = result[col].dtype
            assert np.issubdtype(dtype, np.number) or dtype == bool


# _classify

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


# /predict

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
        payload = {**_VALID_RECORD, "SeniorCitizen": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# /predict/batch

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


# /generate_retention_script (mocked Groq)

class TestRetentionScriptEndpoint:
    """All tests mock api.app.Groq to avoid hitting the live Groq API."""

    MOCKED_SCRIPT = "Thank you for your loyalty. Let me apply a 10% discount to your next bill."

    @patch.dict("os.environ", {"GROQ_API_KEY": "dummy"})
    @patch("api.app.Groq")
    def test_valid_request_returns_script(self, mock_groq_cls, client: TestClient):
        mock_instance = _make_groq_mock(return_value=_mock_groq_response(self.MOCKED_SCRIPT))
        mock_groq_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={
                "risk_level": "High",
                "reasons": "Billing confusion, lack of usage.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        expected = "[Generated by Llama-3] " + self.MOCKED_SCRIPT
        assert data["script"] == expected
        assert "[Generated by Llama-3]" in data["script"]
        assert data["script"].startswith("[Generated by Llama-3]")
        assert "[Fallback Script]" not in data["script"]

    @patch.dict("os.environ", {"GROQ_API_KEY": "dummy"})
    @patch("api.app.Groq")
    def test_mock_was_called_with_correct_model(self, mock_groq_cls, client: TestClient):
        mock_instance = _make_groq_mock(return_value=_mock_groq_response(self.MOCKED_SCRIPT))
        mock_groq_cls.return_value = mock_instance
        client.post(
            "/generate_retention_script",
            json={"risk_level": "Low", "reasons": "No issues reported."},
        )
        mock_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama3-8b-8192"
        assert len(call_kwargs["messages"]) == 1
        assert "Low" in call_kwargs["messages"][0]["content"]

    @patch.dict("os.environ", {"GROQ_API_KEY": "dummy"})
    @patch("api.app.Groq")
    def test_groq_exception_falls_back(self, mock_groq_cls, client: TestClient):
        mock_instance = _make_groq_mock(side_effect=Exception("API timeout"))
        mock_groq_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": "Service outage."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["script"].startswith("[Fallback Script]")
        assert "We value you as a customer" in data["script"]
        assert "Let me review your account" in data["script"]
        assert "[Generated by Llama-3]" not in data["script"]

    @patch.dict("os.environ", {"GROQ_API_KEY": "dummy"})
    @patch("api.app.Groq")
    def test_groq_response_exactly_one_tag_prefix(self, mock_groq_cls, client: TestClient):
        mock_instance = _make_groq_mock(
            return_value=_mock_groq_response("Thank you for choosing us.")
        )
        mock_groq_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "Low", "reasons": "No issues."},
        )
        data = response.json()
        assert data["script"].count("[") == 1
        assert data["script"].count("]") == 1
        assert "[Generated by Llama-3]" in data["script"]
        assert data["script"].endswith("Thank you for choosing us.")

    @patch.dict("os.environ", {"GROQ_API_KEY": "dummy"})
    @patch("api.app.Groq")
    def test_fallback_script_has_no_llama_tag(self, mock_groq_cls, client: TestClient):
        mock_instance = _make_groq_mock(side_effect=RuntimeError("Connection refused"))
        mock_groq_cls.return_value = mock_instance
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": "Network issues."},
        )
        data = response.json()
        assert data["script"].startswith("[Fallback Script]")
        assert "[Generated by Llama-3]" not in data["script"]
        assert data["script"].count("[") == 1

    def test_missing_risk_level_returns_422(self, client: TestClient):
        response = client.post(
            "/generate_retention_script",
            json={"reasons": "Some reason."},
        )
        assert response.status_code == 422

    def test_missing_reasons_returns_422(self, client: TestClient):
        response = client.post(
            "/generate_retention_script",
            json={"risk_level": "High"},
        )
        assert response.status_code == 422


# Round-Trip: /predict -> /generate_retention_script

class TestRoundTrip:
    """Verify that a prediction result can be fed into the retention script
    endpoint in a single mocked flow."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "dummy"})
    @patch("api.app.Groq")
    def test_predict_then_generate_script(self, mock_groq_cls, client: TestClient):
        MOCK_SCRIPT = "Thank you for being a loyal customer."
        mock_instance = _make_groq_mock(return_value=_mock_groq_response(MOCK_SCRIPT))
        mock_groq_cls.return_value = mock_instance

        # Step 1: Get a prediction
        pred_resp = client.post("/predict", json=_VALID_RECORD)
        assert pred_resp.status_code == 200
        pred = pred_resp.json()
        assert "retention_risk" in pred
        assert "churn_probability" in pred

        # Step 2: Feed the risk level into the retention script endpoint
        script_resp = client.post(
            "/generate_retention_script",
            json={
                "risk_level": pred["retention_risk"],
                "reasons": (
                    f"Churn probability "
                    f"{(pred['churn_probability'] * 100):.1f}%. "
                    f"Contract: Month-to-Month. Tenure: 12 months."
                ),
            },
        )
        assert script_resp.status_code == 200
        script_data = script_resp.json()
        assert script_data["script"] == "[Generated by Llama-3] " + MOCK_SCRIPT

        # Step 3: Verify the mock was called with the correct risk level
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert pred["retention_risk"] in content


# / root

class TestRootEndpoint:
    def test_root_returns_message(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
