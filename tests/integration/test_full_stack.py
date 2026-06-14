"""Full-stack integration test: simulates the exact JSON payload dispatched by
the Next.js frontend through the FastAPI backend, covering prediction, retention
script generation (Groq fallback), and badge tag assertions."""

import sys
import pathlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)

from api.app import app

# Payload mirroring the exact structure sent by frontend/src/app/parameters/page.tsx
_PAYLOAD = {
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


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


class TestFullStackPrediction:
    """End-to-end: /predict with full payload mirroring the frontend form."""

    def test_health_check(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_predict_with_full_payload(self, client: TestClient):
        resp = client.post("/predict", json=_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "churn_probability" in data
        assert "retention_risk" in data
        assert isinstance(data["prediction"], int)
        assert isinstance(data["churn_probability"], float)
        assert data["retention_risk"] in ("High", "Medium", "Low")
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_risk_tier_linked_to_probability(self, client: TestClient):
        resp = client.post("/predict", json=_PAYLOAD)
        data = resp.json()
        prob = data["churn_probability"]
        risk = data["retention_risk"]
        if prob >= 0.70:
            assert risk == "High"
        elif prob >= 0.40:
            assert risk == "Medium"
        else:
            assert risk == "Low"

    def test_predict_minimal_payload(self, client: TestClient):
        minimal = {"Gender": "Female", "tenure": 5, "MonthlyCharges": 50.0}
        resp = client.post("/predict", json=minimal)
        assert resp.status_code == 200


class TestFullStackRetentionScript:
    """End-to-end: /generate_retention_script with fallback behavior."""

    @patch("api.app.groq_client.chat.completions.create")
    def test_retention_script_with_fallback(self, mock_groq, client: TestClient):
        """Groq unavailable — endpoint returns fallback script."""
        mock_groq.side_effect = Exception("API key not configured")
        resp = client.post(
            "/generate_retention_script",
            json={
                "risk_level": "High",
                "reasons": "Customer cited billing confusion.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "script" in data
        assert isinstance(data["script"], str)
        assert len(data["script"]) > 0
        assert data["script"].startswith("[Fallback Script]")

    def test_retention_script_invalid_reasons_type(self, client: TestClient):
        resp = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": 123},
        )
        assert resp.status_code == 422

    def test_retention_script_empty_reasons(self, client: TestClient):
        resp = client.post(
            "/generate_retention_script",
            json={"risk_level": "High", "reasons": ""},
        )
        assert resp.status_code in (200, 422)


class TestFullStackRoundTrip:
    """Full end-to-end: /predict → /generate_retention_script.

    Mirrors the exact frontend flow in frontend/src/app/parameters/page.tsx.
    """

    @patch("api.app.groq_client.chat.completions.create")
    def test_predict_then_script_round_trip(self, mock_groq, client: TestClient):
        mock_groq.side_effect = Exception("Groq API unavailable")

        # Step 1: Predict
        pred_resp = client.post("/predict", json=_PAYLOAD)
        assert pred_resp.status_code == 200
        pred = pred_resp.json()

        # Step 2: Generate script using prediction result
        reasons = (
            f"Churn probability {(pred['churn_probability'] * 100):.1f}%. "
            f"Contract: Month-to-Month. Tenure: 12 months."
        )
        script_resp = client.post(
            "/generate_retention_script",
            json={
                "risk_level": pred["retention_risk"],
                "reasons": reasons,
            },
        )
        assert script_resp.status_code == 200
        script_data = script_resp.json()

        # Step 3: Verify fallback tag prefix
        assert script_data["script"].startswith("[Fallback Script]")
        assert "[Generated by Llama-3]" not in script_data["script"]
        assert "We value you as a customer" in script_data["script"]
