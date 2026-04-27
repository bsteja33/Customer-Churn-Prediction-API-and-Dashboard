from fastapi.testclient import TestClient
import sys
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Churn Prediction API" in response.json()["message"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_validation_error(client):
    response = client.post("/predict", json={"Tenure in Months": "Not an Integer"})
    assert response.status_code == 422


def test_predict_high_risk(client):
    payload = {
        "Gender": "Male", "Married": "No", "Tenure in Months": 2,
        "Phone Service": "Yes", "Multiple Lines": "No",
        "Internet Service": "Yes", "Online Security": "No",
        "Online Backup": "No", "Device Protection Plan": "No",
        "Premium Tech Support": "No", "Streaming TV": "No",
        "Streaming Movies": "No", "Contract": "Month-to-Month",
        "Paperless Billing": "Yes", "Payment Method": "Bank Withdrawal",
        "Monthly Charge": 95.0, "Total Charges": 190.0,
        "Number of Dependents": 0, "Number of Referrals": 0,
        "Age": 45, "Offer": "Unknown",
        "Avg Monthly Long Distance Charges": 0.0,
        "Internet Type": "Fiber Optic", "Avg Monthly GB Download": 20.0,
        "Streaming Music": "No", "Unlimited Data": "Yes",
        "Total Refunds": 0.0, "Total Extra Data Charges": 0.0,
        "Total Long Distance Charges": 0.0, "Total Revenue": 190.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["retention_risk"] == "High"
    assert data["churn_probability"] > 0.5
