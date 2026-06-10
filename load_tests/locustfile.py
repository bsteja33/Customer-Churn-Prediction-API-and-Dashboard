import random
from locust import HttpUser, task, between


def _random_payload():
    genders = ["Male", "Female"]
    contracts = ["Month-to-Month", "One Year", "Two Year"]
    internet_types = ["DSL", "Fiber Optic", "Cable", "None"]
    payments = ["Bank Withdrawal", "Credit Card", "Mailed Check"]
    offers = ["None", "Offer A", "Offer B", "Offer C"]

    return {
        "Gender": random.choice(genders),
        "Married": random.choice(["Yes", "No"]),
        "Number of Dependents": random.randint(0, 5),
        "Number of Referrals": random.randint(0, 10),
        "Tenure in Months": random.randint(1, 72),
        "Offer": random.choice(offers),
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": round(random.uniform(0, 50), 2),
        "Multiple Lines": random.choice(["Yes", "No"]),
        "Internet Service": "Yes",
        "Internet Type": random.choice(internet_types),
        "Avg Monthly GB Download": round(random.uniform(0, 500), 1),
        "Online Security": random.choice(["Yes", "No"]),
        "Online Backup": random.choice(["Yes", "No"]),
        "Device Protection Plan": random.choice(["Yes", "No"]),
        "Premium Tech Support": random.choice(["Yes", "No"]),
        "Streaming TV": random.choice(["Yes", "No"]),
        "Streaming Movies": random.choice(["Yes", "No"]),
        "Streaming Music": random.choice(["Yes", "No"]),
        "Unlimited Data": random.choice(["Yes", "No"]),
        "Contract": random.choice(contracts),
        "Paperless Billing": random.choice(["Yes", "No"]),
        "Payment Method": random.choice(payments),
        "Monthly Charge": round(random.uniform(20, 120), 2),
        "Total Charges": round(random.uniform(100, 6000), 2),
        "Total Refunds": round(random.uniform(0, 50), 2),
        "Total Extra Data Charges": round(random.uniform(0, 20), 2),
        "Total Long Distance Charges": round(random.uniform(0, 300), 2),
        "Total Revenue": round(random.uniform(100, 6000), 2),
        "Age": random.randint(18, 80),
    }


class CustomerServiceAgent(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(1)
    def predict_churn(self):
        payload = _random_payload()
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 422:
                resp.failure(f"Validation error: {resp.text}")
            elif resp.status_code == 500:
                resp.failure(f"Server error: {resp.text}")
            elif resp.status_code != 200:
                resp.failure(f"Unexpected status: {resp.status_code}")
