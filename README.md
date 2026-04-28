# Customer Churn Prediction System

## Executive Summary
An enterprise-grade, end-to-end machine learning architecture designed for predicting telecommunications customer churn. The core engine trains a highly optimized `HistGradientBoostingClassifier` evaluated via stratified cross-validation. Inference is decoupled into a high-throughput FastAPI backend that employs strictly-typed Pydantic validation and memory-safe lifespan model preloading. Results are rendered via a decoupled Streamlit dashboard, engineered for resilience with built-in timeout and connection refusal handling. 

## Updated Project Architecture
```text
Customer-Churn-Prediction/
├── api/
│   └── app.py                        FastAPI inference server with strict Pydantic validation and lifespan caching.
├── data/
│   └── telecom_customer_churn.csv    Raw dataset containing the customer features.
├── models/                           [GITIGNORED] Local directory for serialized joblib model pipelines.
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb R&D artifacts, EDA, and SHAP interpretability analysis.
├── src/
│   ├── data_preprocessing.py         Structural data cleaning and pipeline serialization.
│   ├── feature_engineering.py        Vectorized feature derivation algorithms.
│   ├── predict.py                    CLI-driven inference module for batch processing.
│   └── train.py                      Training orchestrator governing cross-validation and hyperparameter tuning.
├── tests/
│   ├── conftest.py                   Pytest fixture orchestrator for generating dynamic test models.
│   ├── test_api.py                   Functional API integration testing.
│   ├── test_data_preprocessing.py    Preprocessing assertions.
│   ├── test_feature_engineering.py   Feature derivation accuracy checks.
│   ├── test_integration_train.py     End-to-end training pipeline execution testing.
│   └── test_predict.py               Inference threshold validation.
├── ui/
│   └── app.py                        Streamlit dashboard equipped with backend connection resilience.
├── .github/workflows/ci.yml          Strict continuous integration pipeline (flake8, pytest).
├── .flake8                           Centralized linting configuration enforcing PEP-8 compliance.
├── .gitignore                        Repository tracking exclusions.
├── config.yaml                       Centralized hyperparameters and structural configurations.
├── docker-compose.yml                Multi-container orchestration configurations.
├── Dockerfile.api                    API microservice build instructions.
├── Dockerfile.ui                     Frontend microservice build instructions.
├── LICENSE                           MIT License application.
├── Makefile                          Automated task execution protocols.
└── requirements.txt                  Strictly pinned Python dependencies.
```

## Installation & Execution

### 1. Environment Initialization
```bash
git clone https://github.com/bsteja33/Customer-Churn-Prediction-API-and-Dashboard.git
cd Customer-Churn-Prediction-API-and-Dashboard
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/MacOS:
# source venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Pipeline Generation
Because the pre-trained model is explicitly untracked to prevent cross-OS binary serialization mismatches, you must generate the model locally before spinning up the backend.
```bash
python src/train.py --config config.yaml
```

### 3. Service Execution
Execute the API backend and the Streamlit UI simultaneously in separate terminal sessions.

**Terminal 1 (Backend API):**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
*API documentation is automatically served at http://localhost:8000/docs.*

**Terminal 2 (Frontend Dashboard):**
```bash
streamlit run ui/app.py
```
*Dashboard is served at http://localhost:8501.*

## Testing Profiles
The `/predict` endpoint can be evaluated using standard curl or Postman payloads.

### High Risk Payload
```json
{
  "Contract": "Month-to-Month",
  "Tenure in Months": 2,
  "Monthly Charge": 95.0,
  "Total Charges": 190.0,
  "Internet Service": "Yes",
  "Internet Type": "Fiber Optic",
  "Payment Method": "Bank Withdrawal",
  "Paperless Billing": "Yes"
}
```

### Medium Risk Payload
```json
{
  "Contract": "One Year",
  "Tenure in Months": 24,
  "Monthly Charge": 65.0,
  "Total Charges": 1560.0,
  "Internet Service": "Yes",
  "Internet Type": "DSL",
  "Payment Method": "Credit Card",
  "Paperless Billing": "Yes"
}
```

### Low Risk Payload
```json
{
  "Contract": "Two Year",
  "Tenure in Months": 72,
  "Monthly Charge": 20.0,
  "Total Charges": 1440.0,
  "Internet Service": "No",
  "Payment Method": "Credit Card",
  "Paperless Billing": "No"
}
```

## License Indicator
This project is licensed under the MIT License. See the `LICENSE` file for full details.
