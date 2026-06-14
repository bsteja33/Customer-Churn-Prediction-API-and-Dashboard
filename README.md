# Enterprise Churn Engine

Predictive retention system for telecoms. Uses LightGBM for churn risk scoring, FastAPI as the inference gateway, and a Next.js dashboard for interaction.

## Tech Stack

* **Frontend:** Next.js 16, TypeScript, Zustand, Zod, Recharts, Tailwind v4, Vitest
* **Backend:** FastAPI, Pydantic v2, Pandas, LightGBM
* **AI:** Groq LLM (llama3-8b-8192)
* **Data:** Hugging Face datasets, Polars streaming
* **Infra:** Docker Compose, Python 3.13

## System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#2563eb',
  'primaryTextColor': '#fff',
  'primaryBorderColor': '#3b82f6',
  'lineColor': '#6b7280',
  'secondaryColor': '#1e293b',
  'tertiaryColor': '#0f172a'
}}}%%
graph TD
    A["User Inputs"]:::frontend -->|"Next.js Form + Zod Validation"| B["JSON Payload POST"]:::frontend
    B -->|"localhost:8000/predict"| C["FastAPI Gateway"]:::backend
    C -->|"CustomerFeatures Pydantic Validation"| D["Feature Matrix Engineering"]:::backend
    D -->|"Column Matching + Regex Sanitization"| E["LightGBM Pipeline<br/>models/churn_model.pkl"]:::ml
    E -->|"predict_proba → Risk Tier"| F["Groq LLM<br/>llama3-8b-8192"]:::ai
    F -->|"Retention Script"| G["Recharts Terminal<br/>Monochrome Bar Chart"]:::frontend

    classDef frontend fill:#1e3a5f,stroke:#3b82f6,stroke-width:2px,color:#e0f2fe
    classDef backend fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#dcfce7
    classDef ml fill:#3b0764,stroke:#a855f7,stroke-width:2px,color:#f3e8ff
    classDef ai fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#ffedd5
```

## Dataset

Training data comes from [aai510-group1/telco-customer-churn](https://huggingface.co/datasets/aai510-group1/telco-customer-churn). It has 52 columns with demographics, services, account details, and churn labels. The pipeline removes 14 columns (IDs, location data, churn metadata) and builds features using one-hot encoding and column name sanitization.

* ROC AUC: **0.9939**
* Model: LightGBM (gradient boosted trees)
* Data Pipeline: Polars streaming
* Training Volume: 50,000 customer records
* Validation: Stratified 80/20 holdout

## How to Run

### Native

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python src/train.py --config config.yaml

# Terminal 1 - Backend
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend && npm install && npm run dev
```

### Docker Compose

```bash
docker compose up --build
```

### API Test

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "SeniorCitizen": 0,
    "Partner": 0,
    "tenure": 2,
    "PhoneService": 1,
    "InternetService": 1,
    "Contract": "Month-to-Month",
    "PaperlessBilling": 1,
    "PaymentMethod": "Bank Withdrawal",
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0
  }'
```

## Project Structure

```text
.dockerignore
.flake8
.gitignore
.github/workflows/ci.yml
LICENSE
Makefile
README.md
api/
  app.py
  __init__.py
config.yaml
docker-compose.yml
Dockerfile
frontend/
  package.json
  next.config.ts
  vitest.config.ts
  src/
    app/
      layout.tsx
      page.tsx
      globals.css
      analysis/page.tsx
      parameters/page.tsx
      status/page.tsx
    components/ui/FormField.tsx
    lib/schema.ts
    store/useChurnStore.ts
    tests/
      store.test.ts
      components.test.tsx
    types/api.ts
load_tests/locustfile.py
models/churn_model.pkl
requirements-dev.txt
requirements.txt
scripts/monitor_health.py
src/
  data_preprocessing.py
  feature_engineering.py
  predict.py
  train.py
  __init__.py
tests/
  __init__.py
  test_api.py
  test_src.py
  integration/
    test_full_stack.py
```

## Load Test Results

Benchmark: Locust with 10 concurrent users, spawn rate 2/s, 30s duration.

* Requests Per Second: 6.72
* Failure Rate: 0.00%
* Average Latency: 187 ms
* p50 (Median): 53 ms
* p95: 2000 ms
* p99: 3700 ms
* Total Requests: 196
* Total Failures: 0

## Testing

```bash
# All backend tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=api --cov=src --cov-report=term-missing

# Frontend tests
cd frontend && npm test

# Load testing
locust -f load_tests/locustfile.py --headless -u 10 -r 2
```

The test suite mocks the Groq API, so no API key is needed for local runs. The CI pipeline enforces a minimum 80% code coverage threshold.

## License

MIT. See [LICENSE](./LICENSE).
