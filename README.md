# Customer Churn Prediction API and Dashboard

An end to end machine learning system for predicting customer churn in a telecommunications context. The pipeline trains a HistGradientBoostingClassifier (ROC-AUC ~0.92) via stratified cross-validation and RandomizedSearchCV, serves inference through a FastAPI backend with lifespan managed model preloading and renders results on a Streamlit dashboard.

---

## Architecture and Tech Stack

- **Language:** Python 3.10+
- **ML Framework:** Scikit-Learn (HistGradientBoostingClassifier, ColumnTransformer, SimpleImputer, RobustScaler, OneHotEncoder)
- **Hyperparameter Tuning:** RandomizedSearchCV with StratifiedKFold (5-fold, class_weight=balanced)
- **Model Serialization:** joblib
- **Backend:** FastAPI, Uvicorn, Pydantic v2
- **Frontend:** Streamlit, Plotly
- **Data Processing:** Pandas, NumPy
- **Containerization:** Docker, Docker Compose
- **CI:** GitHub Actions (lint, pytest, training sanity check)

The model artifact is loaded into `app.state` at server startup via FastAPI's `@asynccontextmanager` lifespan hook. Inference requests read directly from RAM with zero disk I/O per prediction.

---

## Repository Structure

```
Customer-Churn-Prediction/
├── api/
│   └── app.py                        FastAPI server; lifespan model preloading, /predict endpoint
├── data/
│   └── telecom_customer_churn.csv    Raw dataset (~7,000 records, 33 features)
├── models/
│   └── churn_model.pkl               Serialized sklearn pipeline (preprocessor + classifier)
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb   R&D documentation: EDA, ROC curves, SHAP analysis
├── src/
│   ├── data_preprocessing.py         Config-driven loading, target mapping, structural cleaning
│   ├── feature_engineering.py        Vectorized feature derivation (Revenue_per_Tenure, Age_Group)
│   ├── predict.py                    Inference module with CLI interface (single + batch)
│   └── train.py                      Training orchestrator: CV, tuning, evaluation, serialization
├── tests/
│   ├── test_api.py                   Endpoint validation (root, health, type rejection, inference)
│   ├── test_data_preprocessing.py    Data cleaning assertions
│   ├── test_feature_engineering.py   Feature derivation accuracy
│   ├── test_integration_train.py     End-to-end pipeline execution
│   └── test_predict.py              Inference output validation
├── ui/
│   └── app.py                        Streamlit dashboard with gauge visualization
├── .github/workflows/ci.yml         GitHub Actions CI pipeline
├── config.yaml                       Hyperparameters, data paths, API metadata
├── docker-compose.yml                Multi-service orchestration (API on 8000, UI on 8501)
├── Dockerfile.api                    API container
├── Dockerfile.ui                     Frontend container
├── Makefile                          CLI task automation
└── requirements.txt                  Python dependencies
```

---

## Execution Protocol

### 1. Environment Setup

```bash
git clone https://github.com/bsteja33/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

### 2. Model Training

Skip this step if `models/churn_model.pkl` already exists.

```bash
python src/train.py --config config.yaml
```

### 3. Backend Initialization

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

API documentation is served at `http://localhost:8000/docs`.

### 4. Frontend Initialization (separate terminal)

```bash
streamlit run ui/app.py
```

Dashboard is served at `http://localhost:8501`.

### 5. Docker Deployment (alternative)

```bash
docker-compose up --build -d
```

### 6. Test Suite

```bash
pytest -v
```

---

Maintained by [bsteja33](https://github.com/bsteja33)
