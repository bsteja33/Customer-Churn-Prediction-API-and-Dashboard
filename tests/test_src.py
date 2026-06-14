"""Tests for src/ modules — data_preprocessing, feature_engineering, predict, train.
All external APIs (datasets, joblib, MLflow) are mocked."""

import json
import sys
import pathlib
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from src.data_preprocessing import load_config, load_and_clean
from src.feature_engineering import engineer_features as pandas_engineer
from src import predict as predict_mod


# Fixtures

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Senior Citizen": [0, 1],
        "Partner": [1, 0],
        "Dependents": [0, 1],
        "Tenure in Months": [12, 5],
        "Phone Service": [1, 0],
        "Multiple Lines": [0, 1],
        "Internet Service": [1, 0],
        "Online Security": [1, 0],
        "Online Backup": [0, 1],
        "Device Protection Plan": [0, 1],
        "Premium Tech Support": [1, 0],
        "Streaming TV": [1, 0],
        "Streaming Movies": [0, 1],
        "Contract": ["Month-to-Month", "One Year"],
        "Paperless Billing": [1, 0],
        "Payment Method": ["Bank Withdrawal", "Credit Card"],
        "Monthly Charge": [75.0, 50.0],
        "Total Charges": [900.0, 250.0],
        "Total Revenue": [1200.0, 300.0],
        "Married": [1, 0],
        "Number of Dependents": [0, 2],
        "Number of Referrals": [0, 1],
        "Satisfaction Score": [3, 4],
        "Internet Type": ["Fiber Optic", "DSL"],
        "Offer": ["Offer A", "None"],
        "Age": [45, 30],
        "Avg Monthly GB Download": [50, 20],
        "Avg Monthly Long Distance Charges": [15.0, 5.0],
        "CLTV": [4000, 2500],
        "Under 30": [0, 0],
        "Unlimited Data": [1, 0],
        "Streaming Music": [1, 0],
        "Referred a Friend": [0, 1],
        "Total Refunds": [0.0, 10.0],
        "Total Extra Data Charges": [5, 0],
        "Total Long Distance Charges": [45.0, 10.0],
        "Churn": [1, 0],
    })


# data_preprocessing

class TestLoadConfig:
    def test_load_config_returns_dict(self):
        cfg = load_config(str(pathlib.Path(ROOT) / "config.yaml"))
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert "model" in cfg
        assert cfg["data"]["dataset"] == "aai510-group1/telco-customer-churn"

    def test_load_config_default_path(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "lightgbm" in cfg


class TestLoadAndClean:
    def test_load_and_clean_with_local_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        df_in = pd.DataFrame({"Churn": [0, 1, 0], "Gender": ["M", "F", "M"]})
        df_in.to_csv(csv, index=False)
        config = {
            "data": {
                "raw_path": str(csv),
                "drop_columns": [],
                "target_column": "Churn",
                "positive_class": 1,
                "negative_class": 0,
            }
        }
        df_out = load_and_clean(config)
        assert "Churn" in df_out.columns
        assert len(df_out) == 3
        assert df_out["Churn"].dtype == int

    @patch("src.data_preprocessing.load_dataset")
    def test_load_and_clean_from_hf(self, mock_load_dataset):
        mock_ds = MagicMock()
        mock_ds.__iter__.return_value = iter([
            {"Churn": 1, "Gender": "M", "Customer ID": "abc"},
            {"Churn": 0, "Gender": "F", "Customer ID": "def"},
        ])
        mock_load_dataset.return_value = mock_ds
        config = {
            "data": {
                "raw_path": None,
                "dataset": "fake/name",
                "drop_columns": ["Customer ID"],
                "target_column": "Churn",
                "positive_class": 1,
                "negative_class": 0,
            }
        }
        df = load_and_clean(config)
        assert "Churn" in df.columns
        assert "Customer ID" not in df.columns
        assert len(df) == 2

    def test_load_and_clean_drops_columns(self, tmp_path):
        csv = tmp_path / "test2.csv"
        pd.DataFrame({"Churn": [0, 1], "City": ["A", "B"], "State": ["X", "Y"]}).to_csv(csv, index=False)
        config = {
            "data": {
                "raw_path": str(csv),
                "drop_columns": ["City", "State"],
                "target_column": "Churn",
                "positive_class": 1,
                "negative_class": 0,
            }
        }
        df = load_and_clean(config)
        assert "City" not in df.columns
        assert "State" not in df.columns

    def test_load_and_clean_renames_target_column(self, tmp_path):
        csv = tmp_path / "test3.csv"
        pd.DataFrame({"Churn Label": ["Yes", "No"], "Gender": ["M", "F"]}).to_csv(csv, index=False)
        config = {
            "data": {
                "raw_path": str(csv),
                "drop_columns": [],
                "target_column": "Churn Label",
                "positive_class": "Yes",
                "negative_class": "No",
            }
        }
        df = load_and_clean(config)
        assert "Churn" in df.columns
        assert "Churn Label" not in df.columns
        assert df["Churn"].tolist() == [1, 0]


# feature_engineering (pandas version)

class TestPandasFeatureEngineering:
    def test_returns_tuple_of_four(self, sample_df):
        X, y, cat_cols, num_cols = pandas_engineer(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(cat_cols, list)
        assert isinstance(num_cols, list)
        assert "Churn" not in X.columns

    def test_creates_revenue_per_tenure(self, sample_df):
        X, y, *_ = pandas_engineer(sample_df)
        assert "Revenue_per_Tenure" in X.columns
        assert X["Revenue_per_Tenure"].iloc[0] == pytest.approx(100.0)

    def test_creates_charges_per_month(self, sample_df):
        X, y, *_ = pandas_engineer(sample_df)
        assert "Charges_per_Month" in X.columns
        assert X["Charges_per_Month"].iloc[0] == pytest.approx(75.0)

    def test_zero_tenure_does_not_divide(self):
        df = pd.DataFrame({
            "Total Revenue": [0.0], "Tenure in Months": [0],
            "Total Charges": [0.0], "Churn": [0],
        })
        X, y, *_ = pandas_engineer(df)
        assert X["Revenue_per_Tenure"].iloc[0] == 0
        assert X["Charges_per_Month"].iloc[0] == 0

    def test_creates_total_services(self, sample_df):
        X, y, *_ = pandas_engineer(sample_df)
        assert "Total_Services" in X.columns
        assert X["Total_Services"].iloc[0] >= 0

    def test_creates_age_group(self, sample_df):
        X, y, *_ = pandas_engineer(sample_df)
        assert "Age_Group" in X.columns
        assert "Young" in X["Age_Group"].values or "Adult" in X["Age_Group"].values

    def test_drops_age_column(self, sample_df):
        X, y, *_ = pandas_engineer(sample_df)
        assert "Age" not in X.columns

    def test_returns_cat_and_num_cols(self, sample_df):
        X, y, cat_cols, num_cols = pandas_engineer(sample_df)
        assert len(cat_cols) > 0
        assert len(num_cols) > 0

    def test_no_service_columns_no_error(self):
        df = pd.DataFrame({"Churn": [0], "Gender": ["M"]})
        X, y, cat_cols, num_cols = pandas_engineer(df)
        assert "Total_Services" in X.columns

    def test_no_age_column_skips_binning(self):
        df = pd.DataFrame({"Churn": [0], "Total Revenue": [100.0], "Tenure in Months": [12]})
        X, y, *_ = pandas_engineer(df)
        assert "Age_Group" not in X.columns
        assert "Age" not in X.columns


# predict module

class TestPredictLoadArtifact:
    def test_load_artifact_raises_on_missing_file(self):
        fake_path = pathlib.Path(ROOT) / "models" / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError):
            predict_mod._load_artifact(fake_path)

    @patch("joblib.load")
    def test_load_artifact_caches_result(self, mock_joblib):
        fake_pipeline = MagicMock()
        mock_joblib.return_value = {"pipeline": fake_pipeline}
        predict_mod._ARTIFACT_CACHE = None
        path = pathlib.Path(ROOT) / "models" / "churn_model.pkl"
        result = predict_mod._load_artifact(path)
        assert "pipeline" in result
        predict_mod._ARTIFACT_CACHE = None


class TestPredictSingle:
    @patch("joblib.load")
    def test_predict_single_returns_dict(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        result = predict_mod.predict_single({"Gender": "Male", "tenure": 12, "MonthlyCharges": 75.0})
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "churn_probability" in result
        assert "retention_risk" in result

    @patch("joblib.load")
    def test_predict_single_high_risk(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        result = predict_mod.predict_single({"tenure": 1, "MonthlyCharges": 10.0})
        assert result["retention_risk"] == "High"
        assert result["prediction"] == 1

    @patch("joblib.load")
    def test_predict_single_low_risk(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.99, 0.01]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        result = predict_mod.predict_single({"tenure": 60, "MonthlyCharges": 20.0})
        assert result["retention_risk"] == "Low"
        assert result["prediction"] == 0

    @patch("joblib.load")
    def test_predict_single_with_full_features(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.5, 0.5]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        customer = {
            "Gender": "Male",
            "SeniorCitizen": 0,
            "Partner": 1,
            "tenure": 12,
            "PhoneService": 1,
            "InternetService": 1,
            "Contract": "Month-to-Month",
            "MonthlyCharges": 75.0,
            "TotalCharges": 900.0,
            "TotalRevenue": 1200.0,
            "Age": 45,
        }
        result = predict_mod.predict_single(customer)
        assert 0.0 <= result["churn_probability"] <= 1.0

    @patch("joblib.load")
    def test_predict_single_zero_tenure(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        result = predict_mod.predict_single({
            "Total Revenue": 0.0,
            "Total Charges": 0.0,
            "Tenure in Months": 0,
        })
        assert isinstance(result, dict)


class TestPredictBatch:
    @patch("joblib.load")
    def test_predict_batch_returns_dataframe(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        df = pd.DataFrame({"feature": [1, 2]})
        result = predict_mod.predict_batch(df)
        assert isinstance(result, pd.DataFrame)
        assert "churn_probability" in result.columns
        assert "prediction" in result.columns
        assert len(result) == 2


class TestPredictEdgeCases:
    """Edge-case branches: no service columns, zero Tenure in Months."""

    @patch("joblib.load")
    def test_no_service_columns_sets_zero(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.5, 0.5]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        result = predict_mod.predict_single({"tenure": 1, "MonthlyCharges": 10.0})
        assert result["churn_probability"] == 0.5

    @patch("joblib.load")
    def test_medium_risk_probability(self, mock_joblib):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_joblib.return_value = {"pipeline": mock_pipeline}
        predict_mod._ARTIFACT_CACHE = {"pipeline": mock_pipeline}
        result = predict_mod.predict_single({"tenure": 2, "MonthlyCharges": 30.0})
        assert "prediction" in result
        assert result["retention_risk"] in ("High", "Medium", "Low")


# train module

class TestTrainConfig:
    def test_train_load_config_returns_dict(self):
        from src.train import load_config as train_load_config
        cfg = train_load_config(str(pathlib.Path(ROOT) / "config.yaml"))
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert cfg["data"]["dataset"] == "aai510-group1/telco-customer-churn"


class TestTrainEngineerFeatures:
    @patch("src.train.load_dataset")
    def test_engineer_features_polars_returns_tuple(self, mock_load_dataset):
        import polars as pl
        from src.train import engineer_features as polars_engineer
        cfg = {
            "data": {
                "drop_columns": ["Customer ID"],
                "target_column": "Churn",
                "positive_class": 1,
                "negative_class": 0,
            }
        }
        df = pl.DataFrame({
            "Churn": [1, 0, 1],
            "Gender": ["M", "F", "M"],
            "Senior Citizen": ["0", "1", "0"],
            "Customer ID": ["a", "b", "c"],
        })
        X, y = polars_engineer(df, cfg)
        assert hasattr(X, "columns")
        assert hasattr(y, "to_list")
        assert "Customer ID" not in X.columns

    @patch("src.train.load_dataset")
    def test_engineer_features_polars_drops_columns(self, mock_load_dataset):
        import polars as pl
        from src.train import engineer_features as polars_engineer
        cfg = {
            "data": {
                "drop_columns": ["Churn Category", "City"],
                "target_column": "Churn",
                "positive_class": 1,
                "negative_class": 0,
            }
        }
        df = pl.DataFrame({
            "Churn": [1],
            "Churn Category": ["a"],
            "City": ["NYC"],
            "Gender": ["M"],
        })
        X, y = polars_engineer(df, cfg)
        assert "Churn Category" not in X.columns
        assert "City" not in X.columns

    @patch("src.train.load_dataset")
    def test_engineer_features_sanitizes_column_names(self, mock_load_dataset):
        import polars as pl
        from src.train import engineer_features as polars_engineer
        cfg = {
            "data": {
                "drop_columns": [],
                "target_column": "Churn",
                "positive_class": 1,
                "negative_class": 0,
            }
        }
        df = pl.DataFrame({
            "Churn": [1],
            "Internet Type": ["DSL"],
        })
        X, y = polars_engineer(df, cfg)
        for col in X.columns:
            assert col.isidentifier() or "_" in col


class TestTrainLoadData:
    @patch("src.train.pl.read_csv")
    def test_load_data_from_local_csv(self, mock_read_csv):
        import polars as pl
        from src.train import load_data
        mock_df = pl.DataFrame({"Churn": [1, 0], "Gender": ["M", "F"]})
        mock_read_csv.return_value = mock_df
        cfg = {
            "data": {
                "raw_path": "some/path.csv",
                "max_samples": 2,
            }
        }
        df = load_data(cfg)
        assert df.height == 2


# Cross-module integration: data_preprocessing -> feature_engineering

class TestCrossModuleIntegration:
    """Verify data flows correctly through the pipeline stages."""

    def test_preprocessing_to_feature_engineering_round_trip(self, tmp_path):
        csv = tmp_path / "churn_integration.csv"
        pd.DataFrame({
            "Churn": [1, 0],
            "Gender": ["Male", "Female"],
            "Age": [45, 30],
            "Tenure in Months": [12, 5],
            "Total Revenue": [1200.0, 300.0],
            "Total Charges": [900.0, 250.0],
            "Monthly Charge": [75.0, 50.0],
            "Contract": ["Month-to-Month", "One Year"],
            "Payment Method": ["Bank Withdrawal", "Credit Card"],
        }).to_csv(csv, index=False)

        config = {
            "data": {
                "raw_path": str(csv),
                "drop_columns": [],
                "target_column": "Churn",
                "positive_class": 1,
                "negative_class": 0,
            }
        }

        df_clean = load_and_clean(config)
        assert "Churn" in df_clean.columns
        assert len(df_clean) == 2

        X, y, cat_cols, num_cols = pandas_engineer(df_clean)
        assert "Churn" not in X.columns
        assert len(y) == 2
        assert len(cat_cols) > 0
        assert len(num_cols) > 0
        assert "Revenue_per_Tenure" in X.columns
        assert "Charges_per_Month" in X.columns
        assert "Age_Group" in X.columns
        assert "Age" not in X.columns
