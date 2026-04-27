"""
train.py
--------
End-to-end training script for the Customer Churn Prediction pipeline.

Execution flow:
  1. Load config
  2. Preprocess data
  3. Engineer features
  4. Train/compare 3 models with StratifiedKFold CV
  5. Tune best model (HistGradientBoostingClassifier) with RandomizedSearchCV
  6. Evaluate on held-out test set (ROC-AUC, Precision, Recall, F1)
  7. Save best pipeline to models/churn_model.pkl

Usage:
  python src/train.py
  python src/train.py --config config.yaml
"""

import argparse
import logging
import pathlib
import sys
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

# ── resolve project root regardless of where script is called from ───────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import load_and_clean, load_config  # noqa: E402
from src.feature_engineering import engineer_features            # noqa: E402


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Construct the ColumnTransformer: Imputer + RobustScaler + OneHotEncoder."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def evaluate_models(
    models: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    skf: StratifiedKFold,
) -> dict:
    """
    Cross-validate each model and print ROC-AUC means.

    Returns
    -------
    dict mapping model name → array of fold scores
    """
    results = {}
    logger.info("START_CV_COMPARISON")
    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        results[name] = scores
        logger.info("%-25s  ROC-AUC: %.4f ± %.4f", name, scores.mean(), scores.std())
    return results


def tune_best_model(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    skf: StratifiedKFold,
    cfg: dict,
) -> Pipeline:
    """Tune HistGradientBoostingClassifier with RandomizedSearchCV."""
    logger.info("START_HYPERPARAMETER_TUNING")

    raw_grid = cfg["tuning"]["param_grid"]
    # Convert null → None (YAML null becomes None, but list contains strings)
    param_grid = {}
    for k, v in raw_grid.items():
        param_grid[k] = [None if x is None else x for x in v]

    gb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=cfg["model"]["random_state"],
        )),
    ])

    rs = RandomizedSearchCV(
        gb_pipe,
        param_distributions=param_grid,
        n_iter=cfg["tuning"]["n_iter"],
        cv=skf,
        scoring=cfg["tuning"]["scoring"],
        n_jobs=-1,
        random_state=cfg["model"]["random_state"],
        verbose=1,
    )
    rs.fit(X_train, y_train)

    logger.info("Best params : %s", rs.best_params_)
    logger.info("Best CV AUC : %.4f", rs.best_score_)
    return rs.best_estimator_


def print_evaluation(best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the best model on the held-out test set.

    Prints classification report, plots Confusion Matrix, ROC curve, and
    Precision-Recall curve.

    Returns
    -------
    dict with test metrics.
    """
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    logger.info("TEST_SET_EVALUATION")
    logger.info("ROC-AUC: %.4f | PR-AUC: %.4f", roc_auc, pr_auc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ── ROC Curve ────────────────────────────────────────────────────────────
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "navy", lw=1, linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — Gradient Boosting")
    axes[1].legend()

    # ── Precision-Recall Curve ───────────────────────────────────────────────
    axes[2].plot(recall, precision, color="green", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve")
    axes[2].legend()

    plt.tight_layout()
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    logger.info("PLOTS_SAVED: %s", plot_path)
    plt.close()

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    df = load_and_clean(cfg)

    # ── Step 2: Feature Engineering ─────────────────────────────────────────
    X, y, cat_cols, num_cols = engineer_features(df)

    # ── Step 3: Train / Test Split ───────────────────────────────────────────
    seed = cfg["model"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], random_state=seed, stratify=y
    )
    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    preprocessor = build_preprocessor(num_cols, cat_cols)
    skf = RepeatedStratifiedKFold(n_splits=cfg["model"]["cv_folds"], n_repeats=3, random_state=seed)

    # ── Step 4: Cross-Validate 3 Models ─────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed
        ),
    }
    evaluate_models(models, preprocessor, X_train, y_train, skf)

    # ── Step 5: Tune Best Model ──────────────────────────────────────────────
    best_model = tune_best_model(preprocessor, X_train, y_train, skf, cfg)

    # ── Step 6: Evaluate on Test Set ─────────────────────────────────────────
    print_evaluation(best_model, X_test, y_test)

    # ── Step 7: Persist Model ─────────────────────────────────────────────────
    model_path = ROOT / cfg["model"]["save_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model alongside the column metadata needed by predict.py
    artifact = {
        "pipeline": best_model,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    joblib.dump(artifact, model_path)
    logger.info("MODEL_SAVED: %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Churn Prediction Model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
