"""Reusable training pipeline for the churn prediction model.

Exposes train_candidate(), which handles the full training lifecycle:
data loading, feature engineering, preprocessing, model training,
evaluation, and MLflow logging. It deliberately does NOT register
the model under any alias, that decision belongs to the caller.

This separation allows the same pipeline to be invoked from:
  - scripts/run_training.py  (bootstrap / manual runs)
  - dags/retraining_dag.py   (automated scheduled retraining)

Both callers receive the same result dict and decide independently
whether and how to register the artifacts.
"""

from pathlib import Path
import logging
import hashlib
import time
import os

from churn_prediction.data.loader import load_raw_data
from churn_prediction.features.engineering import build_features
from churn_prediction.features.preprocessor import prepare_raw_xy, build_preprocessor
from churn_prediction.models.train import load_params, train_model
from churn_prediction.models.evaluate import evaluate_model

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

logger = logging.getLogger(__name__)


def _compute_dataset_hash(df: pd.DataFrame) -> str:
    """Return a 16-char stable fingerprint of the DataFrame contents."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()[:16]


def _get_git_sha() -> str:
    """Return the short git SHA of the current HEAD."""
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except Exception:
        return "unknown"


def train_candidate(
    data_path: Path,
    feature_schema_path: Path,
    model_params_path: Path,
    experiment_name: str,
    run_name: str,
    extra_tags: dict | None = None,
) -> dict:
    """Train and log a candidate model to MLflow. Does NOT register under any alias.

    Loads raw data, engineers features, splits train/test, fits the preprocessor,
    trains an XGBoost model, evaluates it, and logs everything (params, metrics,
    artifacts) to the specified MLflow experiment. 

    Args:
        data_path: Path to the raw CSV training data.
        feature_schema_path: Path to the YAML feature schema used by the
            preprocessor and feature preparation steps.
        model_params_path: Path to the YAML file containing XGBoost and
            training parameters.
        experiment_name: MLflow experiment to log the run under. Created
            automatically if it does not exist.
        run_name: Base name for the MLflow run. The current git SHA is
            appended automatically (e.g. "train-a1b2c3d").
        extra_tags: Optional dict of additional MLflow tags to set on the run
            (e.g. {"triggered_by": "retraining_dag"}).

    Returns:
        A dict containing:
            - run_id (str): MLflow run ID.
            - metrics (dict[str, float]): Numeric evaluation metrics logged to MLflow.
            - model_uri (str): MLflow artifact URI for the trained model.
            - preprocessor_uri (str): MLflow artifact URI for the fitted preprocessor.
            - X_train_df (pd.DataFrame): Raw (pre-preprocessor) training features.
            - X_test_df (pd.DataFrame): Raw (pre-preprocessor) test features.
            - y_test (pd.Series): Test labels.
            - candidate_model (xgb.Booster): The trained model object.
            - candidate_preprocessor (ColumnTransformer): The fitted preprocessor.

    Raises:
        RuntimeError: If MLFLOW_TRACKING_URI is not set.
    """

    df = load_raw_data(data_path)
    dataset_hash = _compute_dataset_hash(df)
    logger.info("Dataset hash: %s", dataset_hash)

    fe_df = build_features(df)

    X_df, y = prepare_raw_xy(fe_df, feature_schema_path)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(feature_schema_path)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    params = load_params(model_params_path)

    t0 = time.perf_counter()
    model = train_model(X_train, X_test, y_train, y_test, params)
    duration = time.perf_counter() - t0
    logger.info("Training Duration: %s", duration)

    evaluation = evaluate_model(model, X_test, y_test)

    numeric_metrics = {
        k: v for k, v in evaluation.items() if isinstance(v, (int, float))
    }
    logger.info("Evaluation: %s", numeric_metrics)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in .env")

    logger.info("Connecting to MLflow tracking server: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    git_sha = _get_git_sha()
    logger.info("Git SHA: %s", git_sha)

    with mlflow.start_run(run_name=f"{run_name}-{git_sha}"):
        mlflow.set_tag("git_commit_sha", git_sha)
        mlflow.set_tag("dataset_hash", dataset_hash)
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

        mlflow.log_params(params["xgb_params"])
        mlflow.log_params(params["training"])

        for name, value in numeric_metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_metric("training_duration_seconds", duration)

        mlflow.xgboost.log_model(model, artifact_path="model")
        mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessor")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        preprocessor_uri = f"runs:/{run_id}/preprocessor"

        result = {
            "run_id": run_id,
            "metrics": numeric_metrics,
            "model_uri": model_uri,
            "preprocessor_uri": preprocessor_uri,
            "X_train_df": X_train_df,
            "X_test_df": X_test_df,
            "y_test": y_test,
            "candidate_model": model,
            "candidate_preprocessor": preprocessor,
        }

        return result