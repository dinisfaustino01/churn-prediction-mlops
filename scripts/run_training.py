import hashlib
import logging
import os
import time
from pathlib import Path

import git
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from churn_prediction.data_loader import load_raw_data
from churn_prediction.evaluate import evaluate_model
from churn_prediction.feature_engineering import build_features
from churn_prediction.logging_setup import setup_logging
from churn_prediction.preprocessing import build_preprocessor, prepare_raw_xy
from churn_prediction.train import load_params, train_model

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "training" / "train_data.csv"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "config" / "feature_schema.yaml"
MODEL_PARAMS_PATH = PROJECT_ROOT / "config" / "model_params.yaml"
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "reference" / "training_snapshot.csv"


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Return a 16-char stable fingerprint of the DataFrame contents."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()[:16]


def get_git_sha() -> str:
    """Return the short git SHA of the current HEAD."""
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha[:8]


def main() -> None:

    df = load_raw_data(DATA_PATH)
    dataset_hash = compute_dataset_hash(df)
    logger.info("Dataset hash: %s", dataset_hash)

    fe_df = build_features(df)

    X_df, y = prepare_raw_xy(fe_df, FEATURE_SCHEMA_PATH)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_train_df.to_csv(SNAPSHOT_PATH, index=False)
    logger.info("Training snapshot saved to %s", SNAPSHOT_PATH)

    preprocessor = build_preprocessor(FEATURE_SCHEMA_PATH)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    params = load_params(MODEL_PARAMS_PATH)

    t0 = time.perf_counter()
    model = train_model(X_train, X_test, y_train, y_test, params)
    duration = time.perf_counter() - t0
    logger.info("Training Duration: %s", duration)

    evaluation = evaluate_model(model, X_test, y_test)
    
    numeric_metrics = {k: v for k, v in evaluation.items() if isinstance(v, (int, float))}
    logger.info("Evaluation: %s", numeric_metrics) 
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in .env")
    
    logger.info("Connecting to MLflow tracking server: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn-prediction-training")

    git_sha = get_git_sha()
    logger.info("Git SHA: %s", git_sha)

    with mlflow.start_run(run_name=f"train-{git_sha}"):

        mlflow.set_tag("git_commit_sha", git_sha)
        mlflow.set_tag("dataset_hash", dataset_hash)

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
        registered_model = mlflow.register_model(model_uri, "churn-prediction-model")
        registered_preprocessor = mlflow.register_model(preprocessor_uri, "churn-prediction-preprocessor")

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(name="churn-prediction-model", alias="champion", version=registered_model.version)
        client.set_registered_model_alias(name="churn-prediction-preprocessor", alias="champion", version=registered_preprocessor.version)

        logger.info("Registered model v%s as champion", registered_model.version)
        logger.info("Registered preprocessor v%s as champion", registered_preprocessor.version)

if __name__ == "__main__":
    main()