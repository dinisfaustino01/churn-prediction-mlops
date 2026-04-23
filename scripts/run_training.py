import logging
from pathlib import Path
import os

from churn_prediction.logging_setup import setup_logging
from churn_prediction.data_loader import load_raw_data
from churn_prediction.preprocessing import prepare_raw_xy, build_preprocessor
from churn_prediction.train import load_params, train_model
from churn_prediction.evaluate import evaluate_model
from churn_prediction.feature_engineering import build_features

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from dotenv import load_dotenv

import hashlib
import git


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
    model = train_model(X_train, X_test, y_train, y_test, params)

    evaluation = evaluate_model(model, X_test, y_test)
    accuracy = evaluation["accuracy"]
    precision = evaluation["precision"]
    recall = evaluation["recall"]
    f1_score = evaluation["f1_score"]
    roc_auc_score = evaluation["roc_auc_score"]
    brier_score = evaluation["brier_score"]
    true_negatives = evaluation["true_negatives"]
    false_positives = evaluation["false_positives"]
    false_negatives = evaluation["false_negatives"]
    true_positives = evaluation["true_positives"]
    classification_report = evaluation["classification_report"]

    logger.info("Accuracy: %s", accuracy)
    logger.info("Precision: %s", precision)
    logger.info("Recall: %s", recall)
    logger.info("F1 Score: %s", f1_score)
    logger.info("ROC AUC: %s", roc_auc_score)
    logger.info("Brier Score: %s", brier_score)
    logger.info("True Negatives: %s", true_negatives)
    logger.info("False Positives: %s", false_positives)
    logger.info("False Negatives: %s", false_negatives)
    logger.info("True Positives: %s", true_positives)
    
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

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_metric("roc_auc_score", roc_auc_score)
        mlflow.log_metric("brier_score", brier_score)
        mlflow.log_metric("true_negatives", true_negatives)
        mlflow.log_metric("false_positives", false_positives)
        mlflow.log_metric("false_negatives", false_negatives)
        mlflow.log_metric("true_positives", true_positives)

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