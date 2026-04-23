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

import hashlib
import git


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


if __name__ == "__main__":
    main()