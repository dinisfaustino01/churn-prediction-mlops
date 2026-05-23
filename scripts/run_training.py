"""Bootstrap training script. Trains a candidate and immediately promotes it to champion.

Used for the initial model registration and for manual retraining outside of Airflow.
Once the automated retraining DAG is active, this script is superseded for scheduled
runs but remains useful for ad-hoc retraining and local development.

Unlike the retraining DAG, this script skips the champion-challenger comparison and
promotes the newly trained model directly. It also saves a training snapshot used
as the drift detection reference baseline.

Run via: `make train`
"""

import os

import mlflow
from dotenv import load_dotenv

from churn_prediction import PROJECT_ROOT
from churn_prediction.models.training_pipeline import train_candidate
from churn_prediction.utils.logging_setup import setup_logging

DATA_PATH = PROJECT_ROOT / "data" / "training" / "train_data.csv"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "config" / "feature_schema.yaml"
MODEL_PARAMS_PATH = PROJECT_ROOT / "config" / "model_params.yaml"
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "reference" / "training_snapshot.csv"


def main() -> None:

    load_dotenv()
    setup_logging()

    result = train_candidate(
        data_path=DATA_PATH,
        feature_schema_path=FEATURE_SCHEMA_PATH,
        model_params_path=MODEL_PARAMS_PATH,
        experiment_name="churn-prediction-training",
        run_name="train",
        extra_tags={"triggered_by": "manual"},
    )

    # Save training snapshot (only the bootstrap script does this)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result["X_train_df"].to_csv(SNAPSHOT_PATH, index=False)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    client = mlflow.MlflowClient()

    registered_model = mlflow.register_model(
        result["model_uri"], "churn-prediction-model"
    )
    registered_preprocessor = mlflow.register_model(
        result["preprocessor_uri"], "churn-prediction-preprocessor"
    )

    client.set_registered_model_alias(
        name="churn-prediction-model",
        alias="champion",
        version=registered_model.version,
    )
    client.set_registered_model_alias(
        name="churn-prediction-preprocessor",
        alias="champion",
        version=registered_preprocessor.version,
    )


if __name__ == "__main__":
    main()
