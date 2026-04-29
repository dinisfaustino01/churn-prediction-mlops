import logging
import hashlib
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from airflow import DAG
from airflow.decorators import task
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from churn_prediction.preprocessing import get_column_lists, preprocess_inference_data


FEATURE_SCHEMA_PATH = "/opt/airflow/config/feature_schema.yaml"
INCOMING_DIR = "/opt/airflow/data/incoming"
REFERENCE_PATH = "/opt/airflow/data/reference/training_snapshot.csv"
DRIFT_REPORTS_DIR = "/opt/airflow/data/drift_reports"
PROCESSED_DIR = "/opt/airflow/data/processed"


logger = logging.getLogger(__name__)


# IMPLEMENT EXPONENTIAL BACKOFF LATER!
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
    # "retry_exponential_backoff": True,
    # "max_retry_delay": timedelta(minutes=10),
}


def get_db_engine():
    url = URL.create(
        "postgresql",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
    )
    return create_engine(url)


with DAG(
    dag_id="batch_prediction_pipeline",
    default_args=default_args,
    description="Weekly batch churn prediction with drift detection.",
    schedule_interval=None,
    start_date=datetime(2026, 4, 13),
    catchup=False,
    tags=["ml", "batch", "predictions"],
) as dag:

    @task
    def validate_incoming_data() -> str:
        files = list(Path(INCOMING_DIR).glob("*.csv"))

        if not files:
            raise FileNotFoundError(f"No CSV files found in {INCOMING_DIR}")
        if len(files) > 1:
            raise ValueError(
                "Multiple files in incoming/. Process one batch at a time."
            )

        new_batch = str(files[0])
        logger.info("Selected batch: %s", Path(new_batch).name)

        df = pd.read_csv(new_batch)
        logger.info("Row count: %d", len(df))

        numeric_cols, categorical_cols, _, _ = get_column_lists(FEATURE_SCHEMA_PATH)
        expected_columns = numeric_cols + categorical_cols

        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return new_batch

    @task
    def detect_drift(new_batch: str) -> dict:
        numeric_cols, categorical_cols, cols_to_drop, _ = get_column_lists(
            FEATURE_SCHEMA_PATH
        )

        data = pd.read_csv(new_batch)
        data = data.drop(columns=cols_to_drop, errors="ignore")
        data["SeniorCitizen"] = data["SeniorCitizen"].astype(str)

        try:
            reference = pd.read_csv(REFERENCE_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Reference snapshot not found at {REFERENCE_PATH}. "
                "Re-run scripts/run_training.py to generate it."
            ) from e

        reference["SeniorCitizen"] = reference["SeniorCitizen"].astype(str)

        column_mapping = ColumnMapping()
        column_mapping.numerical_features = numeric_cols
        column_mapping.categorical_features = categorical_cols
        column_mapping.target = None
        column_mapping.prediction = None

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference, current_data=data, column_mapping=column_mapping
        )

        batch_stem = Path(new_batch).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(DRIFT_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        html_path = f"{DRIFT_REPORTS_DIR}/drift_report_{batch_stem}_{timestamp}.html"
        report.save_html(html_path)

        result = report.as_dict()["metrics"][0]["result"]

        dataset_drift = result["dataset_drift"]
        num_drifted = result["number_of_drifted_columns"]
        share_drifted = result["share_of_drifted_columns"]

        if dataset_drift:
            logger.warning(
                "Drift detected: %d features drifted (share=%.3f)",
                num_drifted,
                share_drifted,
            )
        else:
            logger.info(
                "No dataset drift detected. Share drifted: %.3f", share_drifted
            )

        batch_filename = Path(new_batch).name
        drift_row = pd.DataFrame(
            [
                {
                    "batch_filename": batch_filename,
                    "dataset_drift_detected": dataset_drift,
                    "num_drifted_features": num_drifted,
                    "share_drifted_features": share_drifted,
                    "report_path": html_path,
                    "report_timestamp": datetime.now(),
                }
            ]
        )

        engine = get_db_engine()
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM drift_reports WHERE batch_filename = :b"),
                {"b": batch_filename},
            )
            drift_row.to_sql("drift_reports", con=conn, if_exists="append", index=False)

        return {"batch_path": new_batch, "drift_result": result}

    @task
    def run_predictions(batch_and_drift: dict) -> str:
        new_batch = batch_and_drift["batch_path"]

        data = pd.read_csv(new_batch)

        input_features_hash = hashlib.sha256(
            pd.util.hash_pandas_object(data, index=True).values
        ).hexdigest()[:16]

        model = mlflow.xgboost.load_model("models:/churn-prediction-model@champion")
        preprocessor = mlflow.sklearn.load_model(
            "models:/churn-prediction-preprocessor@champion"
        )

        data_transformed = preprocess_inference_data(
            data, FEATURE_SCHEMA_PATH, preprocessor
        )

        client = mlflow.MlflowClient()
        model_version = client.get_model_version_by_alias(
            "churn-prediction-model", "champion"
        ).version

        prediction_matrix = xgb.DMatrix(data_transformed)
        y_prob = model.predict(prediction_matrix)
        y_pred = (y_prob > 0.5).astype(int)

        logger.info("Predictions generated: %d", len(y_pred))
        logger.info("Share predicted to churn: %.3f", y_pred.mean())
        logger.info("Model version: %s", model_version)

        results = pd.DataFrame(
            {
                "customer_id": data["customerID"],
                "churn_probability": y_prob,
                "predicted_label": y_pred,
                "prediction_timestamp": datetime.now(),
                "model_version": model_version,
                "input_features_hash": input_features_hash,
                "batch_filename": Path(new_batch).name
            }
        )

        batch_filename = Path(new_batch).name

        engine = get_db_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM predictions "
                    "WHERE batch_filename = :b AND model_version = :v"
                ),
                {"b": batch_filename, "v": str(model_version)},
            )
            results.to_sql("predictions", con=conn, if_exists="append", index=False)

        return new_batch

    @task
    def archive_processed_data(new_batch: str) -> None:
        Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
        dest = f"{PROCESSED_DIR}/{Path(new_batch).name}"
        shutil.move(new_batch, dest)
        logger.info("Moved %s to %s", Path(new_batch).name, PROCESSED_DIR)

    new_batch = validate_incoming_data()
    checked_batch = detect_drift(new_batch)
    predicted_batch = run_predictions(checked_batch)
    archive_processed_data(predicted_batch)
