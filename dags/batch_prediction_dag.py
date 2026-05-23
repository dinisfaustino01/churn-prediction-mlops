"""Weekly batch churn-prediction DAG.

Task chain: validate → detect_drift → run_predictions →
check_data_quality → archive_processed_data.

Failure handling:
- All tasks retry once with a 30s delay.
- Hard data-quality failures stop the DAG before archive, so the
source file remains in incoming/ for re-processing after a fix.
- Soft data-quality warnings are logged but do not stop the pipeline.
- All persistence is idempotent on (batch_filename, model_version).
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from churn_prediction import PROJECT_ROOT
from churn_prediction.data.loader import load_raw_data
from churn_prediction.data.schema import get_column_lists
from churn_prediction.models.predict import make_predictions
from churn_prediction.monitoring.data_quality import aggregate_results, run_all_checks
from churn_prediction.monitoring.drift import check_drift
from churn_prediction.monitoring.metrics import push_dag_run_metrics, push_drift_metrics
from churn_prediction.registry.mlflow_client import (
    load_champion_model,
    load_champion_preprocessor,
)

logger = logging.getLogger(__name__)


FEATURE_SCHEMA_PATH = PROJECT_ROOT / "config" / "feature_schema.yaml"
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "training_snapshot.csv"
DRIFT_REPORTS_DIR = PROJECT_ROOT / "data" / "drift_reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _on_success(context):

    dag_run = context["dag_run"]
    duration = (datetime.now(timezone.utc) - dag_run.start_date).total_seconds()

    push_dag_run_metrics(
        dag_id=dag_run.dag_id,
        run_id=dag_run.run_id,
        status="success",
        duration_seconds=duration,
    )


def _on_failure(context):

    dag_run = context["dag_run"]
    duration = (datetime.now(timezone.utc) - dag_run.start_date).total_seconds()

    push_dag_run_metrics(
        dag_id=dag_run.dag_id,
        run_id=dag_run.run_id,
        status="failed",
        duration_seconds=duration,
    )


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
    on_success_callback=_on_success,
    on_failure_callback=_on_failure,
) as dag:

    @task
    def validate_incoming_data() -> str:
        """
        Validates the incoming batch file before processing.

        Checks that exactly one CSV exists in incoming/, and that it contains
        all expected feature columns defined in the feature schema.

        Returns the absolute path to the batch file as a string.

        Raises:
            FileNotFoundError: If no CSV files are found in incoming/.
            ValueError: If more than one file is found, or if expected columns are missing.
        """

        files = list(Path(INCOMING_DIR).glob("*.csv"))

        if not files:
            raise FileNotFoundError(f"No CSV files found in {INCOMING_DIR}")
        if len(files) > 1:
            raise ValueError(
                "Multiple files in incoming/. Process one batch at a time."
            )

        new_batch = str(files[0])
        logger.info("Selected batch: %s", Path(new_batch).name)

        load_raw_data(new_batch)

        return new_batch

    @task
    def detect_drift(new_batch: str, **context) -> str:
        """
        Runs Evidently drift detection against the training reference snapshot.

        Compares the incoming batch to the reference dataset across all numeric
        and categorical features. Saves an HTML report to disk and persists a
        summary row to the drift_reports table (idempotent on batch_filename).

        Args:
            new_batch: Absolute path to the incoming batch CSV.

        Returns:
            A dict with keys:
                - "batch_path": the input path (passed through to the next task)
                - "drift_result": the raw Evidently result dict

        Raises:
            FileNotFoundError: If the reference snapshot is missing.
        """

        _, _, cols_to_drop, _ = get_column_lists(FEATURE_SCHEMA_PATH)

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

        drift_check_results = check_drift(
            reference_df=reference,
            current_df=data,
            feature_schema_path=FEATURE_SCHEMA_PATH,
        )
        report = drift_check_results["report"]
        dataset_drift = drift_check_results["dataset_drift"]
        num_drifted_features = drift_check_results["num_drifted_features"]
        share_drifted_features = drift_check_results["share_drifted_features"]
        drift_table = drift_check_results["drift_table"]

        batch_stem = Path(new_batch).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(DRIFT_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        html_path = f"{DRIFT_REPORTS_DIR}/drift_report_{batch_stem}_{timestamp}.html"
        report.save_html(html_path)

        batch_filename = Path(new_batch).name
        drift_row = pd.DataFrame(
            [
                {
                    "batch_filename": batch_filename,
                    "dataset_drift_detected": dataset_drift,
                    "num_drifted_features": num_drifted_features,
                    "share_drifted_features": share_drifted_features,
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

        drift_per_feature = {
            feature: column_result["drift_score"]
            for feature, column_result in drift_table["drift_by_columns"].items()
        }

        dag_run = context["dag_run"]
        push_drift_metrics(
            dag_id=dag_run.dag_id,
            run_id=dag_run.run_id,
            batch_filename=batch_filename,
            drift_per_feature=drift_per_feature,
        )

        return new_batch

    @task
    def run_predictions(new_batch: str) -> dict:
        """
        Loads the champion model and generates churn predictions for the batch.

        Preprocesses the input data, runs inference with the current champion
        XGBoost model, and persists results to the predictions table
        (idempotent on batch_filename + model_version).

        Args:
            new_batch: Absolute path to the incoming batch CSV.

        Returns:
            Absolute path to the batch CSV, passed through to the next task.
        """

        data = pd.read_csv(new_batch)
        batch_filename = Path(new_batch).name

        model = load_champion_model()
        preprocessor = load_champion_preprocessor()

        client = mlflow.MlflowClient()
        model_version = client.get_model_version_by_alias(
            "churn-prediction-model", "champion"
        ).version

        results = make_predictions(
            df=data,
            model=model,
            preprocessor=preprocessor,
            feature_schema_path=FEATURE_SCHEMA_PATH,
            batch_filename=batch_filename,
            model_version=model_version,
        )

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

        return {"batch_path": new_batch, "model_version": str(model_version)}

    @task
    def check_data_quality(predictions_input: dict) -> str:
        """
        Runs data quality checks on the predictions written in the previous task.

        Fetches the predictions for this batch from the database, runs all checks
        via run_all_checks(), aggregates the results, and persists a report row
        to data_quality_reports (idempotent on batch_filename + model_version).

        Hard failures raise an exception, stopping the DAG and leaving the source
        file in incoming/ for re-processing. Soft warnings are logged but do not
        halt the pipeline.

        Args:
            predictions_input: Dict with keys "batch_path" and "model_version",
            as returned by run_predictions.

        Returns:
            The same path, passed through to archive_processed_data.

        Raises:
            ValueError: If any hard data quality checks fail.
        """

        new_batch = predictions_input["batch_path"]
        model_version = predictions_input["model_version"]

        data = pd.read_csv(new_batch)

        engine = get_db_engine()
        predictions = pd.read_sql(
            text(
                "SELECT * FROM predictions "
                "WHERE batch_filename = :b AND model_version = :v"
            ),
            con=engine,
            params={"b": Path(new_batch).name, "v": model_version},
        )

        check_results = run_all_checks(predictions, data)

        logger.info("Quality check results: %s", check_results)

        aggregated_results = aggregate_results(check_results)

        report_row = pd.DataFrame(
            [
                {
                    "batch_filename": Path(new_batch).name,
                    "model_version": str(model_version),
                    "check_timestamp": datetime.now(),
                    "all_passed": aggregated_results["all_passed"],
                    "total_checks": aggregated_results["total_checks"],
                    "passed_checks": aggregated_results["passed_checks"],
                    "failed_checks": aggregated_results["failed_checks"],
                    "warnings": aggregated_results["warnings"],
                    "check_details": json.dumps(check_results),
                }
            ]
        )

        with engine.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM data_quality_reports "
                    "WHERE batch_filename = :b AND model_version = :v"
                ),
                {"b": Path(new_batch).name, "v": str(model_version)},
            )
            report_row.to_sql(
                "data_quality_reports", con=conn, if_exists="append", index=False
            )

        if not aggregated_results["all_passed"]:
            logger.error(
                "Data quality hard failures: %s", aggregated_results["hard_failures"]
            )
            raise ValueError(
                f"Data quality checks failed for batch "
                f"{Path(new_batch).name}: {aggregated_results['hard_failures']}"
            )

        if aggregated_results["warnings"] > 0:
            logger.warning(
                "Data quality warnings present: %d", aggregated_results["warnings"]
            )

        return new_batch

    @task
    def archive_processed_data(new_batch: str) -> None:
        """
        Moves the batch file from incoming/ to processed/ after successful completion.

        This is the final step and only runs if all upstream tasks passed, including
        data quality checks. Leaving the file in incoming/ on failure
        is intentional, so the batch can be re-processed after a fix.

        Args:
            new_batch: Absolute path to the batch CSV to archive.
        """

        Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
        dest = f"{PROCESSED_DIR}/{Path(new_batch).name}"
        shutil.move(new_batch, dest)
        logger.info("Moved %s to %s", Path(new_batch).name, PROCESSED_DIR)

    new_batch = validate_incoming_data()
    checked_batch = detect_drift(new_batch)
    predicted_batch = run_predictions(checked_batch)
    quality_checked_data = check_data_quality(predicted_batch)
    archive_processed_data(quality_checked_data)
