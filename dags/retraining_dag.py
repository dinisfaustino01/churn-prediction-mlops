"""Automated weekly retraining pipeline for the churn prediction model.

Task chain: train_candidate_task → compare_and_register_task → notify_retraining_outcome_task

train_candidate_task trains a new candidate model and logs it to the
'churn-prediction-retraining' MLflow experiment. It returns only the
MLflow run ID, no large objects are passed between tasks.

compare_and_register_task loads the candidate from MLflow, re-derives
the held-out test set from the same deterministic split used during
training, and compares candidate vs. champion AUC. If the candidate
beats the champion by at least 1% AUC, it is registered under the
'challenger' alias for human review. The 'champion' alias is never
reassigned automatically, promotion to production is a manual decision
made in the MLflow UI.

notify_retraining_outcome_task sends a Slack message to #ml-retraining
summarising the outcome: new challenger registered, candidate rejected,
or first champion registered.

Failure handling:
- train_candidate_task has retries=0 — training failures are not retried
  to avoid logging duplicate models for the same DAG run.
- compare_and_register_task and notify_retraining_outcome_task inherit
  retries=1 from default_args.
- DAG success/failure pushes metrics to Pushgateway for Grafana and
  AlertManager monitoring.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import mlflow
from airflow import DAG
from airflow.decorators import task
from sklearn.model_selection import train_test_split

from churn_prediction import PROJECT_ROOT
from churn_prediction.data.loader import load_raw_data
from churn_prediction.features.engineering import build_features
from churn_prediction.features.preprocessor import prepare_raw_xy
from churn_prediction.models.comparison import compare_and_register
from churn_prediction.models.training_pipeline import train_candidate
from churn_prediction.monitoring.metrics import push_dag_run_metrics
from churn_prediction.monitoring.notifications import notify_retraining_outcome

logger = logging.getLogger(__name__)


DATA_PATH = PROJECT_ROOT / "data" / "training" / "train_data.csv"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "config" / "feature_schema.yaml"
MODEL_PARAMS_PATH = PROJECT_ROOT / "config" / "model_params.yaml"


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


with DAG(
    dag_id="retraining_pipeline",
    default_args=default_args,
    description="Weekly retraining of the model.",
    schedule_interval=None,
    start_date=datetime(2026, 4, 13),
    catchup=False,
    tags=["ml", "retraining"],
    on_success_callback=_on_success,
    on_failure_callback=_on_failure,
) as dag:

    @task(retries=0)
    def train_candidate_task() -> str:

        logger.info("Starting candidate training.")

        train_result = train_candidate(
            data_path=DATA_PATH,
            feature_schema_path=FEATURE_SCHEMA_PATH,
            model_params_path=MODEL_PARAMS_PATH,
            experiment_name="churn-prediction-retraining",
            run_name="train",
            extra_tags={"triggered_by": "retraining_dag"},
        )

        mlflow_run_id = train_result["run_id"]
        logger.info("Training complete. MLflow run_id: %s", mlflow_run_id)

        return mlflow_run_id

    @task
    def compare_and_register_task(mlflow_run_id: str) -> dict:

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        logger.info("Loading candidate model from MLflow run: %s", mlflow_run_id)

        try:
            candidate_model = mlflow.xgboost.load_model(f"runs:/{mlflow_run_id}/model")
            candidate_preprocessor = mlflow.sklearn.load_model(
                f"runs:/{mlflow_run_id}/preprocessor"
            )
        except Exception as e:
            logger.error(
                "Failed to load candidate artifacts from run %s: %s", mlflow_run_id, e
            )
            raise

        df = load_raw_data(DATA_PATH)
        fe_df = build_features(df)
        X_df, y = prepare_raw_xy(fe_df, FEATURE_SCHEMA_PATH)
        _, X_test_df, _, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("Test set derived: %d rows.", len(X_test_df))

        comparison_result = compare_and_register(
            candidate_run_id=mlflow_run_id,
            candidate_model=candidate_model,
            candidate_preprocessor=candidate_preprocessor,
            X_test_df=X_test_df,
            y_test=y_test,
        )

        logger.info("Comparison complete. Decision: %s", comparison_result["decision"])

        return comparison_result

    @task
    def notify_retraining_outcome_task(comparison_result: dict) -> None:

        logger.info(
            "Sending Slack notification for decision: %s", comparison_result["decision"]
        )
        notify_retraining_outcome(comparison_result)

        return

    mlflow_run_id = train_candidate_task()
    comparison_result = compare_and_register_task(mlflow_run_id)
    notify_retraining_outcome_task(comparison_result)
