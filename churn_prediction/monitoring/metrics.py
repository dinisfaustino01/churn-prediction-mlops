import logging
import os

import numpy as np
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    pushadd_to_gateway,
)

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
JOB_NAME = "batch_prediction_dag"


logger = logging.getLogger(__name__)


def push_prediction_metrics(
    batch_filename: str,
    model_version: str,
    churn_probabilities: np.ndarray,
) -> None:
    """Push prediction-phase metrics to the Pushgateway.

    Emits:
        - predictions_total: rows scored in this batch.
        - predictions_churn_rate: share of predicted-positive labels.
        - prediction_confidence: distribution of churn probabilities.

    Push failures are logged as warnings rather than raised, so a Pushgateway
    outage does not fail the DAG.

    Args:
        batch_filename: Name of the source CSV (used as a label).
        model_version: MLflow model version that produced the predictions.
        churn_probabilities: 1-D array of churn probabilities for every row in the batch.
    """

    registry = CollectorRegistry()

    predictions_total = Counter(
        "predictions_total",
        "Total predictions in this batch.",
        ["model_version", "batch_filename"],
        registry=registry,
    )

    churn_rate = Gauge(
        "predictions_churn_rate",
        "Share of positive (churn) predictions in this batch.",
        ["model_version", "batch_filename"],
        registry=registry,
    )

    prediction_confidence_histogram = Histogram(
        "prediction_confidence",
        "Distribution of churn probabilities in this batch.",
        ["model_version", "batch_filename"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        registry=registry,
    )

    batch_size = len(churn_probabilities)
    share_predicted_churn = (churn_probabilities > 0.5).mean()

    predictions_total.labels(
        model_version=model_version, batch_filename=batch_filename
    ).inc(batch_size)
    churn_rate.labels(model_version=model_version, batch_filename=batch_filename).set(
        float(share_predicted_churn)
    )

    labeled_confidence = prediction_confidence_histogram.labels(
        model_version=model_version, batch_filename=batch_filename
    )
    for p in churn_probabilities:
        labeled_confidence.observe(float(p))

    try:
        pushadd_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
        logger.info(
            "Pushed prediction metrics: batch=%s model_version=%s n=%d",
            batch_filename,
            model_version,
            batch_size,
        )
    except Exception as e:
        logger.warning(
            "Failed to push prediction metrics for batch=%s: %s",
            batch_filename,
            e,
        )


def push_drift_metrics(
    dag_id: str,
    run_id: str,
    batch_filename: str,
    drift_per_feature: dict[str, float],
) -> None:
    """Push per-feature drift scores to the Pushgateway.

    Emits:
        - drift_score: Evidently drift score for each feature, labelled by
          feature name and batch filename.

    Push failures are logged as warnings rather than raised, so a Pushgateway
    outage does not fail the DAG.

    Args:
        batch_filename: Name of the source CSV (used as a label).
        drift_per_feature: Dict mapping feature name to its Evidently drift score.
    """

    registry = CollectorRegistry()

    drift_score = Gauge(
        "drift_score",
        "Per-feature drift score from Evidently.",
        ["feature_name", "batch_filename"],
        registry=registry,
    )

    for feature, score in drift_per_feature.items():
        drift_score.labels(feature_name=feature, batch_filename=batch_filename).set(
            float(score)
        )

    try:
        pushadd_to_gateway(
            PUSHGATEWAY_URL,
            job=JOB_NAME,
            grouping_key={"dag_id": dag_id, "run_id": run_id},
            registry=registry,
        )
        logger.info(
            "Pushed drift metrics: batch=%s n_features=%d",
            batch_filename,
            len(drift_per_feature),
        )
    except Exception as e:
        logger.warning(
            "Failed to push drift metrics for batch=%s: %s",
            batch_filename,
            e,
        )


def push_dag_run_metrics(
    dag_id: str,
    run_id: str,
    status: str,
    duration_seconds: float,
) -> None:
    """Push end-of-DAG metrics: overall status and total duration.

    Emits:
        - dag_run_status: 1 per run, labelled by status ("success" or "failed").
        - dag_run_duration_seconds: histogram of total DAG duration.

    Each run is pushed under a unique grouping_key (dag_id + run_id) so
    pushgateway retains every run as a distinct series, allowing
    count() queries over time in Grafana.
    """

    registry = CollectorRegistry()

    dag_run_status = Gauge(
        "dag_run_status",
        "End-of-DAG status.",
        ["dag_id", "status"],
        registry=registry,
    )

    dag_run_duration = Histogram(
        "dag_run_duration_seconds",
        "Total duration of a DAG run in seconds.",
        ["dag_id"],
        buckets=[10, 30, 60, 120, 300, 600, 1200],
        registry=registry,
    )

    dag_run_status.labels(dag_id=dag_id, status=status).set(1)
    dag_run_duration.labels(dag_id=dag_id).observe(duration_seconds)

    try:
        pushadd_to_gateway(
            PUSHGATEWAY_URL,
            job=JOB_NAME,
            grouping_key={"dag_id": dag_id, "run_id": run_id},
            registry=registry,
        )
        logger.info(
            "Pushed DAG run metrics: dag=%s run_id=%s status=%s duration=%.2fs",
            dag_id,
            run_id,
            status,
            duration_seconds,
        )
    except Exception as e:
        logger.warning(
            "Failed to push DAG run metrics for dag=%s run_id=%s: %s",
            dag_id,
            run_id,
            e,
        )
