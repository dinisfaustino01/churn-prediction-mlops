"""Drift detection logic for the churn prediction pipeline.

Exposes check_drift(), which runs Evidently drift detection between a
reference and current dataset and returns a structured result dict.
"""

import logging
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from churn_prediction.data.schema import get_column_lists

logger = logging.getLogger(__name__)


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_schema_path: str | Path,
) -> dict:
    """Run Evidently drift detection between a reference and current dataset.

    Args:
        reference_df: The training snapshot used as the baseline distribution.
        current_df: The incoming batch to compare against the reference.
        feature_schema_path: Path to the YAML feature schema.

    Returns:
        Dict with keys: dataset_drift, num_drifted_features,
        share_drifted_features, drift_table, report.
    """

    numeric_cols, categorical_cols, _, _ = get_column_lists(feature_schema_path)

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numeric_cols
    column_mapping.categorical_features = categorical_cols
    column_mapping.target = None
    column_mapping.prediction = None

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    result = report.as_dict()["metrics"][0]["result"]
    drift_table = report.as_dict()["metrics"][1]["result"]

    dataset_drift = result["dataset_drift"]
    num_drifted_features = result["number_of_drifted_columns"]
    share_drifted_features = result["share_of_drifted_columns"]

    if dataset_drift:
        logger.warning(
            "Drift detected: %d features drifted (share=%.3f)",
            num_drifted_features,
            share_drifted_features,
        )
    else:
        logger.info(
            "No dataset drift detected. Share drifted: %.3f", share_drifted_features
        )

    return {
        "dataset_drift": dataset_drift,
        "num_drifted_features": num_drifted_features,
        "share_drifted_features": share_drifted_features,
        "drift_table": drift_table,
        "report": report,
    }
