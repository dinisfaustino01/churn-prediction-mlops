"""Inference pipeline for churn prediction.

Exposes make_predictions(), which handles the full prediction lifecycle:
feature preprocessing, XGBoost inference, metric pushing, and result assembly.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from churn_prediction.features.preprocessor import preprocess_inference_data
from churn_prediction.monitoring.metrics import push_prediction_metrics

logger = logging.getLogger(__name__)


def make_predictions(
    df: pd.DataFrame,
    model: xgb.Booster,
    preprocessor: ColumnTransformer,
    feature_schema_path: str | Path,
    batch_filename: str,
    model_version: str,
) -> pd.DataFrame:
    """Run the full inference pipeline on a raw customer DataFrame.

    Preprocesses the input data, runs XGBoost inference, and returns a
    DataFrame with customer IDs, predicted probabilities, binary labels,
    and traceability metadata.

    Args:
        df: Raw customer DataFrame as loaded from the source CSV.
        model: Trained XGBoost Booster object.
        preprocessor: Fitted sklearn ColumnTransformer.
        feature_schema_path: Path to the YAML feature schema.
        batch_filename: Name of the source batch file (for traceability).
        model_version: Champion model version number (for traceability).

    Returns:
        DataFrame with columns: customerID, churn_probability, churn_prediction,
        batch_filename, model_version, input_features_hash.
    """

    input_features_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()[:16]

    data_transformed = preprocess_inference_data(df, feature_schema_path, preprocessor)

    prediction_matrix = xgb.DMatrix(data_transformed)
    y_prob = model.predict(prediction_matrix)
    y_pred = (y_prob > 0.5).astype(int)

    logger.info("Predictions generated: %d", len(y_pred))
    logger.info("Share predicted to churn: %.3f", y_pred.mean())
    logger.info("Model version: %s", model_version)

    push_prediction_metrics(
        batch_filename=batch_filename,
        model_version=str(model_version),
        churn_probabilities=y_prob,
    )

    return pd.DataFrame(
        {
            "customer_id": df["customerID"],
            "churn_probability": y_prob,
            "predicted_label": y_pred,
            "prediction_timestamp": datetime.now(),
            "model_version": model_version,
            "input_features_hash": input_features_hash,
            "batch_filename": batch_filename,
        }
    )
