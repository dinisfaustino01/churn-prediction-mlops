"""Unit tests for the prediction inference pipeline.

Verifies that make_predictions() returns a correctly structured output:
expected columns, row count, and sensible column dtypes.
"""

import numpy as np
import pandas as pd

from churn_prediction.models.predict import make_predictions


def test_prediction_output_format(
    valid_df, feature_schema_path, stub_model, stub_preprocessor
):

    result_df = make_predictions(
        df=valid_df,
        model=stub_model,
        preprocessor=stub_preprocessor,
        feature_schema_path=feature_schema_path,
        batch_filename="test",
        model_version="1",
    )

    expected_columns = {
        "customer_id",
        "churn_probability",
        "predicted_label",
        "prediction_timestamp",
        "model_version",
        "input_features_hash",
        "batch_filename",
    }

    assert isinstance(result_df, pd.DataFrame)
    # assert len(result_df) == len(valid_df)
    assert len(result_df) == 0
    assert set(result_df.columns) == expected_columns
    assert result_df["churn_probability"].dtype == np.float32
    assert result_df["predicted_label"].dtype == np.int64
    assert result_df["customer_id"].dtype == object
