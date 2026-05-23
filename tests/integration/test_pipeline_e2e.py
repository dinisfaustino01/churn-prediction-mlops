import math

import pytest
import numpy as np
import pandas as pd

from churn_prediction.models.training_pipeline import train_candidate
from churn_prediction.models.predict import make_predictions


@pytest.mark.integration
def test_train_and_prediction_e2e(tmp_path, 
        monkeypatch, 
        valid_df,
        valid_sample_path, 
        feature_schema_path, 
        model_params_path
):

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    
    result = train_candidate(
        data_path=valid_sample_path,
        feature_schema_path=feature_schema_path,
        model_params_path=model_params_path,
        experiment_name="integration-test",
        run_name="e2e"
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "run_id",
        "metrics",
        "model_uri",
        "preprocessor_uri",
        "X_train_df",
        "X_test_df",
        "y_test",
        "candidate_model",
        "candidate_preprocessor",
    }
    assert result["run_id"]
    assert result["model_uri"].startswith("runs:/")
    assert result["preprocessor_uri"].startswith("runs:/")
    assert result["metrics"]
    assert all(math.isfinite(v) for v in result["metrics"].values())

    test_df = result["X_test_df"]     
    model = result["candidate_model"]       
    preprocessor = result["candidate_preprocessor"]  

    result_df = make_predictions(
        df=valid_df,
        model=model,
        preprocessor=preprocessor,
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
    assert result_df["churn_probability"].between(0, 1).all()
    assert len(result_df) == len(valid_df)
    assert set(result_df.columns) == expected_columns
    assert result_df["churn_probability"].dtype == np.float32
    assert result_df["predicted_label"].dtype == np.int64
    assert result_df["customer_id"].dtype == object