"""Unit tests for Evidently drift detection.

Verifies that check_drift() correctly identifies no drift on identical
distributions and detects drift on shifted distributions.
"""

from churn_prediction.monitoring.drift import check_drift


def test_no_drift_on_identical_distributions(valid_df, feature_schema_path):

    result = check_drift(
        reference_df=valid_df,
        current_df=valid_df,
        feature_schema_path=feature_schema_path,
    )

    num_drifted_features = result["num_drifted_features"]

    assert num_drifted_features == 0


def test_drift_detected_on_shifted_distributions(
    valid_df, shifted_df, feature_schema_path
):

    result = check_drift(
        reference_df=valid_df,
        current_df=shifted_df,
        feature_schema_path=feature_schema_path,
    )

    num_drifted_features = result["num_drifted_features"]

    assert num_drifted_features > 0
