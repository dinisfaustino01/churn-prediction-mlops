"""Unit tests for the preprocessor.

Verifies that the ColumnTransformer pipeline handles edge cases:
missing values, unseen categorical levels, and output determinism.
"""

import numpy as np

from churn_prediction.features.preprocessor import build_preprocessor, prepare_raw_xy


def test_handles_missing_values(valid_df, feature_schema_path):

    X_df, _ = prepare_raw_xy(valid_df, feature_schema_path)
    preprocessor = build_preprocessor(feature_schema_path)

    X_df.loc[0:5, "tenure"] = np.nan
    X_df.loc[10:15, "Contract"] = np.nan

    result = preprocessor.fit_transform(X_df)

    assert result.shape[0] == len(X_df)
    assert not np.isnan(result).any()


def test_handles_unseen_categoricals(valid_df, feature_schema_path):

    X_df, _ = prepare_raw_xy(valid_df, feature_schema_path)
    preprocessor = build_preprocessor(feature_schema_path)

    preprocessor.fit(X_df)

    X_copy = X_df.copy()
    X_copy.loc[0, "Contract"] = "Quarterly"

    result = preprocessor.transform(X_copy)

    assert result.shape[0] == len(X_df)


def test_output_shape_is_deterministic(valid_df, feature_schema_path):

    X_df, _ = prepare_raw_xy(valid_df, feature_schema_path)

    result_1 = build_preprocessor(feature_schema_path).fit_transform(X_df)
    result_2 = build_preprocessor(feature_schema_path).fit_transform(X_df)

    assert result_1.shape == result_2.shape
    assert np.allclose(result_1, result_2)
