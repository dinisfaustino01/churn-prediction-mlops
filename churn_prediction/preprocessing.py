import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.feature_engineering import build_features

logger = logging.getLogger(__name__)


def get_column_lists(
    config_path: str | Path,
) -> tuple[list[str], list[str], list[str], str]:
    """Load feature schema from YAML and return the column groupings.

    Args:
        config_path: Path to the YAML file containing the feature schema.

    Returns:
        Tuple of (numeric_cols, categorical_cols, cols_to_drop, target_col).
    """

    logger.debug("Loading feature schema from %s", config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    numeric_cols = config["numeric"]
    categorical_cols = config["categorical"]
    cols_to_drop = config["drop"]
    target_col = config["target"]

    logger.debug(
        "Schema loaded: %d numeric, %d categorical, %d dropped, target=%s",
        len(numeric_cols),
        len(categorical_cols),
        len(cols_to_drop),
        target_col,
    )

    return numeric_cols, categorical_cols, cols_to_drop, target_col


def build_preprocessor(config_path: str | Path) -> ColumnTransformer:
    """Build the preprocessor used for transforming features.

    Args:
        config_path: Path to the YAML file containing the feature schema.

    Returns:
        An unfitted ColumnTransformer combining numeric and categorical pipelines.
    """

    numeric_cols, categorical_cols, _, _ = get_column_lists(config_path)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    logger.info(
        "Built preprocessor with %d numeric and %d categorical columns",
        len(numeric_cols),
        len(categorical_cols),
    )

    return preprocessor


def prepare_raw_xy(
    df: pd.DataFrame, config_path: str | Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw DataFrame into features (X) and binary target (y).

    Drops irrelevant columns, casts
    SeniorCitizen to string, and maps the target column to binary integers.
    Does not apply any fitted transformations, so it is safe to call on the
    full dataset before train/test split.

    Args:
        df: Raw DataFrame loaded directly from the source CSV.
        config_path: Path to the YAML feature schema file.

    Returns:
        Tuple of (X_df, y_series).
    """

    logger.info("Preparing raw X/y from DataFrame with shape %s", df.shape)

    X_df = df.copy()
    X_df = build_features(X_df)
    _, _, cols_to_drop, target_col = get_column_lists(config_path)

    X_df = X_df.drop(columns=cols_to_drop, errors="ignore")
    if target_col in X_df.columns:
        X_df = X_df.drop(columns=[target_col])
    X_df["SeniorCitizen"] = X_df["SeniorCitizen"].astype(str)

    y_series = df[target_col].map({"Yes": 1, "No": 0})

    logger.info(
        "Prepared X shape=%s, y positive rate=%.3f",
        X_df.shape,
        y_series.mean(),
    )

    return X_df, y_series


def preprocess_inference_data(
    data: pd.DataFrame,
    config_path: str | Path,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Transform new data for inference using a fitted preprocessor.

    Args:
        data: Raw DataFrame of new customer data (without target column).
        config_path: Path to the YAML feature schema file.
        preprocessor: Fitted ColumnTransformer from training.

    Returns:
        Transformed feature matrix ready for prediction.
    """

    logger.info("Preprocessing inference data with shape %s", data.shape)

    _, _, cols_to_drop, _ = get_column_lists(config_path)

    data = data.copy()
    data = build_features(data)
    data = data.drop(columns=cols_to_drop, errors="ignore")
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(str)

    transformed = preprocessor.transform(data)
    logger.info("Inference data transformed to shape %s", transformed.shape)

    return transformed
