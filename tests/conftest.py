import pandas as pd
import pytest

from churn_prediction import PROJECT_ROOT


@pytest.fixture
def valid_sample_path():
    """Load the small validated sample CSV path used across tests."""

    return PROJECT_ROOT / "tests" / "fixtures" / "valid_sample.csv"


@pytest.fixture
def valid_df():
    """Load the small validated sample CSV used across tests as a dataframe."""

    return pd.read_csv(PROJECT_ROOT / "tests" / "fixtures" / "valid_sample.csv")


@pytest.fixture
def feature_schema_path():
    """Path to the real feature_schema.yaml used by all preprocessors."""

    return PROJECT_ROOT / "config" / "feature_schema.yaml"


@pytest.fixture
def model_params_path():
    """Path to the real model_params.yaml used by all preprocessors."""

    return PROJECT_ROOT / "config" / "model_params.yaml"
