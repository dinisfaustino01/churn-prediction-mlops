import pytest
import pandas as pd

from churn_prediction import PROJECT_ROOT


@pytest.fixture
def valid_df():
    """Load the small validated sample CSV used across tests."""

    return pd.read_csv(PROJECT_ROOT / "tests" / "fixtures" / "valid_sample.csv")


@pytest.fixture
def feature_schema_path():
    """Path to the real feature_schema.yaml used by all preprocessors."""

    return PROJECT_ROOT / "config" / "feature_schema.yaml"