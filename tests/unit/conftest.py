import numpy as np
import pytest


@pytest.fixture
def stub_model():
    """Fake XGBoost model that returns all-ones probability array."""

    class StubModel:
        def predict(self, dmatrix):
            return np.ones(dmatrix.num_row(), dtype=np.float32)

    return StubModel()


@pytest.fixture
def stub_preprocessor():
    """Fake preprocessor that returns a numeric array of the correct shape."""

    class StubPreprocessor:
        def transform(self, X):
            return np.zeros((len(X), 10))

    return StubPreprocessor()


@pytest.fixture
def shifted_df(valid_df):
    """Copy of valid_df with one numeric column shifted to simulate drift."""

    shifted = valid_df.copy()
    shifted["tenure"] = shifted["tenure"] * 10

    return shifted
