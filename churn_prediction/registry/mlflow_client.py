import logging
import os

import mlflow
import xgboost as xgb
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def _load_champion(name: str, loader):
    """Load the champion version of a registered MLflow model.

    Args:
        name: Registered model name in MLflow.
        loader: MLflow loader function (e.g. mlflow.xgboost.load_model).

    Returns:
        The loaded model object.

    Raises:
        RuntimeError: If MLFLOW_TRACKING_URI is not set.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set")
    
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name, "champion").version
    obj = loader(f"models:/{name}@champion")

    logger.info("Loaded %s champion v%s", name, version)

    return obj


def load_champion_model() -> xgb.Booster:
    """Load the champion XGBoost model from the MLflow model registry.

    Returns:
        Trained XGBoost Booster object.
    """

    return _load_champion("churn-prediction-model", mlflow.xgboost.load_model)


def load_champion_preprocessor() -> ColumnTransformer:
    """Load the champion preprocessing pipeline from the MLflow model registry.

    Returns:
        Fitted sklearn ColumnTransformer.
    """

    return _load_champion("churn-prediction-preprocessor", mlflow.sklearn.load_model)