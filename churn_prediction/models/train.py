import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml
from xgboost.core import Booster

logger = logging.getLogger(__name__)


def load_params(params_path: str | Path) -> dict:
    """Load model parameters from a YAML file.

    Args:
        params_path: Path to the YAML model parameters file.

    Returns:
        Dictionary containing 'xgb_params', 'training', and 'costs' keys.
    """
    params_path = Path(params_path)
    logger.info("Loading model parameters from %s", params_path)

    with open(params_path) as f:
        params = yaml.safe_load(f)
    logger.info("Loaded parameters: %s", params)

    return params


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict,
) -> Booster:
    """Train an XGBoost model using the native API with early stopping.

    Args:
        X_train: Transformed training feature matrix.
        X_test: Transformed validation feature matrix.
        y_train: Training target array.
        y_test: Validation target array.
        params: Full config dict containing 'xgb_params' and 'training' keys.

    Returns:
        A trained XGBoost Booster object.
    """
    xgb_params = params["xgb_params"]
    num_boost_round = params["training"]["num_boost_round"]
    early_stopping_rounds = params["training"]["early_stopping_rounds"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, "train"), (dval, "validation")]

    logger.info(
        "Starting XGBoost training: %d boost rounds, early stopping at %d",
        num_boost_round,
        early_stopping_rounds,
    )

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10,
    )

    logger.info(
        "Training complete: best_iteration=%d, best_score=%.4f",
        booster.best_iteration,
        booster.best_score,
    )

    return booster