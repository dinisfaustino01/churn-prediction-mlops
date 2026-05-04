import logging

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost.core import Booster


logger = logging.getLogger(__name__)


def evaluate_model(
    model: Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate a trained XGBoost model on test data.

    Computes standard classification metrics plus Brier score (calibration)
    and a business-cost-weighted metric if `costs` is provided.

    Args:
        model: Trained XGBoost Booster.
        X_test: Transformed test feature matrix.
        y_test: Test target array.
        threshold: Probability threshold for converting probabilities to labels.

    Returns:
        Dict of metric name -> value (plus 'classification_report' as text).
    """
    dtest = xgb.DMatrix(X_test)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc_score": roc_auc_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(y_test, y_pred),
    }

    return results