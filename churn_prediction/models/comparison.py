"""Champion-challenger comparison and model registration gate.

Compares a newly trained candidate model against the current champion
on the same held-out test set. If the candidate beats the champion by
at least PROMOTION_MARGIN AUC, it is registered in the MLflow model
registry under the 'challenger' alias for human review. The 'champion'
alias is never reassigned here, promotion to production is a manual
decision made in the MLflow UI.

If no champion exists yet (first automated retraining run), the
candidate is registered directly as 'champion'.
"""

import os
import logging

from churn_prediction.registry.mlflow_client import load_champion_model, load_champion_preprocessor

import xgboost as xgb
from sklearn.metrics import roc_auc_score
import mlflow

logger = logging.getLogger(__name__)


PROMOTION_MARGIN = 0.01


def _register_with_alias(client, model_uri, preprocessor_uri, alias):
    registered_model = mlflow.register_model(model_uri, "churn-prediction-model")
    registered_preprocessor = mlflow.register_model(preprocessor_uri, "churn-prediction-preprocessor")
    client.set_registered_model_alias("churn-prediction-model", alias, registered_model.version)
    client.set_registered_model_alias("churn-prediction-preprocessor", alias, registered_preprocessor.version)
    return registered_model.version


def compare_and_register(
    candidate_run_id: str,
    candidate_model,
    candidate_preprocessor,
    X_test_df,
    y_test,
) -> dict:
    """Compare candidate vs champion on the same held-out test set.

    If candidate beats champion by >= PROMOTION_MARGIN AUC, register
    candidate as 'challenger'. Never auto-promote to champion.

    Returns:
        Dict with keys:
            decision: "registered_as_challenger" | "rejected" | "no_champion"
            candidate_auc, champion_auc, delta, margin_required
            challenger_version (only when registered)
    """

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = mlflow.MlflowClient()
    candidate_model_uri = f"runs:/{candidate_run_id}/model"
    candidate_preprocessor_uri = f"runs:/{candidate_run_id}/preprocessor"

    try:
        model = load_champion_model()
        preprocessor = load_champion_preprocessor()
    except Exception:

        challenger_version = _register_with_alias(
            client, 
            candidate_model_uri, 
            candidate_preprocessor_uri, 
            "champion"
            )

        decision = "no_champion"

        logger.warning("No champion found. Registering candidate directly as champion (v%s).", challenger_version)

        return {
            "decision": decision,
            "candidate_auc": None,
            "champion_auc": None,
            "delta": None,
            "margin_required": PROMOTION_MARGIN,
            "challenger_version": challenger_version
            }

    X_test_champion = preprocessor.transform(X_test_df)      
    X_test_candidate = candidate_preprocessor.transform(X_test_df) 

    test_prediction_matrix_champion = xgb.DMatrix(X_test_champion)
    test_prediction_matrix_candidate = xgb.DMatrix(X_test_candidate)

    y_prob_champion = model.predict(test_prediction_matrix_champion)

    y_prob_candidate = candidate_model.predict(test_prediction_matrix_candidate)

    champion_auc = roc_auc_score(y_test, y_prob_champion)
    candidate_auc = roc_auc_score(y_test, y_prob_candidate)

    delta = candidate_auc - champion_auc

    logger.info("Champion AUC: %.4f | Candidate AUC: %.4f | Delta: %.4f", champion_auc, candidate_auc, delta)

    if candidate_auc >= champion_auc + PROMOTION_MARGIN:

        challenger_version = _register_with_alias(
            client, 
            candidate_model_uri, 
            candidate_preprocessor_uri, 
            "challenger"
            )

        decision = "registered_as_challenger"

        logger.info("Candidate beats champion by %.4f. Registering as challenger v%s.", delta, challenger_version)

    else:
        decision = "rejected"
        challenger_version = None

        logger.info("Candidate rejected. Delta %.4f below required margin %.4f.", delta, PROMOTION_MARGIN)



    return {
            "decision": decision,
            "candidate_auc": candidate_auc,
            "champion_auc": champion_auc,
            "delta": delta,
            "margin_required": PROMOTION_MARGIN,
            "challenger_version": challenger_version
            }