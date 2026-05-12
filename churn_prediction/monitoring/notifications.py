"""Slack notification helper for automated retraining outcomes.

Sends a message to a Slack channel after each retraining run,
summarising the comparison decision (new challenger registered,
candidate rejected, or first champion registered).

Requires SLACK_RETRAINING_WEBHOOK_URL to be set in the environment.
If not set, notifications are skipped and a warning is logged.
Slack failures never raise, they are logged and swallowed so a
Slack outage cannot fail the retraining pipeline.
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)


def _registered_as_challenger_payload(
        challenger_version: str, 
        candidate_auc: float, 
        champion_auc: float, 
        delta: float,
        margin_required: float) -> dict:
    
    message = (
        f"New challenger registered (v{challenger_version})\n"
        f"Candidate AUC: {candidate_auc:.4f}\n"
        f"Champion AUC: {champion_auc:.4f}\n"
        f"Delta: {delta:.4f} (margin: {margin_required:.4f})"
    )

    return {"text": message}


def _rejected_payload(
        candidate_auc: float, 
        champion_auc: float, 
        delta: float,
        margin_required: float) -> dict:
    
    message = (
        f"Candidate rejected\n"
        f"Candidate AUC: {candidate_auc:.4f}\n"
        f"Champion AUC: {champion_auc:.4f}\n"
        f"Delta: {delta:.4f} (margin: {margin_required:.4f})"
    )

    return {"text": message}


def _no_champion_payload() -> dict:
    
    message = (
        "First champion registered\n"
        "Candidate AUC: N/A (no baseline to compare)"
    )

    return {"text": message}


def notify_retraining_outcome(comparison_result: dict) -> None:
    """Send a Slack message summarising the retraining outcome.

    Reads the decision from comparison_result and routes to the
    appropriate payload builder. Posts to the configured Slack
    webhook and logs a warning if the request fails or Slack
    returns a non-200 status. Never raises.

    Args:
        comparison_result: Dict returned by compare_and_register().
            Must contain keys: decision, candidate_auc, champion_auc,
            delta, margin_required, challenger_version.
    """

    slack_webhook = os.getenv("SLACK_RETRAINING_WEBHOOK_URL")

    if not slack_webhook:
        logger.warning("SLACK_RETRAINING_WEBHOOK_URL not set. Skipping notification.")
        return

    decision = comparison_result["decision"]
    challenger_version = comparison_result["challenger_version"]
    candidate_auc = comparison_result["candidate_auc"]
    champion_auc = comparison_result["champion_auc"]
    delta = comparison_result["delta"]
    margin_required = comparison_result["margin_required"]

    if decision == "registered_as_challenger":

        payload = _registered_as_challenger_payload(
            challenger_version=challenger_version,
            candidate_auc=candidate_auc,
            champion_auc=champion_auc,
            delta=delta,
            margin_required=margin_required
        )

    elif decision == "rejected":

        payload = _rejected_payload(
            candidate_auc=candidate_auc,
            champion_auc=champion_auc,
            delta=delta,
            margin_required=margin_required
        )
    
    elif decision == "no_champion":

        payload = _no_champion_payload()
    
    try:
        response = requests.post(slack_webhook, json=payload, timeout=10)
        if response.status_code != 200:
            logger.warning("Slack returned %d: %s", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        logger.warning("Failed to send Slack notification: %s", e)
        return
    

