import numpy as np
import pandas as pd


def _check_probability_range(predictions_df: pd.DataFrame) -> dict:

    prob = predictions_df["churn_probability"]

    no_nans  = not prob.isna().any()
    no_infs  = not np.isinf(prob).any()
    in_range = prob.between(0, 1).all()

    #if no_nans and no_infs and in_range:
    #    status = "passed"
    #else:
    #    status = "failed"

    status = "passed" if no_nans and no_infs and in_range else "failed"

    return {"status": status, "severity": "hard"}


def _check_label_validity(predictions_df: pd.DataFrame) -> dict:
    label = predictions_df["predicted_label"]
    
    no_nans = not label.isna().any()
    valid_values = label.isin([0, 1]).all()
    
    #if no_nans and valid_values:
    #    status = "passed"
    #else:
    #    status = "failed"

    status = "passed" if no_nans and valid_values else "failed"
    
    return {"status": status, "severity": "hard"}


def _check_row_count_match(predictions_df: pd.DataFrame, input_df: pd.DataFrame) -> dict:

    expected = len(input_df)
    actual = len(predictions_df)

    #if expected == actual:
    #    status = "passed"
    #else:
    #    status = "failed"

    status = "passed" if expected == actual else "failed"

    return {
        "status": status,
        "severity": "hard",
        "expected": int(expected),
        "actual": int(actual),
    }


def _check_customer_id_uniqueness(predictions_df: pd.DataFrame) -> dict:

    customer_id = predictions_df["customer_id"]

    no_nans = not customer_id.isna().any()
    no_duplicated = not customer_id.duplicated().any()

    #if no_nans and no_duplicated:
    #    status = "passed"
    #else:
    #    status = "failed"

    status = "passed" if no_nans and no_duplicated else "failed"

    return {
        "status": status,
        "severity": "hard",
        "null_count": int(customer_id.isna().sum()),
        "duplicate_count": int(customer_id.duplicated().sum()),
    }


def _check_prediction_variance(predictions_df: pd.DataFrame, threshold: float = 0.01) -> dict:

    prob = predictions_df["churn_probability"]

    std = float(prob.std()) 

    status = "passed" if std > threshold else "failed"

    return {
        "status": status,
        "severity": "hard",
        "std": std,
        "threshold": threshold,
    }


def _check_churn_rate_band(predictions_df: pd.DataFrame, lower: float = 0.05, upper: float = 0.80,) -> dict:

    rate = predictions_df["predicted_label"].mean()
    
    status = "passed" if lower <= rate <= upper else "warning"

    return {
        "status": status,
        "severity": "soft",
        "value": float(rate),
        "band": [lower, upper],
    }


def run_all_checks(predictions_df: pd.DataFrame, input_df: pd.DataFrame) -> dict:
    """ Runs all data quality checks on the predictions DataFrame.

    Returns a dict keyed by check name, where each value is:
        {"status": "passed" | "failed", "severity": "hard" | "soft"}
    """

    check_results = {}

    probability_range_check = _check_probability_range(predictions_df)
    label_validity_check = _check_label_validity(predictions_df)
    row_count_match_check = _check_row_count_match(predictions_df, input_df)
    customer_id_uniqueness_check = _check_customer_id_uniqueness(predictions_df)
    prediction_variance_check = _check_prediction_variance(predictions_df)
    churn_rate_band_check = _check_churn_rate_band(predictions_df)

    check_results["probability_range"] = probability_range_check
    check_results["label_validity"] = label_validity_check
    check_results["row_count_match"] = row_count_match_check
    check_results["customer_id_uniqueness"] = customer_id_uniqueness_check
    check_results["prediction_variance"] = prediction_variance_check
    check_results["churn_rate_band"] = churn_rate_band_check


    return check_results


def aggregate_results(check_results: dict) -> dict:

    total = len(check_results)
    passed = sum(1 for r in check_results.values() if r["status"] == "passed")
    failed = sum(1 for r in check_results.values() if r["status"] == "failed")
    warnings = sum(1 for r in check_results.values() if r["status"] == "warning")
    hard_failures = [
        name for name, r in check_results.items()
        if r["status"] == "failed" and r["severity"] == "hard"
    ]

    all_passed = len(hard_failures) == 0

    return {
        "total_checks": int(total),
        "passed_checks": int(passed),
        "failed_checks": int(failed),
        "warnings": int(warnings),
        "hard_failures": hard_failures,
        "all_passed": bool(all_passed),
    }