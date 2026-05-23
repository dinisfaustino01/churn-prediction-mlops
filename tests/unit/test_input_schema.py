"""Unit tests for the raw input data contract.

Verifies that validate_dataframe() correctly enforces the Pydantic
schema: valid data passes, type/categorical/range violations fail,
extra columns are forbidden, and the custom empty-string validator
for TotalCharges works.
"""

import pytest

from churn_prediction.data.input_schema import RawCustomerRecord, validate_dataframe


def test_valid_data_passes(valid_df):
    result = validate_dataframe(valid_df)

    assert result is valid_df


def test_invalid_categorical_fails(valid_df):
    bad_df = valid_df.copy()
    bad_df.loc[0, "Contract"] = "Quarterly"

    with pytest.raises(ValueError):
        validate_dataframe(bad_df)


def test_extra_column_fails(valid_df):
    bad_df = valid_df.copy()
    bad_df["unknown_field"] = "junk"

    with pytest.raises(ValueError):
        validate_dataframe(bad_df)


def test_missing_required_column_fails(valid_df):
    bad_df = valid_df.drop(columns=["customerID"])

    with pytest.raises(ValueError):
        validate_dataframe(bad_df)


def test_wrong_type_fails(valid_df):
    bad_df = valid_df.copy()
    bad_df["tenure"] = bad_df["tenure"].astype(object)
    bad_df.loc[0, "tenure"] = "not_a_number"

    with pytest.raises(ValueError):
        validate_dataframe(bad_df)


def test_negative_numeric_fails(valid_df):
    bad_df = valid_df.copy()
    bad_df.loc[0, "tenure"] = -5

    with pytest.raises(ValueError):
        validate_dataframe(bad_df)


def test_empty_string_total_charges_becomes_none():
    record = RawCustomerRecord(
        customerID="7590-VHVEG",
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=1,
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=29.85,
        TotalCharges="",
        Churn="No",
    )
    assert record.TotalCharges is None
