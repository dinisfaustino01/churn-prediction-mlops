"""Pydantic data contract for raw customer input.

Defines the expected schema for a raw CSV row before any feature engineering.
Used by both the training and prediction pipelines to enforce data quality at
the ingestion boundary.
"""

import pandas as pd
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, ValidationError, field_validator


class RawCustomerRecord(BaseModel):
    """Schema for a single raw customer record as it arrives from the source CSV.

    Enforces types, allowed categorical values, and numeric constraints. Extra
    columns are forbidden so schema drift is caught immediately. TotalCharges is
    nullable to handle new customers with no billing history.
    """

    model_config = ConfigDict(extra="forbid")

    customerID: str
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]	
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)
    Churn: Optional[Literal["Yes", "No"]]

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the RawCustomerRecord contract.

    Args:
        df: Raw DataFrame loaded from the source CSV.

    Returns:
        The original DataFrame unchanged if all rows are valid.

    Raises:
        ValueError: If any row violates the schema, with a description of the
            first failing field and value.
    """
    
    records = df.to_dict(orient="records")

    try:
        TypeAdapter(list[RawCustomerRecord]).validate_python(records)
    except ValidationError as e:
        raise ValueError(f"Data contract violation:\n{e}") from e
    
    return df
