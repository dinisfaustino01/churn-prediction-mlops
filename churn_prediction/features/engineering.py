import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the raw Telco DataFrame.

    Adds three features:
        - tenure_bucket: categorical bucketing of tenure based on EDA breakpoints.
        - contract_payment: interaction of Contract and PaymentMethod.
        - charges_per_tenure: average monthly spend over customer lifetime.

    Args:
        df: Raw DataFrame loaded from the source CSV.

    Returns:
        DataFrame with the three engineered columns added. Original columns
        are preserved; downstream drops are handled by prepare_raw_xy.
    """
    engineered_df = df.copy()

    bins = [-1, 12, 24, 48, float("inf")]
    labels = ["0-12", "13-24", "25-48", "49+"]
    engineered_df["tenure_bucket"] = pd.cut(
        engineered_df["tenure"], bins=bins, labels=labels, right=True
    ).astype(str)

    engineered_df["contract_payment"] = (
        engineered_df["Contract"] + "_" + engineered_df["PaymentMethod"]
    )

    total_charges = pd.to_numeric(engineered_df["TotalCharges"], errors="coerce")
    engineered_df["charges_per_tenure"] = total_charges / (engineered_df["tenure"] + 1)

    logger.info(
        "Engineered 3 features; DataFrame shape %s -> %s",
        df.shape,
        engineered_df.shape,
    )

    return engineered_df