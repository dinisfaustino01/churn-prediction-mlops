import logging
from pathlib import Path

import pandas as pd

from churn_prediction.data.input_schema import validate_dataframe

logger = logging.getLogger(__name__)


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load and validate a CSV file into a pandas DataFrame.

    Performs no column-specific transformations. Any cleaning or type coercion
    belongs in the preprocessing layer. Validates every row against the
    RawCustomerRecord contract before returning, so downstream code can assume
    the data is well-formed.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame with the raw contents of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any row fails the RawCustomerRecord contract validation.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    logger.info("Validating input schema")
    df = validate_dataframe(df)
    logger.info("Validated %d rows against contract.", len(df))

    return df
